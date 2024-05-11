# Github [train.py]: https://github.com/ming024/FastSpeech2/blob/master/train.py

import os
import gc 
import yaml
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from utils.utils import to_device
import wandb 

from tqdm.auto import tqdm, trange

## from FastSpeech2 Modules
from dataset import *
from transformer.Constants import *
from transformer.Models import *
from transformer.Layers import *
from transformer.SubLayers import *

from utils.tools import * ## NOT GPU
from utils.model import * ## NOT GPU

from model.modules import *
from model.fastspeech2 import *
from model.loss import *
from model.optimizer import *

# Accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed 

# torchmallo & evaluate
from torchmalloc import *
from evaluate import *

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
s_ = Fore.CYAN
y_ = Fore.YELLOW
r_ = Fore.RED
g_= Fore.GREEN
sr_ = Style.RESET_ALL
m_ = Fore.MAGENTA 



def main(args, configs):

    ############
    # configs #
    ############
    preprocess_config, model_config, train_config = configs

    ############
    # SET SEED #
    ############
    set_seed(args.seed)

    ## Dataset
    train_dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    print("train_ds, valid_ds: Done")

    ###############
    # DataLoader  #
    ###############
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = args.group_size  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(train_dataset)
    train_loader = DataLoader(
                    train_dataset,
                    batch_size= batch_size * group_size,# train_config["optimizer"]["batch_size"] * 4, ## 16 * 4...?
                    shuffle=True,
                    collate_fn=train_dataset.collate_fn,)
    print("Train Loader: Done")

    assert batch_size * group_size < len(val_dataset)
    valid_loader = DataLoader(
                    val_dataset,
                    batch_size= batch_size * group_size,# train_config["optimizer"]["batch_size"] * 4, ## 16 * 4...?
                    shuffle=True,
                    collate_fn=val_dataset.collate_fn,)
    print("Valid Loader: Done")
    print()

    #######################
    # step: starting step #
    #######################
    if args.restore_step == 0:
        step = args.restore_step + 1
    else:
        step = args.restore_step
    print(f"STEP(START VALUE): {step}")

    ###############################
    # n_epochs: Numboer of Epochs #
    ###############################
    n_epochs = args.n_epochs


    ###############################################################
    # total_steps: Numboer of Epochs                              #
    # total_step = train_config["step"]["total_step"] ## original #
    # total_step = n_epochs * len(train_loader) * group_size # 3  #
    ###############################################################
    total_step = step + n_epochs * len(train_loader) * group_size if step != 1 else n_epochs * len(train_loader) * group_size
    print(f"Total STEP: {total_step }") ## 661080
    
    ###################
    # Steps per EPoch #
    ###################
    epoch_step = len(train_loader) * group_size
    print(f"STEPs per EPOCH: {epoch_step }") ## 661080

    print(f"N_EPOCHS: {n_epochs}") ## 840
    print(f"BATCH SIZE: {batch_size}") ## 16
    print(f"GROUP SIZE: {group_size}") ## 3
    print()

    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    print(f"GRAD ACC STEP: {grad_acc_step}")

    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    print(f"GRAD CLIP THRESH: {grad_clip_thresh}")

    log_step = train_config["step"]["log_step"]
    print(f"Log STEP: {log_step}")

    save_step = train_config["step"]["save_step"]
    print(f"save STEP: {save_step}")

    print(f"Model Save function Starts @ {args.save_start_step} step")
    print(f"Model Save function also acts @ LAST? : {args.save_at_last}")
    print()

    save_epoch = args.save_epochs
    save_epoch_steps =  save_epoch * len(train_loader) * group_size 
    print(f"save EPOCH: {save_epoch} (={save_epoch_steps})")

    # parser.add_argument('--synthesis_logging_epochs', type = int, default = 100, help="Sample logging Epochs")
    synth_step = train_config["step"]["synth_step"]
    print(f"Synth STEP: {synth_step}")

    synth_epoch = args.synthesis_logging_epochs
    synth_epoch_steps = synth_epoch * len(train_loader) * group_size 
    print(f"Synth EPOCH: {synth_epoch} (={synth_epoch_steps})")

    val_step = train_config["step"]["val_step"]
    print(f"VAL STEP: {val_step}")

    sampling_rate = preprocess_config["preprocessing"]["audio"][ "sampling_rate" ]
    sample_rate = preprocess_config["preprocessing"]["audio"][ "sampling_rate" ]
    print(f"sampling_rate(=sample_rate): {sampling_rate}")
    print()

    ###########################################################################################
    # Accelerator : https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation #
    ###########################################################################################

    ##### This is for find unused parameters for avoiding errors
    # from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[kwargs])

    accelerator = Accelerator(gradient_accumulation_steps = grad_acc_step, log_with="wandb", kwargs_handlers=[kwargs])

    ### Model
    model, optimizer = get_model(args, configs, device = accelerator.device, train=True)
    print("Model, Optimizer Done")
    ## model cuda? Check!
    print("MODEL IS CUDA? :", next(model.parameters()).is_cuda)

    num_param = get_param_num(model)
    print("Number of FastSpeech2 Parameters:", num_param)

    ## Loss Function
    loss_fn = FastSpeech2Loss(preprocess_config, model_config) # .to(device)
    print("Loss_fn: Done")
    print()

    # Load vocoder
    vocoder, vocoder_train_setup, denoiser = get_vocoder(model_config, accelerator.device)
    print("Vocoder Downloaded")
    
    ### To Device
    model, vocoder, denoiser, loss_fn, optimizer, train_loader, valid_loader = accelerator.prepare(model, vocoder, denoiser, loss_fn, optimizer, train_loader, valid_loader)
    print("Accelerate Prepared:")
    
    ## model cuda? Check!
    print("MODEL IS CUDA? :", next(model.parameters()).is_cuda)
    print()
    ###### Devices #######
    # accelerator.device

    ## wandb 
    ## you should log in (wandb) in cli 
    wandb_config= {
        'dataset': preprocess_config["path"]["df_path"],
        'model': "FastSpeech2",
        'vocoder': model_config["vocoder"]["model"],
        "n_epochs": n_epochs,
        'batch_size': batch_size,
        'grad_acc_step': grad_acc_step,
        'grad_clip_thresh': grad_clip_thresh,
        'total_step': int(total_step - step),
        'log_step': train_config["step"]["log_step"],
        'synth_step': [synth_step, synth_epoch, synth_epoch_steps],
        'val_step': val_step,
        'save_step': [save_step, save_epoch, save_epoch_steps],
        'sampling_rate': preprocess_config["preprocessing"]["audio"][ "sampling_rate" ]
    }

    ## wandb init with accelerator
    accelerator.init_trackers(
        project_name= args.project_name, # "FastSpeech2_german",
        config= wandb_config,
        init_kwargs={
            "wandb": {
                # "project": "FastSpeech2_german",
                "job_type": 'Train',
                "tags": ["TTS", "FastSpeech2", "HiFiGAN"],
                "name": args.try_name, #"[test2] train_eval_wandb_colorama",
                }
            },
        )
    ##################
    # Training Start #
    ##################

    # HiFiGAN - Denoiser
    denoising_strength = args.denoising_strength # 0.005
    print(f"Vocoder(=HiFi-GAN)'s denoising_strength: {args.denoising_strength}")
    print(f"Vocoder(=HiFi-GAN)'s denoising_strength: {denoising_strength}")

    ## This is for wandb Audio Added
    preview_table = wandb.Table(columns = ['STEP', 'FROM', 'Label Speech', 'Predicted Speech'])

    for epoch in trange(n_epochs, desc='Epoch'):
        print(f"{g_}========================== STEPS: [{step}/{total_step}] =========================={sr_}", end ="\n" )
        print(f"{s_}========================== EPOCH: [{epoch}/{n_epochs}] =========================={sr_}", end ="\n" )
        print("STARTING")

        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            inner_bar = tqdm(train_loader, desc="Step", position= 0)
            for batchs in inner_bar:
                for batch in batchs:
                    with accelerator.accumulate(model):

                        batch = to_device(batch, accelerator.device)

                        # Forward
                        output = model(*(batch[2:]))

                        # Cal Loss
                        losses = loss_fn(batch, output)
                        total_loss = losses[0]

                        # Backward: Original 
                        # total_loss = total_loss / grad_acc_step
                        # total_loss.backward()
                        
                        accelerator.backward(total_loss)

                        # Gradient Clipping
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                        # Update weights
                        optimizer.step_and_update_lr()
                        optimizer.zero_grad()

                        
                        ##################
                        # Wandb: Logging #
                        ##################
                        if step % log_step == 0:
                        # if step % (100 * 3) == 0:
                            losses = [l.item() for l in losses]

                            ## wandb logging
                            accelerator.log({"Train/Total_loss": losses[0],
                                            "Train/Mel_loss" : losses[1],
                                            "Train/Mel_PostNet_loss" : losses[2], ## T4MR_10 this is mel_loss
                                            "Train/Pitch_loss" : losses[3],
                                            "Train/Energy_loss": losses[4],
                                            "Train/Duration_loss": losses[5],
                                            }, step=step)
                        
                         

                        ############################
                        # Syntheize Speech Sample  #
                        ############################
                        if step % synth_step == 0 or step % synth_epoch_steps == 0:

                            fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                                batch,
                                output,
                                model_config,
                                preprocess_config,
                                vocoder,
                                vocoder_train_setup, 
                                denoiser, 
                                denoising_strength,
                            )
                            # mel spectrogram plot 
                            accelerator.log({ 'Train/Mel_SPectrogram' : wandb.Image(fig) })

                            ## wandb Add Audio 
                            label_audio = wandb.Audio(wav_reconstruction, sample_rate = 22050)
                            predict_audio = wandb.Audio(wav_prediction, sample_rate = 22050)
                            preview_table.add_data(step, 'TRAIN', label_audio, predict_audio)


                        ############
                        # Evaluate #
                        ############
                        if step % val_step == 0:
                            model.eval()
                            message, fig, sample_audios_from_vals = evaluate_fn( model, 
                                                                                 step, 
                                                                                 configs,
                                                                                 True, # Logging = True
                                                                                 vocoder, 
                                                                                 vocoder_train_setup, 
                                                                                 denoiser, 
                                                                                 accelerator, 
                                                                                 device = accelerator.device,
                                                                                 sample_needs = True
                                                                                 )
                            # mel-spectrogram plot
                            accelerator.log({'Eval/Mel_SPectrogram' : wandb.Image(fig) })

                            ## wandb Add Audio 
                            label_audio = wandb.Audio(sample_audios_from_vals[0], sample_rate = 22050)
                            predict_audio = wandb.Audio(sample_audios_from_vals[1], sample_rate = 22050)
                            preview_table.add_data(step, 'EVAL', label_audio, predict_audio)
                            
                
                            model.train()


                        ############################
                        # SAVE model and optimizer #
                        ############################
                        if (step >= args.save_start_step) and (step % save_step == 0 or step % save_epoch_steps == 0):

                            accelerator.wait_for_everyone()
                            
                            # Unwrap: model
                            unwrapped_model = accelerator.unwrap_model(model)
                            # Unwrap: optimizer
                            unwrapped_optmizer = accelerator.unwrap_model(optimizer) 
                        
                            # Use accelerator.save()
                            # save_path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step),)
                            save_path = train_config["path"]["ckpt_path"]

                            # state_dict; state_dict = model.module.state_dict() 
                            unwrapped_model_state_dict = unwrapped_model.state_dict() 
                            accelerator.save(unwrapped_model_state_dict, save_path + f"/model_{step}.pth")
                            accelerator.save(unwrapped_optmizer._optimizer.state_dict(), save_path + f"/optimizer_{step}.pth")
                            print(f"Model SAVED @ {step} of {epoch}")
                            print()

                        step += 1

            print(f"{s_} {epoch} EPOCH is completed{sr_}")


        print(f"{b_}========================== Training_EP:[{epoch}/{n_epochs}]_STEP:[{step}] =========================={sr_}", end ="\n" )
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
                )
            )
        
        print(f"{b_}============================================================================{sr_}", end ="\n" )
        print()
    
    #########################################
    # In the End: SAVE model and optimizer #
    #########################################
    if args.save_at_last == "True":

        accelerator.wait_for_everyone()
        
        # Unwrap: model
        unwrapped_model = accelerator.unwrap_model(model)
        # Unwrap: optimizer
        unwrapped_optmizer = accelerator.unwrap_model(optimizer) 
    
        # Use accelerator.save()
        # save_path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step),)
        save_path = train_config["path"]["ckpt_path"]

        # state_dict; state_dict = model.module.state_dict() 
        unwrapped_model_state_dict = unwrapped_model.state_dict() 
        accelerator.save(unwrapped_model_state_dict, save_path + f"/model_{step}.pth")
        accelerator.save(unwrapped_optmizer._optimizer.state_dict(), save_path + f"/optimizer_{step}.pth")
        
        print(f"Model SAVED @ LAST")
        print()

    ## Sample Audio Added
    wandb.log({'Visualization': preview_table})
    accelerator.log({'Sample Speeches': preview_table})
    print(f"{b_}========================== Trainig is Over =========================={sr_}", end ="\n" )

    ## End of wandb Logging
    accelerator.end_training()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument('--seed', type = int, default = 2023, help="Seed")
    parser.add_argument('--n_epochs', type = int, default = 750, help="Epochs")

    parser.add_argument('--save_epochs', type = int, default = 30, help="Model SAVE Epochs")
    parser.add_argument('--save_start_step', type = int, default = 50000, help="Model SAVE function started when step reached this number")
    parser.add_argument('--save_at_last', type = str, default = "True", help="Model SAVE After last step training.")

    parser.add_argument('--synthesis_logging_epochs', type = int, default = 50, help="Sample logging Epochs")
    parser.add_argument('--denoising_strength', type = float, default = 0.005, help="HiFiGAN's Denoiser - denoising_strength")
    # denoising_strength = 0.005

    # parser.add_argument('--batch_size', type = int, default = 48, help="Batch Size")
    parser.add_argument('--group_size', type = int, default = 1, help="Group Size")
    parser.add_argument('--project_name', type = str, default = "fastpeech2_a", help="PROJECT NAME IN WANDB")
    parser.add_argument('--try_name', type = str, default = "T_01", help="Naming tries of PROJECT IN WANDB")

    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        # required=True,
        default = "./config/LJSpeech/preprocess.yaml",
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        # required=True, 
        default = "./config/LJSpeech/model.yaml",
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t", 
        "--train_config", 
        type=str, 
        # required=True, 
        default = "./config/LJSpeech/train.yaml",
        help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    # preprocess_config = yaml.load(open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    # train_config = yaml.load(open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader)
    # model_config = yaml.load(open("./config/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader)

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)

    ## wandb login --relogin '<your-wandb-api-token>'
    ## accelerate config
    ## CUDA_VISIBLE_DEVICES=2,3 accelerate launch train.py --n_epochs 800 --save_start_step 12000 --save_epochs 20 --synthesis_logging_epochs 20 --try_name T_01_LJSpeech
