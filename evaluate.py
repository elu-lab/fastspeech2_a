
import os
import yaml
import argparse
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

import accelerate
from accelerate import Accelerator

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log_fn, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from torchmalloc import *

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
s_ = Fore.CYAN
y_ = Fore.YELLOW
r_ = Fore.RED
g_= Fore.GREEN
sr_ = Style.RESET_ALL
m_ = Fore.MAGENTA 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.inference_mode()
def evaluate_fn(model, 
                step, 
                configs, 
                logging=True, ## Modified 09.27 - Logger -> Logging: True
                vocoder=None, 
                vocoder_train_setup = None, 
                denoiser = None, 
                accelerator= None, 
                device = device, 
                sample_needs = False
                ):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    val_dataset = Dataset( "val.txt", preprocess_config, train_config, sort=False, drop_last=False )
    batch_size = train_config["optimizer"]["batch_size"]
    valid_loader = DataLoader(
                    val_dataset,
                    batch_size= batch_size,
                    shuffle=True,
                    collate_fn=val_dataset.collate_fn,)
    print("Valid Loader: Done")

    # Get loss function
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

    sample_audios = []
    denoising_strength = 0.005
    # logger = val_logger

    with TorchTracemalloc() as tracemalloc:
        model.eval()

        # Evaluation
        loss_sums = [0 for _ in range(6)]

        inner_bar = tqdm(valid_loader, total=len(valid_loader), desc="Evaluate", position=1)
        for batchs in inner_bar:
        # for batchs in valid_loader:
            for batch in batchs:
                batch = to_device(batch, accelerator.device)
                with torch.no_grad():

                    # Forward
                    output = model(*(batch[2:]))

                    # Cal Loss
                    losses = loss_fn(batch, output)
                    for i in range(len(losses)):
                            loss_sums[i] += losses[i].item() * len(batch[0])

        loss_means = [loss_sum / len(val_dataset) for loss_sum in loss_sums]
        message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(*([step] + [l for l in loss_means]))

        ## wandb logging
        accelerator.log({"Eval/Total_loss": losses[0],
                        "Eval/Mel_loss" : losses[1],
                        "Eval/Mel_PostNet_loss" : losses[2],
                        "Eval/Pitch_loss" : losses[3],
                        "Eval/Energy_loss": losses[4],
                        "Eval/Duration_loss": losses[5],
                        }, step = step)
        
        if logging:
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
            
            sample_audios += [wav_reconstruction, wav_prediction]
            ## Removed Tensorboard logging codes 
        
        print(f"{g_} Eval[{step}]: {message} {sr_}")
        print(f"Validation @ {step}: Firnished")

    if accelerate is not None:
        print(f"{y_}==================== Validation_STEP:[{step}]====================", end ="\n" )
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
        print(f"{y_}================================================{sr_}", end ="\n" )
        print()


    if sample_needs:
        return message, fig, sample_audios
    else:
        return message, fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # from accelerate import Accelerator
    accelerator = Accelerator()

    # Get model
    model = get_model(args, configs, device, train=False)
    
    # Get Vocoder: HiFiGAN
    vocoder, vocoder_train_setup, denoiser = get_vocoder(model_config, torch.device('cpu'))

    ### To Device
    model, vocoder, denoiser= accelerator.prepare(model, vocoder, denoiser)
    print("Accelerate Prepared:")
    
    message = evaluate_fn( model, 
                           args.restore_step, 
                           configs, 
                           logging=True, 
                           vocoder=vocoder, 
                           vocoder_train_setup = vocoder_train_setup, 
                           denoiser = denoiser, 
                           accelerator= accelerator, 
                           device =device)
    print(message)
