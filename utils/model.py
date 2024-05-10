import os
import json

import torch
import numpy as np

# import hifigan
# from model import FastSpeech2, ScheduledOptim
# from model.modules import *
from model.fastspeech2 import *
# from model.loss import *
from model.optimizer import *


################################# @ train.py #####################################
def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config, device).to(device)
    if args is not None and args.restore_step:
        ckpt_path = os.path.join(train_config["path"]["ckpt_path"], "model_{}.pth".format(args.restore_step),)
        ckpt = torch.load(ckpt_path, map_location=device)
        # model.load_state_dict(ckpt["model"])
        model.load_state_dict(ckpt)

    if train:
        scheduled_optim = ScheduledOptim( model, train_config, model_config, 0)
        if args is not None and args.restore_step:
            scheduled_optim = ScheduledOptim( model, train_config, model_config, args.restore_step)
            ckpt_opim_path = torch.load(os.path.join(train_config["path"]["ckpt_path"], "optimizer_{}.pth".format(args.restore_step),), map_location=device)
            # scheduled_optim.load_state_dict(ckpt["optimizer"])
            scheduled_optim.load_state_dict(ckpt_opim_path)
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param



###################### Test:nvidia/HiFiGAN #########################
def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
        
    if name == "HiFi-GAN":
        vocoder, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan', trust_repo=True) ## Worked
        print(f"DownLoaded | NVIDIA's HiFi-GAN from torch hub | SR: 22050")
        vocoder.eval()
        vocoder.to(device)

        return vocoder, vocoder_train_setup, denoiser 


################################# @ train.py - HiFiGAN #####################################
def vocoder_infer(mels, model_config, preprocess_config, 
                  vocoder, vocoder_train_setup = None, denoiser = None, denoising_strength = 0.005,
                  lengths=None):
    
    name = model_config["vocoder"]["model"]
    
    with torch.no_grad():

        if name == "HiFi-GAN":
            wavs = vocoder(mels).float()
            wavs = denoiser(wavs.squeeze(1), denoising_strength)
            wavs = wavs.squeeze(1) 

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
