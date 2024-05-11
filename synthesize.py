import re
import os
import gc
import yaml
import argparse
from string import punctuation

# https://github.com/matplotlib/matplotlib/issues/25506
# import maplitlib Before import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from matplotlib.patches import Rectangle

import librosa
import librosa.display
import IPython.display as ipd

from pathlib import Path
from PIL import Image

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.tools import * ## NOT GPU
from utils.model import * ## NOT GPU
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text,
                       preprocess_config = yaml.load(open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
                       ):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence), phones


def convert_to_inputs(raw_texts,
                      preprocess_config = yaml.load(open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
                      ):

    # 1) Speaker_id
    speakers = np.array([13099]) # speaker: 100

    # 2) G2P
    sequence, phones = preprocess_english(raw_texts, preprocess_config)
    print("Sequence: ",sequence)
    print("Phones: ", phones)
    print(sequence.shape)
    texts = sequence.reshape(1, -1)
    print(texts.shape)
    print(texts)
    print()

    text_lens = np.array([len(texts[0])])
    print(text_lens)
    print()
    
    ids = raw_texts[0]
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    return batchs, phones

@torch.inference_mode()
def synthesize_fn(model, 
                  # step, # sample_args.restore_step, 
                  configs, # configs = (preprocess_config, model_config, train_config) 
                  batchs, 
                  control_values, 
                  device, 
                  vocoder, 
                  vocoder_train_setup=None,
                  denoiser = None, 
                  denoising_strength=0.005
                  ):
    
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    model.eval()
    with torch.no_grad():
        for batch in batchs:
            cuda_batch = to_device(batch, device)
            # Forward
            output = model(*(cuda_batch[2:]), 
                            p_control=pitch_control, 
                            e_control=energy_control, 
                            d_control=duration_control
                            )
            
            # Synthesize
            synth_samples(cuda_batch, 
                          output, 
                          model_config, 
                          preprocess_config, 
                          train_config["path"]["result_path"],
                          vocoder, 
                          vocoder_train_setup, 
                          denoiser, 
                          denoising_strength
                          )
            
# To Device
def syn(raw_texts, 
        model,
        configs, 
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
        control_values = (1.0, 1.0, 1.0)
        ):

    # Configs
    preprocess_config, model_config, train_config = configs 

    # Load vocoder
    vocoder, vocoder_train_setup, denoiser = get_vocoder(model_config, device)
    
    # control_values = args.pitch_control, args.energy_control, args.duration_control
    control_values = control_values 
    print(f"Vocoder Downloaded")
    print(f"CONTROL VALUES: {control_values}")

    # device
    model = model.to(device)
    denoiser = denoiser.to(device)
    vocoder = vocoder.to(device)
    print("Accelerate Prepared:")
    
    # Convert
    batchs, phones = convert_to_inputs(raw_texts, preprocess_config)
    ids = batchs[0][0]
    
    # Synthesize
    synthesize_fn(model, 
                  # step, # sample_args.restore_step, 
                  configs, # configs = (preprocess_config, model_config, train_config) 
                  batchs, 
                  control_values, 
                  device, 
                  vocoder, 
                  vocoder_train_setup, 
                  denoiser, 
                  0.0025
                  )
    print("synthesized")

    # Saved Paths: AUDIO, MEL SAVE PATH
    audio_result_path = train_config["path"]["result_path"] + f"/{ids}.wav"
    mel_result_path = train_config["path"]["result_path"] + f"/{ids}.png"

    return ids, raw_texts, phones, audio_result_path, mel_result_path 


def main(args, configs):

    preprocess_config, model_config, train_config = configs

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load Model 
    model_id = args.restore_step # default: 100000 # 84160
    model = get_model(args, configs, device = device, train=False)
    print(f"{model_id}Model Loaded", end ="\n")

    raw_texts = args.raw_texts 
    # raw_texts =  "My name is Ro Hoon and I am researching text-to-speech in my lab."
    ids, raw_texts, phones, audio_result_path, mel_result_path = syn(raw_texts, 
                                                                     model = model,
                                                                     configs= configs, 
                                                                     device = device, 
                                                                     control_values = (1.0, 1.0, 1.0 ) )
    # print("Synthesize Completed")
    # print(f"MODEL ID: {model_id}")
    # print(f"SENTENCE: {raw_texts}")
    # print(f"Pure Length {len(raw_texts)}")
    # print(f"Phones: {phones}")
    # print(audio_result_path)
    # print(mel_result_path)
    # Image.open(mel_result_path).convert("RGB")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_texts', type = str, default = "My name is Ro Hoon and I am researching text-to-speech in my lab.", help="text to synthesize")
    parser.add_argument("--restore_step", type=int, default= 100000)

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
