# fastspeech2_a
 TTS(= Text-To-Speech) Model for studying and researching. This Repository is mainly based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and modified or added some codes. 

Additionally, for faster or more efficient training, I added some codes from:    
- 🤗 `accelerate`: `multi-gpu` - Trained on 2 x NVIDIA GeForece RTX 4090 GPUs
- ✍🏻️ `wandb` [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/fastpeech2_a)
  - `wandb` instead of `Tensorboard`. `wandb` is compatible with 🤗`accelerate` and with :fire:`pytorch`.
  - <details>
    <summary> dashboard screenshots </summary>
    <div>
    <img src="/imgs/스크린샷 2024-05-11 오후 10.18.04.png" width="83%"></img>
    <img src="/imgs/스크린샷 2024-05-11 오후 10.17.47.png" width="83%"></img>
    </div>
    </details>
- [`torchmalloc.py`](https://github.com/elu-lab/FASTSPeech2/blob/main/torchmalloc.py) and :rainbow:[`colorama`](https://github.com/tartley/colorama) can show your resource in real-time (during training)
- [`noisereduce`](https://github.com/timsainb/noisereduce) is available when you run `preprocessor.py`.
  - `Non-Stataionary Noise Reduction`
  - `prop_decrease` can avoid data-distortion. (0.0 ~ 1.0)
  - Actually, NOT USED.
- :fire:[`[Pytorch-Hub]NVIDIA/HiFi-GAN`](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/): used as a vocoder.

## Dataset
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
  - `Language`: English :us:
  - `Speaker`: Single Speaker
  - `sample_rate`: 22.05kHz

## Colab notebooks (Examples):
Theses codes are written and run in my vscode environment.
- **(EXAMPLE_Jupyternotebook) Synthesis.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FOqWONEY26x6HWIiakvONxS_k2-zDjCR?usp=sharing)     
- **(EXAMPLE_CLI) Synthesis.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pW6ahqktYDLivV3bsuMM7Tl1Q8vOYqBm?usp=sharing)     
- **More_Examples_Synthesized.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eOomvRbp5lT79e7j5xkoNpapkLSu4U_3?usp=sharing)

## Preprocess
 This `preprocess.py` can give you the pitch, energy, duration and phones from `TextGrid` files. 
```
python preprocess.py config/LJSpeech/preprocess.yaml 
```

## Train
 First, you should log-in wandb with your token key in CLI. 
```
wandb login --relogin '<your-wandb-api-token>'
```

 Next, you can set your training environment with following commands. 
```
accelerate config
```

 With this command, you can start training. 
```
accelerate launch train.py --n_epochs 800 --save_start_step 12000 --save_epochs 20 --synthesis_logging_epochs 20 --try_name T_01_LJSpeech
```

Also, you can train your TTS model with this command.
```
CUDA_VISIBLE_DEVICES=2,3 accelerate launch train.py --n_epochs 800 --save_start_step 12000 --save_epochs 20 --synthesis_logging_epochs 20 --try_name T_01_LJSpeech
```

## Synthesize
you can synthesize speech in CLI with this command: 
```
python synthesize.py --raw_texts <Text to syntheize to speech> --restore_step 100000
```
You can refer to `Colab notebooks (Examples)` above if you wanna synthesize.    
Also, you can check this [jupyter-notebook](https://github.com/elu-lab/fastspeech2_a/blob/main/(EXAMPLE_CLI)%5BT_01%5D%20synthesis.ipynb). 
 <img src="/imgs/스크린샷 2024-05-11 오후 10.16.17.png" width="83%"></img>


## References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HGU-DLLAB/Korean-FastSpeech2-Pytorch
Public](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch)
- [pytorch_hub/nvidia/HIFI-GAN](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/)
- [🤗 Accelerate](https://huggingface.co/docs/accelerate/package_reference/accelerator)
- [🤗 Accelerate(Github)](https://github.com/huggingface/accelerate) 
- [🤗 huggingface/peft/.../peft_lora_clm_accelerate_ds_zero3_offload.py](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py)
