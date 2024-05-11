# fastspeech2_a
 TTS(= Text-To-Speech) Model for studying and researching. This Repository is mainly based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and modified or added some codes. We use [AI-HUB: Multi-Speaker-Speech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=542) dataset and [MLS(=Multilingual LibriSpeech)](https://www.openslr.org/94/) dataset for training. 

## Dataset
- [LJSpeech)](https://keithito.com/LJ-Speech-Dataset/)
  - `Language`: English :us:
  - `sample_rate`: 22.05kHz



## Features(Differences?)
- 🤗[`accelerate`](https://github.com/huggingface/accelerate) can allow `multi-gpu` training easily: Trained on 2 x NVIDIA GeForece RTX 4090 GPUs. 
- [`torchmalloc.py`](https://github.com/elu-lab/FASTSPeech2/blob/main/torchmalloc.py) and :rainbow:[`colorama`](https://github.com/tartley/colorama) can show your resource in real-time (during training) like below:
  <details>
  <summary> example </summary>
  <div>
   Referred: <a href="https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py">🤗huggingface/peft/ .. example</a> <br/>   
  <img src="/imgs/스크린샷 2023-11-20 오후 11.25.09.png" width="60%"></img>
  </div>
  </details>
- [`noisereduce`](https://github.com/timsainb/noisereduce) is available when you run `preprocessor.py`.
  - `Non-Stataionary Noise Reduction`
  - `prop_decrease` can avoid data-distortion. (0.0 ~ 1.0)
- `wandb` instead of `Tensorboard`. `wandb` is compatible with 🤗`accelerate` and with :fire:`pytorch`.
- :fire:[`[Pytorch-Hub]NVIDIA/HiFi-GAN`](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/): used as a vocoder.
  

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
accelerate launch train.py --n_epochs 600 --save_epochs 50 --synthesis_logging_epochs 30 --try_name T_01_LJSpeech
```

Also, you can train your TTS model with this command.
```
CUDA_VISIBLE_DEVICES=2,3 accelerate launch train.py --n_epochs 600 --save_epochs 50 --synthesis_logging_epochs 30 --try_name T_01_LJSpeech
```

## Synthesize
you can synthesize speech in CLI with this command: 
```
python synthesize.py --raw_texts <Text to syntheize to speech> --restore_step 53100
```
Also, you can check this [jupyter-notebook](https://github.com/elu-lab/FASTSPeech2/blob/main/synthesize_example.ipynb) when you try to synthesize.
 <img src="/imgs/스크린샷 2023-11-20 오후 9.33.27.png" width="83%"></img>


## References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HGU-DLLAB/Korean-FastSpeech2-Pytorch
Public](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch)
- [pytorch_hub/nvidia/HIFI-GAN](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/)
- [🤗 Accelerate](https://huggingface.co/docs/accelerate/package_reference/accelerator)
- [🤗 Accelerate(Github)](https://github.com/huggingface/accelerate) 
- [🤗 huggingface/peft/.../peft_lora_clm_accelerate_ds_zero3_offload.py](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py)
