## Github[preprocessor.py]: https://github.com/ming024/FastSpeech2/blob/master/preprocessor/preprocessor.py
## Github[config][preprocess]: https://github.com/ming024/FastSpeech2/blob/master/config/LibriTTS/preprocess.yaml

import os
import random
import json

import tgt
import librosa
import numpy as np
import pandas as pd
import pyworld as pw

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
# import torchaudio

from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
# from noisereduce.generate_noise import band_limited_noise


# import audio as Audio
from audio.stft import *
from audio.tools import *

############################# Preprocessor Class ################################ 
class Preprocessor:
    def __init__(self, config):
        self.config = config
        # self.in_dir = config["path"]["raw_path"] # "./raw_data/LibriTTS" ## dont need this >> self.df_path
        self.out_dir = config["path"]["preprocessed_path"] # prac dir
        self.val_size = config["preprocessing"]["val_size"] # 512
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"] # 22050
        self.hop_length = config["preprocessing"]["stft"]["hop_length"] # 256

        ## Inevitable
        self.df_path = config["path"]["df_path"]
        self.df = pd.read_csv(self.df_path)

        ## lang
        self.lang = config["preprocessing"]["text"]["language"] ## 'gernman' @ sample-test

        ## for PRINT
        self.pitch_feature = config["preprocessing"]["pitch"]["feature"]
        self.energy_feature = config["preprocessing"]["energy"]["feature"]

        self.pitch_normal = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normal = config["preprocessing"]["energy"]["normalization"]


        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        # self.STFT = Audio.stft.TacotronSTFT(
        self.STFT = TacotronSTFT(
                config["preprocessing"]["stft"]["filter_length"], # 1024
                config["preprocessing"]["stft"]["hop_length"], # 256
                config["preprocessing"]["stft"]["win_length"], # 1024
                config["preprocessing"]["mel"]["n_mel_channels"], # 80
                config["preprocessing"]["audio"]["sampling_rate"], # 22050
                config["preprocessing"]["mel"]["mel_fmin"], # 0
                config["preprocessing"]["mel"]["mel_fmax"], # 8000
            )
        
        self.denoiser = config["preprocessing"]["audio"]["denoiser"]
        self.denoiser_prop_decrease = config["preprocessing"]["audio"]['prop_decrease'] # 0
        self.denoiser_thresh_n_mult_nonstationary = config["preprocessing"]["audio"]['thresh_n_mult_nonstationary']# 2


    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
    

    ### Robust Scaler Thing?
    def remove_outlier(self, values):
        values = np.array(values)

        if np.isnan(values).any():
            p25 = np.nanpercentile(values, 25)
            p75 = np.nanpercentile(values, 75)
        else:   
            p25 = np.percentile(values, 25)
            p75 = np.percentile(values, 75)
            
        # p25 = np.percentile(values, 25)
        # p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]
  
    def get_alignment(self, tier):
        # tier = textgrid.get_tier_by_name("phones")
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # s, e, p  = Interval(0.49, 0.57, "ɪ")

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    # p = "ɪ"
                    # sil_phones = ["sil", "sp", "spn"]
                    continue
                else:
                    # s = 0.49
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                # p = "ɪ"
                phones.append(p)
                end_time = e # 0.57
                end_idx = len(phones)
            else:
                # For silent phones
                # p = "ɪ"
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length) # np.roun(e * 22050 / 1024)  # e = 0.57 ## self.sampling_rate, self.hop_length
                    - np.round(s * self.sampling_rate / self.hop_length)  # -np.roun(s * 22050 / 1024)  # s = 0.49
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        return phones, durations, start_time, end_time #  (124(=길이), 124(=길이), 0.49, 10.58)
    
    def process_utterance(self, speaker, basename, wav_name, sentence, tg_path):
        wav_path = wav_name
        # wav_path = os.path.join(CUDA_VISIBLE_DEVICES=2, speaker, "{}.wav".format(basename))
        # text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        # tg_path = os.path.join(
        #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        # )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        ) ## (124(=길이), 124(=길이), 0.49, 10.58)

        ## TEXT = {ɪ n d ɔʏ tʃ l a n t p ʁ ɔ tʰ ɛ s t iː ɐ tʰ n̩ ...
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        ## Read and trim wav files

        ## Librosa Load
        wav, org_sr = librosa.load(wav_path, sr =None) ## numpy # (184800,)

        ## Denoise Option - Non-Stationary Noise Reduction
        if self.denoiser == "non-stationary-noise-reduction":
            wav = nr.reduce_noise(y = wav, 
                                  sr= org_sr, 
                                  prop_decrease = self.denoiser_prop_decrease, # 0, 
                                  thresh_n_mult_nonstationary= self.denoiser_thresh_n_mult_nonstationary , # 2,
                                  stationary=False)

        ## Resample
        wav = librosa.resample(wav, orig_sr = org_sr, target_sr = self.sampling_rate )

        wav = wav[ ## sampling_rate = 22050
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32) # (173996,)

        # Read raw text
        # with open(text_path, "r") as f:
        #     raw_text = f.readline().strip("\n")
        raw_text = sentence.strip("\n")

        # Compute fundamental frequency
        # pw: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
        pitch, t = pw.dio( ## raw pitch extractor
            wav.astype(np.float64),
            self.sampling_rate, ## 22050
            frame_period=self.hop_length / self.sampling_rate * 1000, ## frame_period = 1024/ 22050 * 1000 # ms
            ) 
        ## pitch, t Shape = # 170, (170,)
        ## sum(duration) = 188
        ## pitch refinement
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        ## pitch
        pitch = pitch[: sum(duration)] # (170,)
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        # mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram, energy = get_mel_from_wav(wav, self.STFT)
        # Shape/Length: (80, 170) (170,)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        # Shape/Length: (80, 170) (170,), 124


        if self.pitch_phoneme_averaging: # True
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0] # 170 -> 67
            # interp1d : https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
            interp_fn = interp1d(
                nonzero_ids, # 67
                pitch[nonzero_ids], # 67
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]), # (68.63053143749191, 278.22930664321774)
                bounds_error=False, 
                # kind = 'linear' # Default is ‘linear’.
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        # if self.energy_phoneme_averaging:
        if self.energy_phoneme_averaging: # True
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )
            
    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        # print()
        print(f"Lang: {self.lang}")
        print(f"SPEAKER_ID: {self.df.speaker_id.values[0]}")
        print(f"Noise Reduction? : {self.denoiser}")
        print(f"Sampling Rate: -[Resampled]-> {self.sampling_rate}")
        # print()
        print(f"PITCH AVERAGING: {self.pitch_feature}")
        print(f"ENERGY AVERAGING: {self.energy_feature}")
        # print()
        print(f"PITCH Normalization: {self.pitch_normal}")
        print(f"ENERGY Normalization: {self.energy_normal}")
        # print()
        # print(f"INPUT DIR: {self.in_dir}")
        print(f"INPUT DIR: {self.df_path}")
        print(f"OUTPUT DIR: {self.out_dir}")
        ## for PRINT
        # self.pitch_feature = config["preprocessing"]["pitch"]["feature"]
        # self.energy_feature = config["preprocessing"]["energy"]["feature"]

        # self.pitch_normal = config["preprocessing"]["pitch"]["normalization"]
        # self.energy_normal = config["preprocessing"]["energy"]["normalization"]


        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
 
        # lang = 'german'
        # tg_base_path = f"/data/speech-data/mls-align/mls_{lang}_opus/train/"
      
        # self.lang = 'german'
        tg_base_path = f"/nfs/data/speech-data/mls-align/mls_{self.lang}_opus/train/"

        # for i, speaker in enumerate(tqdm(os.listdir(in_dir))):
        # for i, row in self.df.iterrows():
        for i, row in tqdm(self.df.iterrows(), total = self.df.shape[0]):
            
            speaker = str(row['speaker_id'])
            speakers[speaker] = i ## SPEAKER_ID: STR # i = index number
            wav_name = row['audio_path']
            # basename = wav_name.split(".")[0] ## This looks like file_id
            basename = row['file_id']
            # tg_path = os.path.join(
            #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
            # )
            sentence = row['sentence']

            if self.lang == "kor_ai_hub" or self.lang == "en_ljs":
                tg_path = row['tg_path']
            else:
                tg_path = os.path.join(
                    tg_base_path, speaker, "{}.TextGrid".format(basename)
                )
            # print(basename, speaker) 
            # 10087_10388_000000 10087

            # print(tg_path, wav_name) 
            # /data/speech-data/mls-align/mls_german_opus/train/10087/10087_10388_000000.TextGrid /
            # # data/speech-data/mls/mls_german_opus/train/audio/10087/10388/10087_10388_000000.opus
            # break

            if os.path.exists(tg_path): ## True
                ret = self.process_utterance(speaker, basename, wav_name, sentence, tg_path)
                if ret is None:
                    continue
                else:
                    info, pitch, energy, n = ret
                out.append(info)

            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

            n_frames += n
            # if i == 2:
            #     break
    
        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out
