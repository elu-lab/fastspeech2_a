# Github[dataset.py]: https://github.com/ming024/FastSpeech2/blob/master/dataset.py
# Github[config.LibriTTS.preprocess.yaml]: https://github.com/ming024/FastSpeech2/blob/master/config/LibriTTS/preprocess.yaml
# Github[config.LibriTTS.train.yaml] : https://github.com/ming024/FastSpeech2/blob/master/config/LibriTTS/train.yaml
# Github[utils.tools]: https://github.com/ming024/FastSpeech2/blob/master/utils/tools.py#L265

import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(self, filename, preprocess_config, train_config, sort=False, drop_last=False):
        self.dataset_name = preprocess_config["dataset"] # "LibriTTS" 
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"] #'/home/heiscold/prac/'
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"] #["english_cleaners"]
        ## >> "german_cleaners" Non-existence
        self.batch_size =  train_config["optimizer"]["batch_size"] # 16
        # filename?? "train.txt",  "val.txt"

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filename)
         # (512, 512, 512, 512)

        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # {'10087': 456, '10148': 13099}
       
        self.sort = sort  # False
        self.drop_last = drop_last # False
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]

        ##### idx = 0'10148_10349_003323',
        #  '10148',
        #  13099,
        #  'meine strafe hätte für die genüsse um die sie mich beneidet hat und kitty die  ...
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        ### phone
        # (array([158, 159, 165, 152, 154, 152, 151, 159, 163, 158, 151, 169, 158,
        # 148, 159, 151, 165, 154, 165, 159, 165, 151, 151, 169, 159, 167,
        # 158, 152, 165, 154, 148, 151, 163, 151, 159, 159, 165, 169, 163,
        # 358, 158, 159, 159, 159, 157, 157, 148, 159, 163, 151, 169, 148,
        # 159, 159, 165, 151, 163, 165, 169, 159, 159, 152, 169, 159, 165,
        # 358, 158]),
        # (67,))

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path) ## Shape: (1391, 80)


        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path) ## Shape: (181,)

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path) ## Shape: (181,)

        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path) ## Shape: (181,)
        
        sample = {
            "id": basename,        # 1
            "speaker": speaker_id, # 2
            "text": phone,         # 3
            "raw_text": raw_text,  # 4
            "mel": mel,            # 5
            "pitch": pitch,        # 6
            "energy": energy,      # 7
            "duration": duration,  # 8
        }

        return sample
    
    ##### @ __init__
    def process_meta(self, filename):
        ### train.txt, val.txt: '|'로 다 결합되어있음. 
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            name = [] ## file_id or basename
            speaker, text, raw_text = [], [], []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text
        

    ########## @ collate_fn ##############
    def reprocess(self, data, idxs):
        ### Paddind in Each Batch ### 
        ids = [data[idx]["id"] for idx in idxs]            # 1
        speakers = [data[idx]["speaker"] for idx in idxs]  # 2
        texts = [data[idx]["text"] for idx in idxs]        # 3
        raw_texts = [data[idx]["raw_text"] for idx in idxs]# 4
        mels = [data[idx]["mel"] for idx in idxs]          # 5
        pitches = [data[idx]["pitch"] for idx in idxs]     # 6
        energies = [data[idx]["energy"] for idx in idxs]   # 7
        durations = [data[idx]["duration"] for idx in idxs]# 8

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids, 
            raw_texts, 
            speakers,
            texts,
            text_lens,     ## New 1
            max(text_lens),## New 2
            mels,
            mel_lens,      ## New 3
            max(mel_lens), ## New 4
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data) # Batch SiZE[0] ?? what if 4

        if self.sort: ## False? -> else:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr) 
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
    

##################### TextDataset ###########################
class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        ## SAME as Dataset Above
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        ### self.process_meta is also same 
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filepath)

        ### Direct use instead of self.preprocessed_path 
        # self.preprocessed_path = preprocess_config["path"]["preprocessed_path"] #'/home/heiscold/prac/'
        with open( os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        ## SAME 
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    ##### @ __init__
    def process_meta(self, filename):
        ### train.txt, val.txt: '|'로 다 결합되어있음. 
        with open(filename, "r", encoding="utf-8") as f:
            name = [] ## file_id or basename
            speaker, text, raw_text = [], [], []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ### Padding in Each Batch

        ## Bath -> LIST
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]

        text_lens = np.array([text.shape[0] for text in texts])
        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)
