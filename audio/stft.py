## Github
## audio folder: https://github.com/ming024/FastSpeech2/blob/master/audio
##  ㄴ stft.py : https://github.com/ming024/FastSpeech2/blob/master/audio/stft.py
##  ㄴ audio_processing: https://github.com/ming024/FastSpeech2/blob/master/audio/audio_processing.py
##  ㄴ tools.py: https://github.com/ming024/FastSpeech2/blob/master/audio/tools.py

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

import librosa
# import librosa.display
import librosa.util as librosa_util
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center, tiny

from scipy.signal import get_window

import torch
import torch.nn as nn
import torch.nn.functional as F
# import IPython.display as ipd


################### @ STFT ################################
def window_sumsquare(
            window,       # hann
            n_frames,     # = magnitude.size(-1) # 1272 
            hop_length,   # 1024
            win_length,   # 1024
            n_fft,        # = filter_length (=1024)
            dtype=np.float32,
            norm=None,
            ):
    
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1) ## n : audio length? aprrocimate? # 326400 
    x = np.zeros(n, dtype=dtype)            ## np.zeros (Length of n)         # 326400 

    # # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True) ## window = hann ## (1024)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2 ## Energy??? Power? ## (1024)
    win_sq = librosa_util.pad_center(data =win_sq, size = n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length # 1024, 2*1024, 3*1024, 4*1024, ...
        # x - Length: ## (326400,)
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
        # min(n, sample + n_fft) --> sample + n_fft (= i*1024 + 1024)
        # min(n_fft, n - sample) --> n_fft (=1024)
        # max(0, min(n_fft, n - sample) -->   n_fft (=1024)
    return x

################# @ TacotronSTFT ######################
def dynamic_range_compression(x, 
                              C=1, 
                              clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C) 


def dynamic_range_decompression(x, 
                                C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C ## 그냥 Exponential 이네?


################################################################
class STFT(nn.Module):
    def __init__(self, 
                 filter_length, # = 1024, 
                 hop_length, # = 256, 
                 win_length, # = 1024, 
                 window="hann"
                 ):
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None

        scale = self.filter_length / self.hop_length

        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1)) ## n_fft // 2 + 1 ; 반띵 나이키스트?
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), ## 실수부 # [filter_length //2 + 1, filter_length]
             np.imag(fourier_basis[:cutoff, :])  ## 허수부 # [filter_length //2 + 1, filter_length]
                           ] )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :]) 
        # [filter_length+ 2, 1, filter_length] (1026, 1024)
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :] ## 역행렬 -> foward_basis와 Shape 맞춤!
        ) # filter_length //2 + 1, filter_length

        if window is not None:
            ## window = 'hann'
            assert filter_length >= win_length
            ## get window and zero center pad it to filter_length
            ## get_window: Return a window of a given length and type.
            fft_window = get_window(window, win_length, fftbins=True)       ## (filter_length) (=1024)
            fft_window = pad_center(data =fft_window, size = filter_length) ## paddin: (filter_length) (=1024)
            fft_window = torch.from_numpy(fft_window).float()               ## FFT 진행 -> torch.tensor

            # window the bases
            forward_basis *= fft_window ## forward_basis = forward_basis * fft_window | Shape: [filter_length+ 2, 1, filter_length] (1026, 1024)
            inverse_basis *= fft_window ## inverse_basis = inverse_basis * fft_window | Shape: [filter_length+ 2, 1, filter_length] (1026, 1024)

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())


    def transform(self, input_data):
        ## input_data: audio 
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data.cuda(), # Shape: [1, 1, 326482])
            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),
            # forward_basis - Shape: [1026, 1, 1024] (2*cutoff, 1, filter_length)
            stride=self.hop_length,
            padding=0,
        ).cpu()
        ## torch.Size([1, 1026, 1272]) == (1, 2*cutoff, 1272) ## 1272 : Don't know yet

        cutoff = int((self.filter_length / 2) + 1) ## self.cutoff 로 해도 되지 않을까?
        real_part = forward_transform[:, :cutoff, :] ### 실수부...? ## [1, 513, 1272]
        imag_part = forward_transform[:, cutoff:, :] ### 허수부...? ## [1, 513, 1272]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2) 
        ### Energy ??

        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        ## Phase: 위상? 복소수 부분?

        return magnitude, phase
        # magnitude ## from self.transform ## Shape: [1, 513, 1272]
        # phase     ## from self.transform ## Shape: [1, 513, 1272]
    

    def inverse(self, magnitude, phase):
        # magnitude ## from self.transform ## Shape: [1, 513, 1272]
        # phase     ## from self.transform ## Shape: [1, 513, 1272]

        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        ) ## recombine_magnitude_phase Shape: [1, 1026, 1272]
        ## recombine_magnitude_phase : magnitude * cosine(phase) | magnitude * sin(phase)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,   
            ## [1, 1026, 1272]
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            ## [1026, 1, 1024]
            stride=self.hop_length,
            padding=0,
        )
        ## audio.shape : (325458,)
        ## inverse_transform.shape # (1, 1, 326400)

        if self.window is not None:
            window_sum = window_sumsquare(
                        self.window, ## hann
                        magnitude.size(-1), ## 1272
                        hop_length = self.hop_length, ## 1024
                        win_length = self.win_length, ## 1024
                        n_fft = self.filter_length, ## 1024
                        dtype = np.float32,)
            ## window_sum Shape: (326400,)
            
            # remove modulation effects 라고는 하는데 뭔지 모르겠음. 
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
                ## tiny 함수의 정확한 역할을 알 수 없음
            ) ## Shape: [326399]

            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            ## window_sum Shape: (326400,)

            ## cuda
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum

            ## inverse_transform
            ## window_sum : [[326400]
            ## approx_nonzero_indices : [326399]
            ## inverse_transform: [1, 1, 326399]
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]            
            ## inverse_transform: [1, 1, 326399]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
            ## inverse_transform: [1, 1, 326399]

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        # inver_transform : [1, 1, 325888]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]
        # inver_transform : [1, 1, 325376]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        # magnitude ## from self.transform ## Shape: [1, 513, 1272]
        # phase     ## from self.transform ## Shape: [1, 513, 1272]

        reconstruction = self.inverse(self.magnitude, self.phase)
        # reconstruction(=inver_transform) : [1, 1, 325376]

        return reconstruction
    

###################### TacotronSTFT ###################################
class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,  # 1024
        hop_length,     # 1024
        win_length,     # 1024
        n_mel_channels, # 80
        sampling_rate,  # sample_rate (= 22050)
        mel_fmin,       # 0
        mel_fmax,       # 8000
    ):
        super(TacotronSTFT, self).__init__()

        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        ## https://librosa.org/doc/0.10.0/generated/librosa.filters.mel.html#librosa-filters-mel
        mel_basis = librosa_mel_fn(sr = sampling_rate, n_fft = filter_length, n_mels = n_mel_channels, fmin = mel_fmin, fmax =mel_fmax) 
        ## librosa_mel_fn( sr, n_fft, n_mels=128, fmin=0.0, fmax=None, )
        ## Return: np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        ## print(mel_basis.shape) # [n_mel_channels, cutoff] (80, 513)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """

        ## Example: 
        #   y = torch.randn(audio.shape).view(1, -1) 
        #   y_max = torch.max(y)
        #   y /= (y_max + 1)
        #   y - shape: torch.Size([1, 325458])
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        ## magnitudes: 1, 513, 1272
        ##  phases   : 1, 513, 1272

        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        # me_basis: [n_mel_channels, cutoff] (80, 513)
        # mel_output: [1, 80, 1272]

        mel_output = self.spectral_normalize(mel_output)
        # mel_output: [1, 80, 1272]

        energy = torch.norm(magnitudes, dim=1)
        ## magnitude = [1, 513, 1272]
        ## energy = torch.Size([1, 1272])
        return mel_output, energy
