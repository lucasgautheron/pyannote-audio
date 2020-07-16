#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
Feature extraction using [`librosa`](https://librosa.github.io/librosa/)
"""

import librosa
import numpy as np

from .base import FeatureExtraction
from pyannote.core.segment import SlidingWindow


class LibrosaFeatureExtraction(FeatureExtraction):
    """librosa feature extraction base class

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025 (25ms).
    step : float, optional
        Defaults to 0.010 (10ms).
    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01):

        super().__init__(sample_rate=sample_rate,
                         augmentation=augmentation)
        self.duration = duration
        self.step = step

        self.sliding_window_ = SlidingWindow(start=-.5*self.duration,
                                             duration=self.duration,
                                             step=self.step)

    def get_resolution(self):
        return self.sliding_window_


class LibrosaSpectrogram(LibrosaFeatureExtraction):
    """librosa spectrogram

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.010):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)

        self.n_fft_ = int(self.duration * self.sample_rate)
        self.hop_length_ = int(self.step * self.sample_rate)

    def get_dimension(self):
        return self.n_fft_ // 2 + 1

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_dimensions) numpy array
            Features
        """

        fft = librosa.core.stft(y=y.squeeze(), n_fft=self.n_fft_,
                                hop_length=self.hop_length_,
                                center=True, window='hamming')
        return np.abs(fft).T


class LibrosaMelSpectrogram(LibrosaFeatureExtraction):
    """librosa mel-spectrogram

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    n_mels : int, optional
        Defaults to 96.
    """

    means64 = np.array([-72.35066,  -76.04799  ,-76.98745 , -76.11548 , -75.50013  ,-74.53862,
                        -75.42524,  -75.928825 ,-76.747536, -76.86116,  -77.05067 , -76.82145,
                        -76.86239, -76.78075  ,-76.91255 , -76.98166 , -77.091484 ,-77.256065,
                        -77.368  ,-77.604164 ,-77.68206 , -77.8754 ,  -78.012474 ,-78.04885,
                        -78.21733 , -78.25786,  -78.33423 , -78.4127 ,  -78.48568  ,-78.490746,
                        -78.58492 , -78.61684,  -78.696655, -78.750496, -78.82386  ,-78.8745,
                        -78.90399 , -78.95819,  -78.982796, -79.01734 , -79.03398  ,-79.05584,
                        -79.073074, -79.06086,  -79.08149 , -79.15306 , -79.238976 ,-79.291046,
                        -79.30616 , -79.31063,  -79.321686, -79.36596 , -79.41566  ,-79.46898,
                        -79.49906 , -79.51222,  -79.52569,  -79.55023 , -79.59014 , -79.63495,
                        -79.65209 , -79.664856, -79.69292,  -79.71706 ])

    stds64 = np.array([15.008968 , 10.3390665,  8.762007 ,  9.57795  , 10.284524,  11.484503,
                        10.80691  , 10.328627 ,  9.214289 ,  9.013638 ,  8.720401,   9.091676,
                        9.129078  , 9.330731  , 9.202403  , 9.144097  , 8.992769 ,  8.739887,
                        8.562684  , 8.172761  , 8.040799  , 7.6979456 , 7.4391127,  7.3654356,
                        7.004751  , 6.896536  , 6.7404127 , 6.5700574 , 6.400997 ,  6.3567567,
                        6.138168  , 6.0443068 , 5.839879  , 5.6976113 , 5.5050106,  5.3660774,
                        5.284663  , 5.154798  , 5.082323  , 4.9900556 , 4.937515 ,  4.857299,
                        4.795007  , 4.820238  , 4.7643633 , 4.577041  , 4.3510222,  4.2341733,
                        4.1528363 , 4.14321   , 4.1376243 , 4.049931  , 3.9655843,  3.8467138,
                        3.830402  , 3.9075005 , 3.9385607 , 3.828856  , 3.6112642,  3.338928,
                        3.2680924 , 3.2624815 , 3.1928308 , 3.1359286])
    means128 = np.array(
                [-73.00442,  -73.49532,  -76.876785, -76.81982 , -77.66969 , -77.65878,
                -77.83385 , -77.030136 ,-76.9962   ,-76.51872  ,-76.49268  ,-75.72631,
                -76.55445 , -76.512276 ,-76.98029  ,-76.963    ,-77.620224 ,-77.6039,
                -77.835075, -77.606064 ,-77.94766  ,-77.82903  ,-77.89221  ,-77.53984,
                -77.81121 , -77.686264 ,-77.76637  ,-77.52012  ,-77.82931  ,-77.75326,
                -77.853546, -77.672066 ,-77.95048  ,-77.905754 ,-77.99899  ,-77.8809,
                -78.14643 , -78.13294  ,-78.21345  ,-78.14327  ,-78.3737   ,-78.36743,
                -78.44779 , -78.365326 ,-78.563576, -78.52445 , -78.55568 , -78.60859,
                -78.66763 , -78.72577  ,-78.76409  ,-78.80764  ,-78.840805 ,-78.858284,
                -78.86534 , -78.87987  ,-78.906296 ,-78.91828  ,-78.933975 ,-78.95325,
                -78.97738 , -78.95369  ,-78.980865 ,-79.05623  ,-79.07161  ,-79.02075,
                -79.14191 , -79.10744  ,-79.142975 ,-79.17029  ,-79.179565 ,-79.200264,
                -79.237305, -79.20812  ,-79.26375  ,-79.26995  ,-79.26649  ,-79.29178,
                -79.299385, -79.30507  ,-79.30875  ,-79.315674 ,-79.32689  ,-79.33691,
                -79.34571 , -79.348206 ,-79.32492  ,-79.33963  ,-79.35065  ,-79.34866,
                -79.398705, -79.41347  ,-79.45693  ,-79.46582  ,-79.48237  ,-79.49201,
                -79.50116 , -79.5027   ,-79.50089  ,-79.50027  ,-79.50517  ,-79.51778,
                -79.5313  , -79.54387  ,-79.564125 ,-79.57621  ,-79.59934  ,-79.61079,
                -79.61288 , -79.62035  ,-79.61819  ,-79.62156  ,-79.62289  ,-79.62674,
                -79.64209 , -79.653496 ,-79.673584 ,-79.6984   ,-79.71076  ,-79.71944,
                -79.72615 , -79.72529  ,-79.72938  ,-79.73901  ,-79.75059  ,-79.7532,
                -79.76191 , -79.7687])

    stds128 = np.array(
            [14.786413,  13.6437845,  9.28494 ,   9.325214,   7.7865443,  7.678337,
            7.3499985 , 8.4762     ,8.415782  , 8.951134  , 9.159416  ,10.354463,
            9.327971  , 9.338247   ,8.901754  , 9.042919  , 7.8613048 , 7.840806,
            7.530613  , 7.933298   ,7.2513022 , 7.413867  , 7.3939476 , 8.066774,
            7.560603  , 7.7540145  ,7.7499194 , 8.252044  , 7.6490884 , 7.767383,
            7.7011743 , 8.074898   ,7.4820714 , 7.5444183 , 7.4583645 , 7.707518,
            7.117355  , 7.129987   ,7.047929  , 7.2117653 , 6.6683364 , 6.675195,
            6.565239  , 6.762912   ,6.2945337 , 6.3562965 , 6.2847157 , 6.1667542,
            6.035055  , 5.908828   ,5.823886  , 5.7166047 , 5.629608  , 5.5800095,
            5.554892  , 5.510243   ,5.4342675 , 5.392146  , 5.3393216 , 5.279994,
            5.2142434 , 5.2603173  ,5.1855097 , 4.990829  , 4.936173  , 5.053377,
            4.7370205 , 4.8069286  ,4.702802  , 4.6232553 , 4.5917335 , 4.533516,
            4.4302278 , 4.5112042  ,4.3565803 , 4.335996  , 4.3457193 , 4.272566,
            4.242607  , 4.222052   ,4.206646  , 4.178562  , 4.13687   , 4.1024256,
            4.0703573 , 4.0579395  ,4.124645  , 4.085038  , 4.050272  , 4.048106,
            3.9069748 , 3.855534   ,3.7258766 , 3.701149  , 3.685939  , 3.6404617,
            3.5779767 , 3.5740194  ,3.585124  , 3.5970523 , 3.595137  , 3.5677662,
            3.5396507 , 3.5175047  ,3.485393  , 3.4644833 , 3.3956714 , 3.378417,
            3.4154055 , 3.4241478  ,3.526602  , 3.5264404 , 3.5743918 , 3.5608404,
            3.462187  , 3.4145389  ,3.2734444 , 3.0961275 , 3.0241492 , 2.9814441,
            2.9527688 , 2.9827504  ,2.9877746 , 2.9615192 , 2.9192162 , 2.934742,
            2.8957856 , 2.8578253])

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.010, n_mels=96):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)

        self.n_mels = n_mels
        self.n_fft_ = int(self.duration * self.sample_rate)
        self.hop_length_ = int(self.step * self.sample_rate)

    def get_dimension(self):
        return self.n_mels

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_mels) numpy array
            Features
        """

        X = librosa.feature.melspectrogram(
            y.squeeze(), sr=sample_rate, n_mels=self.n_mels,
            n_fft=self.n_fft_, hop_length=self.hop_length_,
            power=2.0)
        X = librosa.amplitude_to_db(X, ref=1.0, amin=1e-5, top_db=80.0).T

        # normalize
        X -= LibrosaMelSpectrogram.means64
        X /= LibrosaMelSpectrogram.stds64

        return X


class LibrosaMFCC(LibrosaFeatureExtraction):
    """librosa MFCC

    ::

            | e    |  energy
            | c1   |
            | c2   |  coefficients
            | c3   |
            | ...  |
            | Δe   |  energy first derivative
            | Δc1  |
        x = | Δc2  |  coefficients first derivatives
            | Δc3  |
            | ...  |
            | ΔΔe  |  energy second derivative
            | ΔΔc1 |
            | ΔΔc2 |  coefficients second derivatives
            | ΔΔc3 |
            | ...  |


    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    e : bool, optional
        Energy. Defaults to True.
    coefs : int, optional
        Number of coefficients. Defaults to 11.
    De : bool, optional
        Keep energy first derivative. Defaults to False.
    D : bool, optional
        Add first order derivatives. Defaults to False.
    DDe : bool, optional
        Keep energy second derivative. Defaults to False.
    DD : bool, optional
        Add second order derivatives. Defaults to False.

    Notes
    -----
    Internal setup
        * fftWindow = Hanning
        * melMaxFreq = 6854.0
        * melMinFreq = 130.0
        * melNbFilters = 40

    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01,
                 e=False, De=True, DDe=True,
                 coefs=19, D=True, DD=True,
                 fmin=0.0, fmax=None, n_mels=40):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD

        self.n_mels = n_mels  # yaafe / 40
        self.fmin = fmin      # yaafe / 130.0
        self.fmax = fmax      # yaafe / 6854.0

    def get_context_duration(self):
        return 0.

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_dimensions) numpy array
            Features
        """

        # adding because C0 is the energy
        n_mfcc = self.coefs + 1

        n_fft = int(self.duration * sample_rate)
        hop_length = int(self.step * sample_rate)

        mfcc = librosa.feature.mfcc(
            y=y.squeeze(), sr=sample_rate, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=self.n_mels, htk=True,
            fmin=self.fmin, fmax=self.fmax)

        if self.De or self.D:
            mfcc_d = librosa.feature.delta(
                mfcc, width=9, order=1, axis=-1)

        if self.DDe or self.DD:
            mfcc_dd = librosa.feature.delta(
                mfcc, width=9, order=2, axis=-1)

        stack = []

        if self.e:
            stack.append(mfcc[0, :])

        stack.append(mfcc[1:, :])

        if self.De:
            stack.append(mfcc_d[0, :])

        if self.D:
            stack.append(mfcc_d[1:, :])

        if self.DDe:
            stack.append(mfcc_dd[0, :])

        if self.DD:
            stack.append(mfcc_dd[1:, :])

        return np.vstack(stack).T

    def get_dimension(self):
        n_features = 0
        n_features += self.e
        n_features += self.De
        n_features += self.DDe
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD
        return n_features
