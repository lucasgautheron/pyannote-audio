#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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
# Noise-based data augmentation
"""


import random
import numpy as np
import librosa
import torch
from pyannote.core import Segment
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.features.utils import get_audio_duration
from pyannote.core.utils.random import random_subsegment
from pyannote.core.utils.random import random_segment
from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.database import FileFinder
from .base import Augmentation
import augment


normalize = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


class AddNoise(Augmentation):
    """Additive noise data augmentation

    Parameters
    ----------
    collection : `str` or `list` of `str`
        `pyannote.database` collection(s) used for adding noise. Defaults to
        'MUSAN.Collection.BackgroundNoise' available in `pyannote.db.musan`
        package.
    snr_min, snr_max : int, optional
        Defines Signal-to-Noise Ratio (SNR) range in dB. Defaults to [5, 20].
    """

    def __init__(self, collection=None, snr_min=5, snr_max=20):
        super().__init__()

        if collection is None:
            collection = 'MUSAN.Collection.BackgroundNoise'
        if not isinstance(collection, (list, tuple)):
            collection = [collection]
        self.collection = collection

        self.snr_min = snr_min
        self.snr_max = snr_max

        # load noise database
        self.files_ = []
        preprocessors = {'audio': FileFinder(),
                         'duration': get_audio_duration}
        for collection in self.collection:
            protocol = get_protocol(collection,
                                    preprocessors=preprocessors)
            self.files_.extend(protocol.files())

    def __call__(self, original, sample_rate):
        """Augment original waveform

        Parameters
        ----------
        original : `np.ndarray`
            (n_samples, n_channels) waveform.
        sample_rate : `int`
            Sample rate.

        Returns
        -------
        augmented : `np.ndarray`
            (n_samples, n_channels) noise-augmented waveform.
        """

        raw_audio = RawAudio(sample_rate=sample_rate, mono=True)

        original_duration = len(original) / sample_rate

        # accumulate enough noise to cover duration of original waveform
        noises = []
        left = original_duration
        while left > 0:

            # select noise file at random
            file = random.choice(self.files_)
            duration = file['duration']

            # if noise file is longer than what is needed, crop it
            if duration > left:
                segment = next(random_subsegment(Segment(0, duration), left))
                noise = raw_audio.crop(file, segment,
                                       mode='center', fixed=left)
                left = 0

            # otherwise, take the whole file
            else:
                noise = raw_audio(file).data
                left -= duration

            noise = normalize(noise)
            noises.append(noise)

        # concatenate
        # FIXME: use fade-in between concatenated noises
        noise = np.vstack(noises)

        # select SNR at random
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        alpha = np.exp(-np.log(10) * snr / 20)

        return normalize(original) + alpha * noise


class WavAugment(Augmentation):

    def __init__(self, with_musan, augmentations):
        super().__init__()
        self.with_musan = with_musan
        if self.with_musan:
            self.musan = AddNoise()
        
        self.effect_chain = augment.EffectChain()
        for i, aug in enumerate(augmentations):
            if aug['name'] == 'reverb':
                self.random_room_scale = lambda: np.random.randint(0,100)
                self.effect_chain = self.effect_chain.reverb(50, 50, self.random_room_scale).channels(1)
            elif aug['name'] == 'pitch':
                min_shift = aug['params']['min_shift']
                max_shift = aug['params']['max_shift']
                sr = aug['params']['sr'] if 'sr' in aug['params'] else 16000
                self.random_pitch = lambda: np.random.randint(min_shift, max_shift)
                self.effect_chain = self.effect_chain.pitch("-q", self.random_pitch).rate(sr)
            elif aug['name'] == 'tdrop':
                max_seconds = aug['params']['max_seconds']
                self.effect_chain = self.effect_chain.time_dropout(max_seconds=max_seconds)
            elif aug['name'] == 'bandrej':
                band_width = aug['params']['band_width']
                self.random_bandrej = lambda: WavAugment.random_bandrej(band_width)
                self.effect_chain = self.effect_chain.sinc('-a', '120', self.random_bandrej)
        
    @staticmethod
    def random_bandrej(band_width, lower_bound=0, upper_bound=8000):
        start = np.random.randint(lower_bound, upper_bound-band_width)
        return str(start+band_width)+"-"+str(start)

            
    def __call__(self, original, sample_rate):
        augmented = original
        
        if self.with_musan:
            augmented = self.musan(augmented, sample_rate)

        augmented = self.effect_chain.apply(torch.from_numpy(augmented).view(1,-1),
                                            src_info={'rate': sample_rate},
                                            target_info={'rate': sample_rate})
        
        augmented = augmented.view(-1, 1).numpy()
        augmented = normalize(augmented)
        
        return augmented


class PitchShift(Augmentation):
    
    def __init__(self, with_noise, min_shift, max_shift):
        super().__init__()
        self.with_noise = with_noise
        self.min_shift = min_shift
        self.max_shift = max_shift
        if with_noise:
            self.add_noise = AddNoise()
        # WavAugment arguments
        self.src_info = {'rate': 16000}
        self.target_info = {'rate': 16_000}
        self.random_pitch_shift = lambda: np.random.randint(-300, +300)
        self.effect_chain = augment.EffectChain().pitch("-q", self.random_pitch_shift).rate(16_000)



    def __call__(self, original, sample_rate):
        augmented = self.effect_chain.apply(torch.from_numpy(original), 
                                            src_info=self.src_info, 
                                            target_info=self.target_info)
        augmented = augmented.view(-1, 1)
        augmented = augmented.numpy()
        if self.with_noise:
            augmented = self.add_noise(augmented, sample_rate)
            return augmented
        else:
            return normalize(augmented)


class AddNoiseFromGaps(Augmentation):
    """Additive noise data augmentation.

    While AddNoise assumes that files contain only noise, this class uses
    non-speech regions (= gaps) as noise. This is expected to generate more
    realistic noises.

    Parameters
    ----------
    protocol : `str` or `pyannote.database.Protocol`
        Protocol name (e.g. AMI.SpeakerDiarization.MixHeadset)
    subset : {'train', 'development', 'test'}, optional
        Use this subset. Defaults to 'train'.
    snr_min, snr_max : int, optional
        Defines Signal-to-Noise Ratio (SNR) range in dB. Defaults to [5, 20].

    See also
    --------
    `AddNoise`
    """

    def __init__(self, protocol=None, subset='train',
                 snr_min=5, snr_max=20):
        super().__init__()

        self.protocol = protocol
        self.subset = subset
        self.snr_min = snr_min
        self.snr_max = snr_max

        # returns gaps in annotation as pyannote.core.Timeline instance
        get_gaps = lambda f: f['annotation'].get_timeline().gaps(
            support=get_annotated(f))

        if isinstance(protocol, str):
            preprocessors = {
                'audio': FileFinder(),
                'duration': get_audio_duration,
                'gaps': get_gaps}
            protocol = get_protocol(self.protocol,
                                    preprocessors=preprocessors)
        else:
            protocol.preprocessors['gaps'] = get_gaps

        self.files_ = list(getattr(protocol, self.subset)())

    def __call__(self, original, sample_rate):
        """Augment original waveform

        Parameters
        ----------
        original : `np.ndarray`
            (n_samples, n_channels) waveform.
        sample_rate : `int`
            Sample rate.

        Returns
        -------
        augmented : `np.ndarray`
            (n_samples, n_channels) noise-augmented waveform.
        """

        raw_audio = RawAudio(sample_rate=sample_rate, mono=True)

        # accumulate enough noise to cover duration of original waveform
        noises = []
        len_left = len(original)
        while len_left > 0:

            # select noise file at random
            file = random.choice(self.files_)

            # select noise segment at random
            segment = next(random_segment(file['gaps'], weighted=False))
            duration = segment.duration
            segment_len = duration * sample_rate

            # if noise segment is longer than what is needed, crop it at random
            if segment_len > len_left:
                duration = len_left / sample_rate
                segment = next(random_subsegment(segment, duration))

            noise = raw_audio.crop(file, segment,
                                   mode='center', fixed=duration)

            # decrease the `len_left` value by the size of the returned noise
            len_left -= len(noise)

            noise = normalize(noise)
            noises.append(noise)

        # concatenate
        # FIXME: use fade-in between concatenated noises
        noise = np.vstack(noises)

        # select SNR at random
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        alpha = np.exp(-np.log(10) * snr / 20)

        return normalize(original) + alpha * noise
