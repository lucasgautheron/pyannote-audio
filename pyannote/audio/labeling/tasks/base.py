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

from typing import Optional
from typing import Text

import torch
import torch.nn.functional as F
from torch import autograd

import numpy as np
import scipy.signal

from pyannote.core import Segment
from pyannote.core import SlidingWindow
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from pyannote.database import get_unique_identifier
from pyannote.database import get_annotated
from pyannote.database.protocol.protocol import Protocol

from pyannote.core.utils.numpy import one_hot_encoding

from pyannote.audio.features import RawAudio
from pyannote.audio.features.wrapper import Wrapper, Wrappable

from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment

from pyannote.audio.train.trainer import Trainer
from pyannote.audio.train.generator import BatchGenerator

from pyannote.audio.train.task import Task, TaskType, TaskOutput

from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.model import Alignment

SECONDS_IN_A_DAY = 24 * 60 * 60


class LabelingTaskGenerator(BatchGenerator):
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    feature_extraction : Wrappable
        Describes how features should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
    protocol : Protocol
    subset : {'train', 'development', 'test'}, optional
        Protocol and subset.
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
        Defaults to `feature_extraction.sliding_window`
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models that
        include the feature extraction step (e.g. SincNet) and therefore use a
        different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1. Has not effect when exhaustive is False.
    mask : str, optional
        When provided, protocol files are expected to contain a key named after
        this `mask` variable and providing a `SlidingWindowFeature` instance.
        Generated batches will contain an additional "mask" key (on top of
        existing "X" and "y" keys) computed as an excerpt of `current_file[mask]`
        time-aligned with "y". Defaults to not add any "mask" key.
    local_labels : bool, optional
        Set to True to yield samples with local (file-level) labels.
        Defaults to use global (protocol-level) labels.
    """

    def __init__(self,
                 feature_extraction: Wrappable,
                 protocol: Protocol,
                 subset: Text = 'train',
                 resolution: Optional[Resolution] = None,
                 alignment: Optional[Alignment] = None,
                 duration: float = 2.0,
                 batch_size: int = 32,
                 per_epoch: float = None,
                 exhaustive: bool = False,
                 step: float = 0.1,
                 mask: Text = None,
                 local_labels: bool = False):

        self.feature_extraction = Wrapper(feature_extraction)
        self.duration = duration
        self.exhaustive = exhaustive
        self.step = step
        self.mask = mask
        self.local_labels = local_labels

        self.resolution_ = resolution

        if alignment is None:
            alignment = 'center'
        self.alignment = alignment

        self.batch_size = batch_size

        # load metadata and estimate total duration of training data
        total_duration = self._load_metadata(protocol, subset=subset)

        #
        if per_epoch is None:

            # 1 epoch = covering the whole training set once
            #
            per_epoch = total_duration / SECONDS_IN_A_DAY

            # when exhaustive is False, this is not completely correct.
            # in practice, it will randomly sample audio chunk until their
            # overall duration reaches the duration of the training set.
            # but nothing guarantees that every single part of the training set
            # has been seen exactly once: it might be more than once, it might
            # be less than once. on average, however, after a certain amount of
            # epoch, this will be correct

            # when exhaustive is True, however, we can actually make sure every
            # single part of the training set has been seen. we just have to
            # make sur we account for the step used by the exhaustive sliding
            # window
            if self.exhaustive:
                per_epoch *= np.ceil(1 / self.step)

        self.per_epoch = per_epoch

    # TODO. use cached property (Python 3.8 only)
    # https://docs.python.org/fr/3/library/functools.html#functools.cached_property
    @property
    def resolution(self):

        if self.resolution_ in [None, RESOLUTION_FRAME]:
            return self.feature_extraction.sliding_window

        if self.resolution_ == RESOLUTION_CHUNK:
            return self.SlidingWindow(duration=self.duration,
                                      step=self.step * self.duration)

        return self.resolution_

    def postprocess_y(self, Y: np.ndarray) -> np.ndarray:
        """This function does nothing but return its input.
        It should be overriden by subclasses.

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray

        Returns
        -------
        postprocessed :

        """
        return Y

    def initialize_y(self, current_file):
        """Precompute y for the whole file

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.

        Returns
        -------
        y : `SlidingWindowFeature`
            Precomputed y for the whole file
        """

        if self.local_labels:
            labels = current_file['annotation'].labels()
        else:
            labels = self.segment_labels_

        y, _ = one_hot_encoding(current_file['annotation'],
                                get_annotated(current_file),
                                self.resolution,
                                labels=labels,
                                mode='center')

        return SlidingWindowFeature(self.postprocess_y(y.data),
                                    y.sliding_window)

    def crop_y(self, y, segment):
        """Extract y for specified segment

        Parameters
        ----------
        y : `pyannote.core.SlidingWindowFeature`
            Output of `initialize_y` above.
        segment : `pyannote.core.Segment`
            Segment for which to obtain y.

        Returns
        -------
        cropped_y : (n_samples, dim) `np.ndarray`
            y for specified `segment`
        """

        return y.crop(segment,
                      mode=self.alignment,
                      fixed=self.duration)

    def _load_metadata(self, protocol, subset='train') -> float:
        """Load training set metadata

        This function is called once at instantiation time, returns the total
        training set duration, and populates the following attributes:

        Attributes
        ----------
        data_ : dict

            {'segments': <list of annotated segments>,
             'duration': <total duration of annotated segments>,
             'current_file': <protocol dictionary>,
             'y': <labels as numpy array>}

        segment_labels_ : list
            Sorted list of (unique) labels in protocol.

        file_labels_ : dict of list
            Sorted lists of (unique) file labels in protocol

        Returns
        -------
        duration : float
            Total duration of annotated segments, in seconds.
        """

        self.data_ = {}
        segment_labels, file_labels = set(), dict()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # ensure annotation/annotated are cropped to actual file duration
            support = Segment(start=0, end=current_file['duration'])
            current_file['annotated'] = get_annotated(current_file).crop(
                support, mode='intersection')
            current_file['annotation'] = current_file['annotation'].crop(
                support, mode='intersection')

            # keep track of unique segment labels
            segment_labels.update(current_file['annotation'].labels())

            # keep track of unique file labels
            for key, value in current_file.items():
                if isinstance(value, (Annotation, Timeline, SlidingWindowFeature)):
                    continue
                if key not in file_labels:
                    file_labels[key] = set()
                file_labels[key].add(value)

            segments = [s for s in current_file['annotated']
                          if s.duration > self.duration]

            # corner case where no segment is long enough
            # and we removed them all...
            if not segments:
                continue

            # total duration of label in current_file (after removal of
            # short segments).
            duration = sum(s.duration for s in segments)

            # store all these in data_ dictionary
            datum = {'segments': segments,
                     'duration': duration,
                     'current_file': current_file}
            uri = get_unique_identifier(current_file)
            self.data_[uri] = datum

        self.file_labels_ = {k: sorted(file_labels[k]) for k in file_labels}
        self.segment_labels_ = sorted(segment_labels)

        for uri in list(self.data_):
            current_file = self.data_[uri]['current_file']
            y = self.initialize_y(current_file)
            self.data_[uri]['y'] = y
            if self.mask is not None:
                mask = current_file[self.mask]
                current_file[self.mask] = mask.align(y)

        return sum(datum['duration'] for datum in self.data_.values())

    @property
    def specifications(self):
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`pyannote.audio.train.Task`) : task
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = {
            'task': Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
                         output=TaskOutput.SEQUENCE),
            'X': {'dimension': self.feature_extraction.dimension}}

        if not self.local_labels:
            specs['y'] = {'classes': self.segment_labels_}

        return specs

    def samples(self):
        if self.exhaustive:
            return self._sliding_samples()
        else:
            return self._random_samples()

    def _random_samples(self):
        """Random samples

        Returns
        -------
        samples : generator
            Generator that yields {'X': ..., 'y': ...} samples indefinitely.
        """

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        while True:

            # choose file at random with probability
            # proportional to its (annotated) duration
            uri = uris[np.random.choice(len(uris), p=probabilities)]

            datum = self.data_[uri]
            current_file = datum['current_file']

            # choose one segment at random with probability
            # proportional to its duration
            segment = next(random_segment(datum['segments'], weighted=True))

            # choose fixed-duration subsegment at random
            subsegment = next(random_subsegment(segment, self.duration))

            X = self.feature_extraction.crop(current_file,
                                             subsegment,
                                             mode='center',
                                             fixed=self.duration)

            y = self.crop_y(datum['y'],
                            subsegment)
            sample = {'X': X, 'y': y}

            if self.mask is not None:
                mask = self.crop_y(current_file[self.mask],
                                   subsegment)
                sample['mask'] = mask

            for key, classes in self.file_labels_.items():
                sample[key] = classes.index(current_file[key])

            yield sample

    def _sliding_samples(self):

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)
        sliding_segments = SlidingWindow(duration=self.duration,
                                         step=self.step * self.duration)

        while True:

            np.random.shuffle(uris)

            # loop on all files
            for uri in uris:

                datum = self.data_[uri]

                # make a copy of current file
                current_file = dict(datum['current_file'])

                # compute features for the whole file
                features = self.feature_extraction(current_file)

                # randomly shift 'annotated' segments start time so that
                # we avoid generating exactly the same subsequence twice
                annotated = Timeline()
                for segment in get_annotated(current_file):
                    shifted_segment = Segment(
                        segment.start + np.random.random() * self.duration,
                        segment.end)
                    if shifted_segment:
                        annotated.add(shifted_segment)

                samples = []
                for sequence in sliding_segments(annotated):

                    X = features.crop(sequence, mode='center',
                                      fixed=self.duration)
                    y = self.crop_y(datum['y'], sequence)
                    sample = {'X': X, 'y': y}

                    if self.mask is not None:

                        # extract mask for current sub-segment
                        mask = current_file[self.mask].crop(sequence,
                                                            mode='center',
                                                            fixed=self.duration)

                        # it might happen that "mask" and "y" use different
                        # sliding windows. therefore, we simply resample "mask"
                        # to match "y"
                        if len(mask) != len(y):
                            mask = scipy.signal.resample(mask, len(y), axis=0)
                        sample['mask'] = mask

                    for key, classes in self.file_labels_.items():
                        sample[key] = classes.index(current_file[key])

                    samples.append(sample)

                np.random.shuffle(samples)
                for sample in samples:
                    yield sample

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_epoch = self.per_epoch * SECONDS_IN_A_DAY
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))


class LabelingTask(Trainer):
    """Base class for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1. Has not effect when exhaustive is False.
    """

    def __init__(self, duration: float = 2.0,
                       batch_size: int = 32,
                       per_epoch: float = None,
                       exhaustive: bool = False,
                       step: float = 0.1):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.exhaustive = exhaustive
        self.step = step

    def get_batch_generator(self, feature_extraction: Wrappable,
                                  protocol: Protocol,
                                  subset: Text = 'train',
                                  resolution: Optional[Resolution] = None,
                                  alignment: Optional[Alignment] = None) -> LabelingTaskGenerator:
        """This method should be overriden by subclass

        Parameters
        ----------
        feature_extraction : Wrappable
            Describes how features should be obtained.
            See pyannote.audio.features.wrapper.Wrapper documentation for details.
        protocol : Protocol
        subset : {'train', 'development'}, optional
            Defaults to 'train'.
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """

        return LabelingTaskGenerator(feature_extraction,
                                     protocol,
                                     subset=subset,
                                     resolution=resolution,
                                     alignment=alignment,
                                     duration=self.duration,
                                     per_epoch=self.per_epoch,
                                     batch_size=self.batch_size,
                                     exhaustive=self.exhaustive,
                                     step=self.step)

    @property
    def weight(self):
        """Class/task weights

        Returns
        -------
        weight : None or `torch.Tensor`
        """
        return None

    def on_train_start(self):
        """Set loss function (with support for class weights)

        loss_func_ = Function f(input, target, weight=None) -> loss value
        """

        self.task_ = self.model_.task

        if self.task_.is_multiclass_classification:

            self.n_classes_ = len(self.model_.classes)

            def loss_func(input, target, weight=None, mask=None):
                if mask is None:
                    return F.nll_loss(input, target, weight=weight,
                                      reduction='mean')
                else:
                    return torch.mean(
                        mask * F.nll_loss(input, target,
                                          weight=weight,
                                          reduction='none'))

        if self.task_.is_multilabel_classification:
            
            # store losses for each task separately to log them later. cleared every epoch
            self.task_batch_losses = [[] for _ in range(len(self.model_.classes))]
            self.weighted_task_batch_losses = [[] for _ in range(len(self.model_.classes))]
            self.task_batch_grad_norms = [[] for _ in range(len(self.model_.classes))]
            self.task_batch_progress_rates = [[] for _ in range(len(self.model_.classes))]
            #self.class_weights = {'KCHI': torch.tensor([0.6433, 2.2444], device=self.device_),
            #                      'CHI': torch.tensor([0.5164, 15.7594], device=self.device_),
            #                      'MAL': torch.tensor([0.5049, 51.6005], device=self.device_),
            #                      'FEM': torch.tensor([0.8013, 1.3298], device=self.device_),
            #                      'SPEECH': torch.tensor([1.2841, 0.8188], device=self.device_)}
            self.class_weights = {'KCHI': torch.tensor([0., 0.], device=self.device_),
                                  'CHI': torch.tensor([0., 0.], device=self.device_),
                                  'MAL': torch.tensor([1., 1.], device=self.device_),
                                  'FEM': torch.tensor([1., 1.], device=self.device_),
                                  'SPEECH': torch.tensor([1., 1.], device=self.device_)}

            def loss_func(input,
                          target,
                          weight=None,
                          mask=None,
                          return_separate_losses=False,
                          count_for_batch_loss=True):
                if mask is None:
                    num_labels = target.size()[-1]
                    # label-wise cross entropy
                    losses = []
                    for i in range(num_labels):
                        voice_type = self.model_.classes[i]
                        class_weights = self.class_weights[voice_type][target[..., i].to(int)]
                        task_i_loss = F.binary_cross_entropy(input[..., i], target[..., i], weight=class_weights, reduction='mean')
                        losses.append(task_i_loss)

                    aggregated_loss = sum(losses)/float(num_labels)
                    
                    if count_for_batch_loss:
                        for i in range(num_labels):
                            self.task_batch_losses[i].append(losses[i].clone().detach().cpu())
                    
                    if not return_separate_losses:
                        return aggregated_loss
                    else:
                        return aggregated_loss, losses
                else:
                    return torch.mean(
                        mask * F.binary_cross_entropy(input, target,
                                                      weight=weight,
                                                      reduction='none'))

        if self.task_.is_regression:
            def loss_func(input, target, weight=None, mask=None):
                if mask is None:
                    return F.mse_loss(input, target,
                                      reduction='mean')
                else:
                    return torch.mean(
                        mask * F.mse_loss(input, target,
                                          reduction='none'))

        self.loss_func_ = loss_func

    def on_epoch_start(self):
        """
        - Clearing per batch losses saved during last epoch
        - Logging GradNorm weights
        """
        if self.task_.is_multilabel_classification:
            self.batch_ = 0
            weight_dict = {}
            normed_weight_dict = {}
            weight_sum = sum(self.weights).clone().detach().cpu().numpy()
            for i in range(len(self.task_batch_losses)):
                # clearing batch log memory
                self.task_batch_losses[i].clear()
                self.weighted_task_batch_losses[i].clear()
                self.task_batch_progress_rates[i].clear()
                self.task_batch_grad_norms[i].clear()
                # weights logging
                weight = self.weights[i].clone().detach().cpu().numpy()
                weight_dict[self.label_names[i]] = weight
                normed_weight_dict[self.label_names[i]] = weight/(weight_sum/len(self.weights))

            #self.tensorboard_.add_scalars(
            #                        f'train/loss_weights',
            #                        weight_dict,
            #                        global_step=self.epoch_)
            #self.tensorboard_.add_scalars(
            #                        f'train/normed_loss_weights',
            #                        normed_weight_dict,
            #                        global_step=self.epoch_)
            

    def on_epoch_end(self):
        """Log per task loss to tensorboard for multilabel_classification
        """
        if self.task_.is_multilabel_classification:
            # Log loss
            loss_dict = {}
            weighted_loss_dict = {}
            progress_ratio_dict = {}
            grad_norm_dict = {}
            for i in range(len(self.task_batch_losses)):
                label = self.label_names[i]
                loss = torch.mean(torch.tensor(self.task_batch_losses[i], device=self.device_, requires_grad=False))
                loss_dict[label] = loss.cpu().numpy()
                if self.epoch == 1:
                    self.first_epoch_losses.append(loss)
            
                weighted_loss = torch.mean(torch.tensor(self.weighted_task_batch_losses[i]))
                weighted_loss_dict[label] = weighted_loss.cpu().numpy()
            
                rel_inv_rate = torch.mean(torch.tensor(self.task_batch_progress_rates[i]))
                progress_ratio_dict[label] = rel_inv_rate.cpu().numpy()

                grad_norm = torch.mean(torch.tensor(self.task_batch_grad_norms[i]))
                grad_norm_dict[label] = grad_norm.cpu().numpy()

            self.tensorboard_.add_scalars(
                    f'train/loss',
                    loss_dict,
                    global_step=self.epoch_)
            #self.tensorboard_.add_scalars(
            #        f'train/weighted-loss',
            #        weighted_loss_dict,
            #        global_step=self.epoch_)
            #self.tensorboard_.add_scalars(
            #        f'train/grad-norms',
            #        grad_norm_dict,
            #        global_step=self.epoch_)
            #self.tensorboard_.add_scalars(
            #        f'train/rel-inv-rates',
            #        progress_ratio_dict,
            #        global_step=self.epoch_)


    def grad_norm(self, fX, target, mask, progress_weighting=False):
        """Weighting the loss functions by the GradNorm scheme
        https://arxiv.org/abs/1711.02257
        """
        # Weights normalization
        weights_sum = sum(self.weights)
        for i, w in enumerate(self.weights):
            #w.to(device=self.device_)
            w /= weights_sum / len(self.weights)
        
        # Gradient norms
        joint_loss, single_losses = self.loss_func_(fX,
                                                    target,
                                                    weight=self.weights,
                                                    mask=mask,
                                                    return_separate_losses=True,
                                                    count_for_batch_loss=False)
        grad_norms = []
        grad_layer = self.model_.get_last_shared_layer()
        for i, single_loss in enumerate(single_losses):
            grad = autograd.grad(single_loss, [grad_layer.weight], retain_graph=True)[0]
            grad_norms.append(torch.norm(grad, p=2))

            self.task_batch_grad_norms[i].append(grad_norms[-1].clone().detach().cpu()) #logging
        avg_grad_norm = sum(grad_norms)/float(len(grad_norms))

        # progress ratios
        relative_inv_rates = []
        if progress_weighting:
            #if self.batch_ == 1 and self.epoch_ == 1:
            #    self.first_epoch_losses = single_losses

            unweighted_losses = []
            inverse_rates = []
            for i in range(len(single_losses)):
                unweighted_losses.append(single_losses[i]/self.weights[i])
                inverse_rates.append(unweighted_losses[i].clone().detach() / self.first_epoch_losses[i])
            avg_inv_rate = sum(inverse_rates)/float(len(inverse_rates))
            for i in range(len(single_losses)):
                relative_inv_rates.append(inverse_rates[i] / avg_inv_rate)
        else:
             for i in range(len(single_losses)):
                 relative_inv_rates.append(torch.tensor([1.], device=self.device_))

        for i in range(len(single_losses)):
            # logging
            self.task_batch_progress_rates[i].append(relative_inv_rates[i].clone().detach().cpu())

        # desired gradient norms
        desired_norms = []
        alpha = 1.0
        for i in range(len(single_losses)):
            desired_norms.append(avg_grad_norm * relative_inv_rates[i]**alpha)
        
        # adapt weights to meet or get closer to desired gradient norms
        for i, w in enumerate(self.weights):
            w.grad = grad_norms[i]/w
            if grad_norms[i] < desired_norms[i]:
                w.grad *= -1.0
        
        self.weights_optimizer.step()
        # don't let the weights get too small
        for i, w in enumerate(self.weights):
            if w < 0.01:
                w -= (w-0.01)
        self.weights_optimizer.zero_grad()

        # recompute gradients with new weights
        loss = self.loss_func_(fX,
                               target,
                               weight=self.weights,
                               mask=mask,
                               return_separate_losses=False)
        return loss

    def batch_loss(self, batch, grad_norm=False):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)
            ['mask'] (`numpy.ndarray`, optional)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """

        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)
        fX = self.model_(X)

        mask = None
        if self.task_.is_multiclass_classification:

            fX = fX.view((-1, self.n_classes_))

            target = torch.tensor(
                batch['y'],
                dtype=torch.int64,
                device=self.device_).contiguous().view((-1, ))

            if 'mask' in batch:
                mask = torch.tensor(
                    batch['mask'],
                    dtype=torch.float32,
                    device=self.device_).contiguous().view((-1, ))


        elif self.task_.is_multilabel_classification or \
             self.task_.is_regression:
            target = torch.tensor(
                batch['y'],
                dtype=torch.float32,
                device=self.device_)

            if 'mask' in batch:
                mask = torch.tensor(
                    batch['mask'],
                    dtype=torch.float32,
                    device=self.device_)

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)
        
        if grad_norm and (self.epoch_ > 1):
            self.batch_ += 1
            loss = self.grad_norm(fX, target, mask, progress_weighting=True)
            return {'loss':loss}

        return {
            'loss': self.loss_func_(fX, target,
                                    weight=weight,
                                    mask=mask),
        }
