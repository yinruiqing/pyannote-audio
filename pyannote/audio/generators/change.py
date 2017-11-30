#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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
# Ruiqing YIN
# Hervé BREDIN - http://herve.niderb.fr


from pyannote.audio.generators.periodic import PeriodicFeaturesMixin
from pyannote.core import SlidingWindowFeature
from pyannote.core import Segment, Timeline
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.database.util import get_annotated
import numpy as np
from scipy import signal


class ChangeDetectionBatchGenerator(PeriodicFeaturesMixin,
                                    FileBasedBatchGenerator):

    """(X_batch, y_batch) batch generator

    Yields batches made of subsequences obtained using a sliding
    window over the audio files.

    Parameters
    ----------
    feature_extractor : YaafeFeatureExtractor
    duration: float, optional
        yield segments of length `duration`
        Defaults to 3.2s.
    step: float, optional
        step of sliding window (in seconds).
        Defaults to 0.8s.
    balance: float, optional
        Artificially increase the number of positive labels by
        labeling as positive every frame in the direct neighborhood
        (less than balance seconds apart) of each change point.
        Defaults to 0.01s (10 ms).
    batch_size: int, optional
        Size of batch
        Defaults to 32

    Returns
    -------
    X_batch : (batch_size, n_samples, n_features) numpy array
        Batch of feature sequences
    y_batch : (batch_size, n_samples) numpy array
        Batch of corresponding label sequences

    Usage
    -----
    >>> batch_generator = ChangeDetectionBatchGenerator(feature_extractor)
    >>> for X_batch, y_batch in batch_generator.from_file(current_file):
    ...     # do something with
    """

    def __init__(self, feature_extractor, fuzzy=False,
                 balance=0.01, duration=3.2, step=0.8, batch_size=32):

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step
        self.balance = balance
        self.fuzzy = fuzzy
        segment_generator = SlidingSegments(duration=duration,
                                            step=step,
                                            source='annotated')
        super(ChangeDetectionBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

    def signature(self):

        shape = self.shape

        return [
            {'type': 'ndarray', 'shape': shape},
            {'type': 'ndarray', 'shape': (shape[0], 2)}
        ]

    @property
    def loss(self):
        if self.fuzzy:
            return 'mean_squared_error'
        else:
            return 'binary_crossentropy'

    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X and y"""

        # extract features for the whole file
        # (if it has not been done already)
        current_file = self.periodic_preprocess(
            current_file, identifier=identifier)

        # if labels have already been extracted, do nothing
        if identifier in self.preprocessed_.setdefault('y', {}):
            return current_file

        # get features as pyannote.core.SlidingWindowFeature instance
        X = self.preprocessed_['X'][identifier]
        sw = X.sliding_window
        n_samples = X.getNumber()

        annotated = get_annotated(current_file)
        annotation = current_file['annotation']

        segments = []
        for segment, _ in annotation.itertracks():
            segments.append(Segment(segment.start - self.balance, segment.start + self.balance))
            segments.append(Segment(segment.end - self.balance, segment.end + self.balance))
        change_part = Timeline(segments).support().crop(annotated, mode='intersection')

        # iterate over change regions
        if self.fuzzy:
            y = np.zeros((n_samples + 4, 1), dtype=np.float)-1
            # iterate over non-change regions
            for non_changes in change_part.gaps(annotated):
                indices = sw.crop(non_changes, mode='loose')
                y[indices,0] = 0
            for changes in change_part:
                indices = sw.crop(changes, mode='loose')
                window_size = len(indices)
                trian_window = signal.triang(window_size)
                orign_window = y[indices,0]
                y[indices,0] = [max(t, o) for t, o in zip(trian_window, orign_window)]
        else:
            y = np.zeros((n_samples + 4, 1), dtype=np.int8)-1
            # iterate over non-change regions
            for non_changes in change_part.gaps(annotated):
                indices = sw.crop(non_changes, mode='loose')
                y[indices,0] = 0
            for changes in change_part:
                indices = sw.crop(changes, mode='loose')
                y[indices,0] = 1

        y = SlidingWindowFeature(y[:-1], sw)
        self.preprocessed_['y'][identifier] = y

        return current_file

    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        """Extract X and y subsequences"""

        X = self.periodic_process_segment(
            segment, signature=signature, identifier=identifier)

        duration = signature.get('duration', None)

        y = self.preprocessed_['y'][identifier].crop(
            segment, mode='center', fixed=duration)

        return [X, y]
