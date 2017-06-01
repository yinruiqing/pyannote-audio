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
# Hervé BREDIN - http://herve.niderb.fr

from pyannote.audio.generators.periodic import PeriodicFeaturesMixin
from pyannote.core import SlidingWindowFeature
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.databse.util import get_annotated
import numpy as np



class GenderSegmentationBatchGenerator(PeriodicFeaturesMixin,
                                       FileBasedBatchGenerator):

    def __init__(self, feature_extractor,
                 duration=3.2, step=0.8, batch_size=32):

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step

        # yield sliding segments on the 'annotated' part
        segment_generator = SlidingSegments(duration=duration,
                                            step=step,
                                            source='annotated')
        super(GenderSegmentationBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

        self.mapping_ = {'male': 1, 'female': 2, 'unknown': 3}

    def signature(self):

        shape = self.shape

        return [
            {'type': 'ndarray', 'shape': shape},
            {'type': 'ndarray', 'shape': (shape[0], 4)}
        ]

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

        y = np.zeros((n_samples + 4, 4), dtype=np.int8)
        # [1,0,0,0] ==> non-speech
        # [0,1,0,0] ==> male
        # [0,0,1,0] ==> female
        # [0,0,0,1] ==> unknown

        annotated = get_annotated(current_file)
        annotation = current_file['annotation']

        support = annotation.get_timeline().support()

        # iterate over non-speech regions
        for non_speech in support.gaps(annotated):
            indices = sw.crop(non_speech, mode='loose')
            y[indices, 0] = 1

        # iterate over speech regions
        for segment, _, label in annotation.itertracks(yield_label=True):
            indices = sw.crop(segment, mode='loose')
            y[indices, self.mapping_[label]] = 1

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
