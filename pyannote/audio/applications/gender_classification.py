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

"""
Speech activity detection

Usage:
  pyannote-speech-detection train [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
  pyannote-speech-detection apply [--database=<db.yml> --subset=<subset>] <tune_dir> <database.task.protocol>
  pyannote-speech-detection -h | --help
  pyannote-speech-detection --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  <train_dir>                Set path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  <tune_dir>                 Set path to the directory containing optimal
                             hyper-parameters (i.e. the output of "tune" mode).
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             In "train" mode, default subset is "train".
                             In "tune" mode, default subset is "development".
                             In "apply" mode, default subset is "test".
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file <db.yml>:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the architecture of the neural
    network used for sequence labeling (0 vs. 1, non-speech vs. speech), the
    feature extraction process (e.g. MFCCs) and the sequence generator used for
    both training and testing.

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives

    architecture:
       name: StackedLSTM
       params:                       # this experiments relies
         n_classes: 4                # on one LSTM layer (16 outputs)
         lstm: [16]                  # and one dense layer.
         mlp: [16]                   # LSTM is bidirectional
         bidirectional: True

    sequences:
       duration: 3.2                 # this experiments relies
       step: 0.8                     # on sliding windows of 3.2s
                                     # with a step of 0.8s
       batch_size: 1024
    ...................................................................

"train" mode:
    First, one should train the raw sequence labeling neural network using
    "train" mode. This will create the following directory that contains
    the pre-trained neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "tune" mode.

"apply" mode
    Finally, one can apply speech activity detection using "apply" mode.
    This will create the following files that contains the hard (mdtm) and
    soft (h5) outputs of speech activity detection.

        <tune_dir>/apply/<database.task.protocol>.<subset>.mdtm
        <tune_dir>/apply/{database}/{uri}.h5

    This means that file whose unique resource identifier is {uri} has been
    processed.

"""

import io
import yaml
import time
import warnings
from os.path import dirname, isfile, expanduser
import numpy as np
from collections import Counter

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.generators.speech import \
    GenderSegmentationBatchGenerator
from pyannote.audio.optimizers import SSMORMS3

from pyannote.audio.callback import LoggingCallback

from pyannote.audio.labeling.aggregation import SequenceLabelingAggregation
from pyannote.audio.signal import Argmax
from pyannote.database.util import get_unique_identifier
from pyannote.database.util import get_annotated
from pyannote.database import get_protocol

from pyannote.metrics.binary_classification import det_curve

from pyannote.parser import MDTMParser

from pyannote.audio.util import mkdir_p
import pyannote.core.json

from pyannote.audio.features.utils import Precomputed
import h5py

from .base import Application

from tqdm import tqdm

import skopt
import skopt.space
from pyannote.metrics.detection import DetectionErrorRate


class SpeechActivityDetection(Application):

    # created by "train" mode
    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    APPLY_DIR = '{tune_dir}/apply'

    # created by "validate" mode
    VALIDATE_DIR = '{train_dir}/validate/{protocol}'
    VALIDATE_TXT = '{validate_dir}/{subset}.eer.txt'
    VALIDATE_TXT_TEMPLATE = '{epoch:04d} {eer:5f}\n'
    VALIDATE_PNG = '{validate_dir}/{subset}.eer.png'
    VALIDATE_EPS = '{validate_dir}/{subset}.eer.eps'

    # created by "tune" mode
    TUNE_DIR = '{train_dir}/tune/{protocol}.{subset}'
    TUNE_YML = '{tune_dir}/tune.yml'
    TUNE_PNG = '{tune_dir}/tune.png'

    HARD_MDTM = '{apply_dir}/{protocol}.{subset}.mdtm'

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None):
        experiment_dir = dirname(dirname(train_dir))
        speech_activity_detection = cls(experiment_dir, db_yml=db_yml)
        speech_activity_detection.train_dir_ = train_dir
        return speech_activity_detection

    @classmethod
    def from_tune_dir(cls, tune_dir, db_yml=None):
        train_dir = dirname(dirname(tune_dir))
        speech_activity_detection = cls.from_train_dir(train_dir,
                                                       db_yml=db_yml)
        speech_activity_detection.tune_dir_ = tune_dir
        return speech_activity_detection

    def __init__(self, experiment_dir, db_yml=None):

        super(SpeechActivityDetection, self).__init__(
            experiment_dir, db_yml=db_yml)

        # architecture
        architecture_name = self.config_['architecture']['name']
        models = __import__('pyannote.audio.labeling.models',
                            fromlist=[architecture_name])
        Architecture = getattr(models, architecture_name)
        self.architecture_ = Architecture(
            **self.config_['architecture'].get('params', {}))

    def train(self, protocol_name, subset='train'):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        # sequence batch generator
        batch_size = self.config_['sequences'].get('batch_size', 8192)
        duration = self.config_['sequences']['duration']
        step = self.config_['sequences']['step']
        batch_generator = GenderSegmentationBatchGenerator(
            self.feature_extraction_, duration=duration, step=step,
            batch_size=batch_size)
        batch_generator.cache_preprocessed_ = self.cache_preprocessed_

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        # total train duration
        train_total = protocol.stats(subset)['annotated']
        # number of batches per epoch
        steps_per_epoch = int(np.ceil((train_total / step) / batch_size))

        # input shape (n_frames, n_features)
        input_shape = batch_generator.shape

        # generator that loops infinitely over all training files
        train_files = getattr(protocol, subset)()
        generator = batch_generator(train_files, infinite=True)

        labeling = SequenceLabeling()
        labeling.fit(input_shape, self.architecture_,
                     generator, steps_per_epoch, 1000,
                     optimizer=SSMORMS3(), log_dir=train_dir)

        return labeling

    def apply(self, protocol_name, subset='test'):

        apply_dir = self.APPLY_DIR.format(tune_dir=self.tune_dir_)

        mkdir_p(apply_dir)

        # load tuning results
        tune_yml = self.TUNE_YML.format(tune_dir=self.tune_dir_)
        with io.open(tune_yml, 'r') as fp:
            self.tune_ = yaml.load(fp)

        # load model for epoch 'epoch'
        epoch = self.get_epochs(self.train_dir_) - 1
        sequence_labeling = SequenceLabeling.from_disk(
            self.train_dir_, epoch)

        # initialize sequence labeling
        duration = self.config_['sequences']['duration']
        step = self.config_['sequences']['step']
        aggregation = SequenceLabelingAggregation(
            sequence_labeling, self.feature_extraction_,
            duration=duration, step=step)

        # initialize protocol
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for i, item in enumerate(getattr(protocol, subset)()):

            prediction = aggregation.apply(item)

            if i == 0:
                # create metadata file at root that contains
                # sliding window and dimension information
                path = Precomputed.get_config_path(apply_dir)
                f = h5py.File(path)
                f.attrs['start'] = prediction.sliding_window.start
                f.attrs['duration'] = prediction.sliding_window.duration
                f.attrs['step'] = prediction.sliding_window.step
                f.attrs['dimension'] = 4
                f.close()

            path = Precomputed.get_path(apply_dir, item)

            # create parent directory
            mkdir_p(dirname(path))

            f = h5py.File(path)
            f.attrs['start'] = prediction.sliding_window.start
            f.attrs['duration'] = prediction.sliding_window.duration
            f.attrs['step'] = prediction.sliding_window.step
            f.attrs['dimension'] = 4
            f.create_dataset('features', data=prediction.data)
            f.close()

        # initialize binarizer
        argmax = Argmax()

        precomputed = Precomputed(root_dir=apply_dir)

        writer = MDTMParser()
        path = self.HARD_MDTM.format(apply_dir=apply_dir, protocol=protocol_name,
                                subset=subset)
        with io.open(path, mode='w') as gp:
            for item in getattr(protocol, subset)():
                prediction = precomputed(item)
                result = argmax.apply(prediction)
                writer.write(result, f=gp, uri=item['uri'], modality='gender')

def main():

    arguments = docopt(__doc__, version='Speech activity detection')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']
        if subset is None:
            subset = 'train'
        application = SpeechActivityDetection(experiment_dir, db_yml=db_yml)
        application.train(protocol_name, subset=subset)

    if arguments['validate']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        application = SpeechActivityDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.validate(protocol_name, subset=subset)

    if arguments['tune']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        application = SpeechActivityDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.tune(protocol_name, subset=subset)

    if arguments['apply']:
        tune_dir = arguments['<tune_dir>']
        if subset is None:
            subset = 'test'
        application = SpeechActivityDetection.from_tune_dir(
            tune_dir, db_yml=db_yml)
        application.apply(protocol_name, subset=subset)
