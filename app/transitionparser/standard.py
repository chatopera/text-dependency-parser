#!/usr/bin/env python
# Copyright 2010 Yoav Goldberg
##
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.
"""
driver for ArcStandard parser.

Author: Yoav Goldberg (yoav.goldberg@gmail.com)

training:
   standard.py -o model_file conll_data_file

alternatively, producing feature vectors for training an extarnal classifier:
   standard.py -f vectors_file conll_data_file

testing:
   standard.py -m model conll_test_file
"""

from __future__ import print_function
from __future__ import division

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curdir, os.path.pardir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

from absl import app
from absl import flags
from absl import logging

from ml import ml
from pio import io
from transitionparser.oracles import *
from transitionparser.deciders import *
from transitionparser.parsers import *
from features import extractors
from common.utils import is_projective

FLAGS = flags.FLAGS
'''
General
'''

flags.DEFINE_boolean('ignore_punc', False, 'Ignore Punct File.')  # nopep8
flags.DEFINE_boolean('only_projective', False, 'Only Projective.')  # nopep8
flags.DEFINE_boolean('lazypop', True, 'Lazy pop.')   # nopep8
flags.DEFINE_boolean('unlex', False, 'unlex')   # nopep8
flags.DEFINE_string('feature_extarctor', 'standard.wenbin', 'Feature Extarctor')  # nopep8
flags.DEFINE_string('model', os.path.join(curdir, os.path.pardir, "tmp", "standard.model"), 'Transition Parser Model.')  # nopep8

'''
Train
'''
flags.DEFINE_boolean('train', False, 'Train model with train data')  # nopep8
flags.DEFINE_integer('epoch', 1, 'Train Epoch.')  # nopep8
flags.DEFINE_string('train_data', os.path.join(curdir, os.path.pardir, "data", "conll.example"), 'Train Data')  # nopep8
flags.DEFINE_string('externaltrainfile', None, 'External Train File.')  # nopep8
# flags.DEFINE_string('modelfile', 'data/weights', 'Model File.')

'''
Test
'''
flags.DEFINE_boolean('test', False, 'Evalutate with test data')  # nopep8
flags.DEFINE_string('test_data', os.path.join(curdir, os.path.pardir, "data", "conll.example"), 'Test data.')  # nopep8
flags.DEFINE_string('test_results', os.path.join(curdir, os.path.pardir, "tmp", "standard.test.results"), 'Save scores into disk.')  # nopep8


from optparse import OptionParser
from features import extractors

import sys
import ml
import random
from pio import io
from transitionparser.parsers import *

def transform_conll_sents(conll_file_path):
    '''
    Transform CoNLL data as feeding
    '''
    sents = list(io.conll_to_sents(file(conll_file_path)))

    if FLAGS.only_projective:
        sents = [s for s in sents if is_projective(s)]

    if FLAGS.unlex:
        from shared.lemmatize import EnglishMinimalWordSmoother
        smoother = EnglishMinimalWordSmoother.from_words_file("1000words")
        for sent in sents:
            for tok in sent:
                tok['oform'] = tok['form']
                tok['form'] = smoother.get(tok['form'])

    return sents

def train():
    MODE = 'train'
    featExt = extractors.get(FLAGS.feature_extarctor)
    sents = transform_conll_sents(FLAGS.train_data)
    trainer = MLTrainerActionDecider(
        ml.MultitronParameters(3),
        ArcStandardParsingOracle(),
        featExt)
    p = ArcStandardParser2(trainer)
    total = len(sents)
    # random.seed("seed")
    for x in xrange(FLAGS.epoch):
        random.shuffle(sents)
        logging.info("iter %s/%s", x + 1, FLAGS.epoch)
        logging.info("  shuffle data ...")
        for i, sent in enumerate(sents):
            if i % 500 == 0:
                logging.info("  step %s/%s ...", i, total)
            try:
                d = p.parse(sent)
            except Exception as e:
                logging.info("prob in sent: %s", i)
                logging.info("\n".join(
                    ["%s %s %s %s" % (t['id'], t['form'], t['tag'], t['parent']) for t in sent]))
                raise e

    with open(FLAGS.model, "w") as fout:
        logging.info("save model file to disk [%s] ...", FLAGS.model)
        trainer.save(fout)

def test():
    pass


def main(argv):
    print(
        'Running under Python {0[0]}.{0[1]}.{0[2]}'.format(
            sys.version_info),
        file=sys.stderr)
    if FLAGS.train:
        train()
    if FLAGS.test:
        test()

if __name__ == '__main__':
    # FLAGS([__file__, '--verbosity', '1'])
    app.run(main)
