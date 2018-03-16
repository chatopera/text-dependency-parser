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
from optparse import OptionParser
from features import extractors

parser = OptionParser()
parser.add_option("-o", "--trainfile", dest="trainfile", default=None)
parser.add_option(
    "-f",
    "--externaltrainfile",
    dest="externaltrainfile",
    default=None)
parser.add_option("-m", "--model", dest="modelfile", default="data/weights")
parser.add_option(
    "-e",
    "--eval",
    dest="eval",
    action="store_true",
    default=False)
parser.add_option(
    "-i",
    "--ignorepunc",
    dest="ignore_punc",
    action="store_true",
    default=False)
parser.add_option(
    "-p",
    "--onlyprojective",
    dest="only_proj",
    action="store_true",
    default=False)
parser.add_option(
    "-s",
    "--scores_out",
    dest="SCORES_OUT",
    action="store_true",
    default=False)
parser.add_option(
    "-r",
    "--reverse",
    dest="REVERSE",
    action="store_true",
    default=False)
parser.add_option(
    "--features",
    dest="feature_extarctor",
    default="standard.wenbin")

opts, args = parser.parse_args()

if opts.trainfile:
    MODE = 'train'
    TRAIN_OUT_FILE = opts.trainfile
elif opts.externaltrainfile:
    MODE = 'write'
    TRAIN_OUT_FILE = opts.externaltrainfile

else:
    MODE = 'test'

if opts.SCORES_OUT:
    scores_out = file("standard.scores", "w")

DATA_FILE = args[0]

########

import sys
import ml

from pio import io
from transitionparser import *

featExt = extractors.get(opts.feature_extarctor)

sents = list(io.conll_to_sents(file(DATA_FILE)))  # [:20]

if opts.REVERSE:
    sents = [list(reversed(sent)) for sent in sents]

if opts.only_proj:
    import isprojective
    sents = [s for s in sents if isprojective.is_projective(s)]
if MODE == "write":
    fout = file(TRAIN_OUT_FILE, "w")
    trainer = LoggingActionDecider(ArcStandardParsingOracle(), featExt, fout)
    p = ArcStandardParser2(trainer)
    for i, sent in enumerate(sents):
        sys.stderr.write(".")
        sys.stderr.flush()
        d = p.parse(sent)
    sys.exit()

if MODE == "train":
    fout = file(TRAIN_OUT_FILE, "w")
    trainer = MLTrainerActionDecider(
        ml.MultitronParameters(3),
        ArcStandardParsingOracle(),
        featExt)
    p = ArcStandardParser2(trainer)
    import random
    random.seed("seed")
    random.shuffle(sents)
    for x in xrange(10):
        print "iter ", x
        for i, sent in enumerate(sents):
            if i % 500 == 0:
                print i,
            try:
                d = p.parse(sent)
            except Exception as e:
                print "prob in sent:", i
                print "\n".join(["%s %s %s %s" % (t['id'], t['form'], t['tag'], t['parent']) for t in sent])
                raise e
    trainer.save(fout)
    sys.exit()

elif MODE == "test":
    p = ArcStandardParser2(
        MLActionDecider(
            ml.MulticlassModel(
                opts.modelfile,
                True),
            featExt))

good = 0.0
bad = 0.0
complete = 0.0

for i, sent in enumerate(sents):
    mistake = False
    sgood = 0.0
    sbad = 0.0
    sys.stderr.write("%s %s %s\n" % ("@@@", i, good / (good + bad + 1)))
    try:
        d = p.parse(sent)
    except MLTrainerWrongActionException:
        continue
    if opts.REVERSE:
        sent = list(reversed(sent))
    sent = d.annotate(sent)
    for tok in sent:
        print tok['id'], tok['form'], "_", tok['tag'], tok['tag'], "_", tok['pparent'], "_ _ _"
        if opts.ignore_punc and tok['form'][0] in "`',.-;:!?{}":
            continue
        if tok['parent'] == tok['pparent']:
            good += 1
            sgood += 1
        else:
            bad += 1
            sbad += 1
            mistake = True
    print
    if not mistake:
        complete += 1
    if opts.SCORES_OUT:
        scores_out.write("%s\n" % (sgood / (sgood + sbad)))

if opts.SCORES_OUT:
    scores_out.close()

if opts.eval:
    print "accuracy:", good / (good + bad)
    print "complete:", complete / len(sents)
