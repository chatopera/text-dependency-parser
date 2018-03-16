#!/usr/bin/env python
## Copyright 2010 Yoav Goldberg
##
##    This is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This software is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this software.  If not, see <http://www.gnu.org/licenses/>.
"""
driver for ArcEager parser.

Author: Yoav Goldberg (yoav.goldberg@gmail.com)
"""
from features import extractors 
from params import parser

opts, args = parser.parse_args()

if opts.trainfile:
   MODE='train'
   TRAIN_OUT_FILE=opts.trainfile
elif opts.externaltrainfile:
   MODE='write'
   TRAIN_OUT_FILE=opts.externaltrainfile
else:
   MODE='test'

if opts.SCORES_OUT:
   scores_out = file("eager.scores","w")

DATA_FILE=args[0]

########

import sys
from ml import ml

from pio import io
from transitionparser import *

featExt = extractors.get(opts.feature_extarctor)

sents = list(io.conll_to_sents(file(DATA_FILE)))

if opts.only_proj:
   import isprojective
   sents = [s for s in sents if isprojective.is_projective(s)]

if opts.UNLEX:
   from shared.lemmatize import EnglishMinimalWordSmoother
   smoother = EnglishMinimalWordSmoother.from_words_file("1000words")
   for sent in sents:
      for tok in sent:
         tok['oform']=tok['form']
         tok['form'] = smoother.get(tok['form'])

if MODE=="write":
   fout = file(TRAIN_OUT_FILE,"w")
   trainer = LoggingActionDecider(ArcEagerParsingOracle(pop_when_can=opts.POP_WHEN_CAN),featExt,fout)
   p = ArcEagerParser( trainer)
   for i,sent in enumerate(sents):
      sys.stderr.write(". %s " % i)
      sys.stderr.flush()
      d=p.parse(sent)
   sys.exit()


if MODE=="train":
   fout = file(TRAIN_OUT_FILE,"w")
   nactions = 4
   trainer = MLTrainerActionDecider(ml.MultitronParameters(nactions), ArcEagerParsingOracle(pop_when_can=opts.POP_WHEN_CAN), featExt)
   p = ArcEagerParser( trainer)
   import random
   random.seed("seed")
   #random.shuffle(sents)
   for x in xrange(10):
      print "iter ",x
      for i,sent in enumerate(sents):
         if i % 500 == 0: print i,
         try:
            d=p.parse(sent)
         except IndexError,e:
            print "prob in sent:",i
            print "\n".join(["%s %s %s %s" % (t['id'],t['form'],t['tag'],t['parent']) for t in sent])
            raise e
   trainer.save(fout)
   sys.exit()
# test
elif MODE=="test":
   p = ArcEagerParser(MLActionDecider(ml.MulticlassModel(opts.modelfile),featExt))

good = 0.0
bad  = 0.0
complete=0.0

#main test loop
reals = set()
preds = set()

for i,sent in enumerate(sents):
   sgood=0.0
   sbad=0.0
   mistake=False
   sys.stderr.write("%s %s %s\n"% ( "@@@",i,good/(good+bad+1)))
   try:
      d=p.parse(sent)
   except MLTrainerWrongActionException:
      # this happens only in "early update" parsers, and then we just go on to
      # the next sentence..
      continue
   sent = d.annotate_allow_none(sent)
   for tok in sent:
      if opts.ignore_punc and tok['form'][0] in "`',.-;:!?{}": continue
      reals.add((i,tok['parent'],tok['id']))
      preds.add((i,tok['pparent'],tok['id']))
      if tok['pparent']==-1:continue
      if tok['parent']==tok['pparent'] or tok['pparent']==-1:
         good+=1
         sgood+=1
      else:
         bad+=1
         sbad+=1
         mistake=True
   #print 
   if opts.UNLEX:
      io.out_conll(sent,parent='pparent',form='oform')
   else:
      io.out_conll(sent,parent='pparent',form='form')
   if not mistake: complete+=1
   #sys.exit()
   if opts.SCORES_OUT:
      scores_out.write("%s\n" % (sgood/(sgood+sbad)))
  
if opts.SCORES_OUT:
   scores_out.close()

if opts.eval:
   print "accuracy:", good/(good+bad)
   print "complete:", complete/len(sents)
   preds = set([(i,p,c) for i,p,c in preds if p != -1])
   print "recall:", len(preds.intersection(reals))/float(len(reals))
   print "precision:", len(preds.intersection(reals))/float(len(preds))
   print "assigned:",len(preds)/float(len(reals))
   
