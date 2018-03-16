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
assorted ML learners and predictors (linear classification)

Author: Yoav Goldberg (yoav.goldberg@gmail.com)
"""

import sys
import math

from stdlib cimport *


cdef class DoublesArr:
   cdef double* vals
   cdef public int n
   def __cinit__(self, list nums):
      cdef double n
      cdef int i
      self.n = len(nums)
      self.vals=<double *>malloc(sizeof(double)*self.n)
      for i in xrange(self.n):
         n = float(nums[i])
         self.vals[i] = n
   def __dealloc__(self):
      free(self.vals)

cdef class MulticlassModel: 

   cdef dict W
   cdef DoublesArr biases # specific type?
   cdef int nclas
   cdef double* scores
   cdef int probs_output

   cdef load(self,fname):
      sys.stderr.write("loading model %s" % fname)
      for line in file(fname):
         f,ws = line.strip().split(None,1)
         self.W[f]=DoublesArr([float(w) for w in ws.split()])
      try:
         self.biases = self.W['**BIAS**']
      except KeyError:
         self.biases = self.W.itervalues().next()
         for i in xrange(self.biases.n):
            self.biases.vals[i]=0

      self.nclas = self.biases.n

      self.scores=<double *>malloc(sizeof(double)*self.nclas)

   def __init__(self, fname, probs_output=False):
      self.W = {}
      self.probs_output=probs_output
      self.load(fname)
      sys.stderr.write(" done\n")

   def __dealloc__(self):
      free(self.scores)

   cpdef object predict(self,list features):
      cdef int i
      cdef double w
      cdef DoublesArr ws
      for i in xrange(self.nclas):
         self.scores[i]=self.biases.vals[i]
      for f in features:
         try:
            i=0
            ws = self.W[f]
            for i in xrange(self.nclas):
               self.scores[i]+=ws.vals[i]
         except KeyError: pass
      cdef double tot = 0
      if self.probs_output:
         for i in xrange(self.nclas): 
            self.scores[i]=math.exp(self.scores[i])
            tot+=self.scores[i]
      res=[]
      cdef int besti=0
      cdef double best=0
      for i in xrange(self.nclas):
         if self.scores[i] > best:
            best = self.scores[i]
            besti = i
         if self.probs_output:
            res.append(self.scores[i]/tot)
         else:
            res.append(self.scores[i])
      return besti,res

   cpdef object predict_r(self,list features): #@@TODO fix
      cdef int i
      cdef double w
      cdef double v
      for i in xrange(self.nclas):
         self.scores[i]=self.biases.vals[i]
      for f,v in features:
         try:
            i=0
            ws = self.W[f]
            for i in xrange(self.nclas):
               self.scores[i]+=ws.vals[i]*v
         except KeyError: pass
      cdef double tot = 0
      if self.probs_output:
         for i in xrange(self.nclas): 
            self.scores[i]=math.exp(self.scores[i])
            tot+=self.scores[i]
      res=[]
      cdef int besti=0
      cdef double best=0
      for i in xrange(self.nclas):
         if self.scores[i] > best:
            best = self.scores[i]
            besti = i
         if self.probs_output:
            res.append(self.scores[i]/tot)
         else:
            res.append(self.scores[i])
      return besti,res

   cpdef object get_scores(self,list features):
      cdef int i
      cdef DoublesArr ws
      cdef list res
      for i in xrange(self.nclas):
         self.scores[i]=self.biases.vals[i]
      for f in features:
         try:
            ws = self.W[f]
            for i in xrange(ws.n):
               self.scores[i]+=ws.vals[i]
         except KeyError: pass
      res=[]
      for i in xrange(self.nclas):
         res.append(self.scores[i])
      return res

   cpdef list get_scores_r(self,list features): #@@TODO FIX
      """
      like get_scores but with real values features
         each feature is a pair (f,v), where v is the value.
      """
      cdef int i
      cdef double w
      cdef double v
      cdef list res
      for i in xrange(self.nclas):
         self.scores[i]=self.biases.vals[i]
      for f,v in features:
         try:
            ws = self.W[f]
            for i in xrange(ws.n):
               self.scores[i]+=ws.vals[i]*v
         except KeyError: pass
      res=[]
      for i in xrange(self.nclas):
         res.append(self.scores[i])
      return res
   #}}}

### Model trainers {{{

cdef class MulticlassParamData:
   cdef:
      double *acc
      double *w
      int *lastUpd
   def __cinit__(self, int nclasses):
      cdef int i
      self.lastUpd = <int *>malloc(nclasses*sizeof(int))
      self.acc     = <double *>malloc(nclasses*sizeof(double))
      self.w       = <double *>malloc(nclasses*sizeof(double))
      for i in range(nclasses):
         self.lastUpd[i]=0
         self.acc[i]=0
         self.w[i]=0

   def __dealloc__(self):
      free(self.lastUpd)
      free(self.acc)
      free(self.w)

cdef class MultitronParameters:
   cdef:
      int nclasses
      int now
      dict W

      double* scores # (re)used in calculating prediction
   
   def __cinit__(self, nclasses):
      self.scores = <double *>malloc(nclasses*sizeof(double))

   cpdef getW(self, clas): 
      d={}
      cdef MulticlassParamData p
      for f,p in self.W.iteritems():
         d[f] = p.w[clas]
      return d

   def __init__(self, nclasses):
      self.nclasses = nclasses
      self.now = 0
      self.W = {}

   cdef _tick(self):
      self.now=self.now+1

   def tick(self): self._tick()

   cpdef scalar_multiply(self, double scalar):
      """
      note: DOES NOT support averaging
      """
      cdef MulticlassParamData p
      cdef int c
      for p in self.W.values():
         for c in xrange(self.nclasses):
            p.w[c]*=scalar

   cpdef add(self, list features, int clas, double amount):
      cdef MulticlassParamData p
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
         p.w[clas]+=amount
         p.lastUpd[clas]=self.now

   cpdef add_r(self, list features, int clas, double amount):
      """
      like "add", but with real values features: 
         each feature is a pair (f,v), where v is the value.
      """
      cdef MulticlassParamData p
      cdef double v
      cdef str f
      for f,v in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
         p.w[clas]+=amount*v
         p.lastUpd[clas]=self.now

   cpdef set(self, list features, int clas, double amount):
      """
      like "add", but replaces instead of adding
      """
      cdef MulticlassParamData p
      cdef double v
      cdef str f
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
         p.w[clas]+=amount
         p.lastUpd[clas]=self.now

   cpdef add_params(self, MultitronParameters other, double factor):
      """
      like "add", but with data from another MultitronParameters object.
      they must both share the number of classes
      add each value * factor
      """
      cdef MulticlassParamData p
      cdef MulticlassParamData op
      cdef double v
      cdef str f
      cdef int clas
      assert(self.nclasses==other.nclasses),"incompatible number of classes in add_params"
      for f,op in other.W.items():
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         for clas in xrange(self.nclasses):
            if op.w[clas]<0.0000001: continue
            p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
            #print p.w[clas], op.w[clas]
            p.w[clas]+=(op.w[clas]*factor)
            p.lastUpd[clas]=self.now

   cpdef do_pa_update(self, list feats, int gold_cls, double C=1.0):
      cdef double go_scr
      cdef double gu_scr
      cdef double loss
      cdef double norm
      cdef double tau
      cdef int prediction
      cdef dict scores
      self._tick()
      prediction = self._predict_best_class(feats)
      if prediction==gold_cls: return prediction
      scores = self.get_scores(feats)
      go_scr = scores[gold_cls]
      gu_scr = scores[prediction]

      loss = gu_scr - go_scr + 1
      norm = len(feats)+len(feats)
      tau = loss / norm
      if tau>C: tau=C
      self.add(feats,prediction,-tau)
      self.add(feats,gold_cls,+tau)
      return prediction

   cpdef pa_update(self, object gu_feats, object go_feats, int gu_cls, int go_cls,double C=1.0):
      cdef double go_scr
      cdef double gu_scr
      cdef double loss
      cdef double norm
      cdef double tau
      go_scr = self.get_scores(go_feats)[go_cls]
      gu_scr = self.get_scores(gu_feats)[gu_cls]
      loss = gu_scr - go_scr + 1
      norm = len(go_feats)+len(gu_feats)
      tau = loss / norm
      if tau>C: tau=C
      self.add(gu_feats,gu_cls,-tau)
      self.add(go_feats,go_cls,+tau)

   cpdef get_scores(self, features):
      cdef MulticlassParamData p
      cdef int i
      cdef double w
      for i in xrange(self.nclasses):
         self.scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            for c in xrange(self.nclasses):
               self.scores[c] += p.w[c]
         except KeyError: pass
      cdef double tot = 0
      res={}
      for i in xrange(self.nclasses):
         res[i] = self.scores[i]
      return res

   cpdef get_scores_r(self, features):
      """
      like get_scores but with real values features
         each feature is a pair (f,v), where v is the value.
      """
      cdef MulticlassParamData p
      cdef int i
      cdef double w
      cdef double v
      for i in xrange(self.nclasses):
         self.scores[i]=0
      for f,v in features:
         try:
            p = self.W[f]
            for c in xrange(self.nclasses):
               self.scores[c] += p.w[c]*v
         except KeyError: pass
      cdef double tot = 0
      res={}
      for i in xrange(self.nclasses):
         res[i] = self.scores[i]
      return res

   def update(self, correct_class, features):
      """
      does a prediction, and a parameter update.
      return: the predicted class before the update.
      """
      self._tick()
      prediction = self._predict_best_class(features)
      if prediction != correct_class:
         self._update(correct_class, prediction, features)
      return prediction

   cpdef predict_best_class_r(self, list features):
      scores = self.get_scores_r(features)
      scores = [(s,c) for c,s in scores.iteritems()]
      s = max(scores)[1]
      return max(scores)[1]

   def update_r(self, correct_class, features):
      self._tick()
      prediction = self.predict_best_class_r(features)
      if prediction != correct_class:
         self._update_r(correct_class, prediction, features)
      return prediction

   cdef int _predict_best_class(self, list features):
      cdef int i
      cdef MulticlassParamData p
      for i in range(self.nclasses):
         self.scores[i]=0
      for f in features:
         #print "lookup", f
         try:
            p = self.W[f]
            for c in xrange(self.nclasses):
               self.scores[c] += p.w[c]
         except KeyError: 
            #print "feature",f,"not found"
            pass
      # return best_i
      cdef int best_i = 0
      cdef double best = self.scores[0]
      for i in xrange(1,self.nclasses):
         if best < self.scores[i]:
            best_i = i
            best = self.scores[i]
      return best_i

   cdef _update(self, int goodClass, int badClass, list features):
      cdef MulticlassParamData p
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[badClass]+=(self.now-p.lastUpd[badClass])*p.w[badClass]
         p.acc[goodClass]+=(self.now-p.lastUpd[goodClass])*p.w[goodClass]
         p.w[badClass]-=1.0
         p.w[goodClass]+=1.0
         p.lastUpd[badClass]=self.now
         p.lastUpd[goodClass]=self.now

   cdef _update_r(self, int goodClass, int badClass, list features):
      cdef MulticlassParamData p
      for f,v in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[badClass]+=(self.now-p.lastUpd[badClass])*p.w[badClass]
         p.acc[goodClass]+=(self.now-p.lastUpd[goodClass])*p.w[goodClass]
         p.w[badClass]-=v
         p.w[goodClass]+=v
         p.lastUpd[badClass]=self.now
         p.lastUpd[goodClass]=self.now

   def finalize(self):
      cdef MulticlassParamData p
      # average
      for f in self.W.keys():
         for c in xrange(self.nclasses):
            p = self.W[f]
            p.acc[c]+=(self.now-p.lastUpd[c])*p.w[c]
            p.w[c] = p.acc[c] / self.now

   def dump(self, out=sys.stdout):
      cdef MulticlassParamData p
      for f in self.W.keys():
         out.write("%s" % f)
         for c in xrange(self.nclasses):
            p = self.W[f]
            out.write(" %s" % p.w[c])
         out.write("\n")

   def dump_fin(self,out=sys.stdout):
      cdef MulticlassParamData p
      # write the average
      for f in self.W.keys():
         out.write("%s" % f)
         for c in xrange(self.nclasses):
            p = self.W[f]
            out.write(" %s " % ((p.acc[c]+((self.now-p.lastUpd[c])*p.w[c])) / self.now))
         out.write("\n")

##################
cdef class ParamData:
   cdef:
      double acc
      double w
      int lastUpd
   def __init__(self):
      self.acc=0
      self.w=0
      self.lastUpd=0

cdef class PerceptronParameters:
   cdef:
      int now
      dict W

   def __init__(self):
      self.now = 0
      self.W = {}

   def tick(self):
      self.now+=1

   def score(self,features):
      return self._score(features)

   cdef _score(self, features):
      cdef ParamData p
      cdef double score=0
      for f in features:
         try:
            p = self.W[f]
            score += p.w
         except KeyError: 
            pass
      return score

   cpdef updateFeatures(self, features, double amount):
      cdef ParamData p
      cdef str f
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = ParamData()
            self.W[f] = p
         p.acc+=(self.now-p.lastUpd)*p.w
         p.w+=amount
         p.lastUpd=self.now

   def finalize(self):
      cdef ParamData p
      # average
      for f in self.W.keys():
         p = self.W[f]
         p.acc+=(self.now-p.lastUpd)*p.w
         p.w = p.acc / self.now

   def dump(self, out=sys.stdout):
      cdef ParamData p
      for f,p in self.W.iteritems():
         out.write("%s" % f)
         out.write(" %s" % p.w)
         out.write("\n")

   def dump_fin(self, out=sys.stdout):
      cdef ParamData p
      for f,p in self.W.iteritems():
         out.write("%s" % f)
         out.write(" %s " % ((p.acc+((self.now-p.lastUpd)*p.w)) / self.now))
         out.write("\n")

cdef class _float:
   cdef:
      float val

cdef _float makef(float f):
   cdef _float mf = _float.__new__(_float)
   mf.val = f
   return mf
   
cdef class LinearModel:
   cdef:
      dict W

   def __init__(self):
      self.W = {}

   def __cinit__(self):
      self.W = {}

   cpdef _load(self, fh):
      for line in fh:
         if not line.strip(): break
         f,w = line.strip().split()
         #self.W[f]=float(w)
         self.W[f]=makef(float(w))
      return self
   
   @classmethod
   def from_file(cls, fname):
      sys.stderr.write("loading model %s" % fname)
      o=cls()
      o._load(file(fname))
      sys.stderr.write(" done\n")
      return o

   @classmethod
   def from_fh(cls, fh):
      o=cls()
      o._load(fh)
      return o


   def score(self,features):
      return self._score(features)

   cdef double _score(self, list features):
      cdef double score=0
      cdef str f
      cdef _float v
      for f in features:
         try:
            v = self.W[f]
            score += v.val
            #score += self.W[f]
         except KeyError: 
            pass
      return score

#}}}


cdef class MultipleVectorsMulticlassModel: #{{{
   def __init__(self):pass
   def get_scores(self, features):
      """
      just an "interface" 

      features: a list of feature vectors, one per class
      """
      raise "Not Implemented"
#}}}

cdef class MultipleVectorsMulticlassParams(MultipleVectorsMulticlassModel):
   cdef:
      list params # TODO change to specific type..
      int    nclasses

   def __init__(self, nclasses):
      self.nclasses=nclasses
      self.params=[PerceptronParameters() for x in xrange(nclasses)]

   def __cinit__(self, int nclasses):
      self.nclasses=nclasses
      self.params=[PerceptronParameters() for x in xrange(nclasses)]

   cpdef add(self, list features, int clas, double amount):
      cdef PerceptronParameters params
      params=self.params[clas]
      features=features[clas]
      params.updateFeatures(features, amount)

   cpdef tick(self):
      [p.tick() for p in self.params]

   cpdef get_scores(self, list features):
      cdef dict scores={}
      cdef int i
      cdef PerceptronParameters pars
      for i in xrange(self.nclasses):
         pars = self.params[i]
         scores[i]=pars._score(features[i])
      return scores

   def dump(self, out=sys.stdout):
      out.write("%s\n" % self.nclasses)
      cdef PerceptronParameters p
      for p in self.params:
         p.dump(out)
         out.write("\n")

   def dump_fin(self, out=sys.stdout):
      out.write("%s\n" % self.nclasses)
      cdef PerceptronParameters p
      for p in self.params:
         p.dump_fin(out)
         out.write("\n")

cdef class MulticlassLinearModel(MultipleVectorsMulticlassModel):
   cdef:
      list linearmodels  # TODO specific type?
      int _nclasses

   def __init__(self):
      self.linearmodels=[]

   def _addmodel(self, m):
      self.linearmodels.append(m)
      self._nclasses = len(self.linearmodels)

   cpdef int nclasses(self):
      return self._nclasses

   @classmethod
   def from_file(cls, fname):
      o=cls()
      fh=file(fname)
      nclasses=int(fh.next())
      for i in xrange(nclasses):
         model=LinearModel.from_fh(fh)
         o._addmodel(model)
      return o

   cpdef get_scores(self, list features):
      cdef dict scores={}
      cdef int i
      cdef LinearModel lm
      cdef list feats
      for i in xrange(self._nclasses):
         lm = self.linearmodels[i]
         feats = features[i]
         scores[i]=lm._score(feats)
      return scores

