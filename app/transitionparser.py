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
Transition based parsing (both arc-standard and arc-eager).
Easily extended to support other variants.
"""
from __future__ import print_function
from __future__ import division

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

from absl import app
from absl import flags
from absl import logging
from collections import defaultdict
import copy
import random
from deps import DependenciesCollection
from ml import ml

from features.extractors import *
from common import *

# misc / pretty print / temp junk

def _ids(tok):
    if tok['id'] == 0:
        tok['form'] = 'ROOT'
    return (tok['id'], tok['tag'])



class MLActionDecider:
    '''
    action deciders / policeis
    '''

    def __init__(self, model, featExt):
        self.m = model
        self.fs = featExt

    def next_action(self, stack, deps, sent, i):
        if len(stack) < 2:
            return SHIFT
        fs = self.fs.extract(stack, deps, sent, i)
        action, scores = self.m.predict(fs)
        if i >= len(sent) and action == SHIFT:
            action = scores.index(max(scores[1:]))
        return action

    def next_actions(self, stack, deps, sent, i, conf=None):
        fs = self.fs.extract(stack, deps, sent, i)
        action, scores = self.m.predict(fs)
        # [-122, 0.3, 3] -> {0:-122, 1:0.3, 2:3}
        scores = dict(enumerate(scores))
        actions = [
            item for item,
            score in sorted(
                scores.items(),
                key=lambda x:-
                x[1])]
        return actions

    def scores(self, conf):  # TODO: who uses this??
        if len(conf.stack) < 2:
            return {SHIFT: 1}

        fs = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        action, scores = self.m.predict(fs)
        # [-122, 0.3, 3] -> {0:-122, 1:0.3, 2:3}
        scores = dict(enumerate(scores))
        if conf.i >= len(conf.sent):
            del scores[SHIFT]
        return scores

    def get_scores(self, conf):
        if len(conf.stack) < 2:
            return {SHIFT: 1}
        fs = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        scores = self.m.get_scores(fs)
        return scores

    def get_prob_scores(self, conf):
        if len(conf.stack) < 2:
            return [1.0, 0, 0]
        fs = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        besti, scores = self.m.predict(fs)
        return scores

    def update(self, stack, deps, sent, i):
        self.m.update(wrong, correct, self.fs.extract(stack, deps, sent, i))


class OracleActionDecider:  # {{{
    def __init__(self, oracle):
        self.o = oracle

    def next_action(self, stack, deps, sent, i):
        return self.o.next_action(stack, deps, sent, i)

    def next_actions(self, stack, deps, sent, i):
        return self.o.next_actions(stack, deps, sent, i)

    def get_scores(self, conf):
        return {self.next_action(conf.stack, conf.deps, conf.sent, conf.i): 1}


class AuditMLActionDecider:  # {{{
    def __init__(self, model, featExt):
        self.m = model
        self.fs = featExt

        self.current_sent = None
        self.childs_of_x = None
        self.connected_childs = set()
        self.idtotok = {}

    def next_action(self, stack, deps, sent, i):
        def _enrich(set_of_node_ids):
            return [_ids(self.idtotok[i]) for i in set_of_node_ids]

        if self.current_sent != sent:
            self.current_sent = sent
            idtotok = {}
            for tok in self.current_sent:
                self.idtotok[tok['id']] = tok
            self.childs_of_x = defaultdict(set)
            self.connected_childs = set([-1])
            for tok in sent:
                self.childs_of_x[tok['parent']].add(tok['id'])

        if len(stack) < 2:
            return SHIFT
        fs = self.fs.extract(stack, deps, sent, i)
        action, scores = self.m.predict(fs)
        logging.debug("action [%s], scores [%s]", action, scores)
        if i >= len(sent) and action == SHIFT:
            action = scores.index(max(scores[1:]))

        if action == REDUCE_R:
            if stack[-1]['parent'] == stack[-2]['id']:
                if len(self.childs_of_x[stack[-1]['id']
                                        ] - self.connected_childs) > 0:
                    logging.error("R not connecting: %s | %s , because: %s", _ids(stack[-1]), _ids(stack[-2]), _enrich(self.childs_of_x[stack[-1]['id']] - self.connected_childs))
                else:
                    logging.error("R not XX")

        if action == REDUCE_L:
            if len(self.childs_of_x[stack[-2]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-2]['id'])
        if action == REDUCE_R:
            if len(self.childs_of_x[stack[-1]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-1]['id'])

        return action, scores

class LoggingActionDecider:  # {{{
    def __init__(self, decider, featExt, out=sys.stdout):
        self.decider = decider
        self.fs = featExt
        self.out = out

    def next_action(self, stack, deps, sent, i):
        features = self.fs.extract(stack, deps, sent, i)
        logging.debug("features [%s]", features) 
        action = self.decider.next_action(stack, deps, sent, i)
        self.out.write("%s %s\n" % (action, " ".join(features)))
        return action

    def next_actions(self, stack, deps, sent, i):
        action = self.next_action(stack, deps, sent, i)
        return [action]

    def save(self, param=None):
        self.out.close()


class MLTrainerWrongActionException(Exception):
    pass


class MLTrainerActionDecider:  # {{{
    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i, conf=None):
        return [self.next_action(stack, deps, sent, i, conf)]

    def next_action(self, stack, deps, sent, i, conf=None):
        action = self.decider.next_action(stack, deps, sent, i)
        mlaction = self.ml.update(
            action, self.fs.extract(
                stack, deps, sent, i))
        if action != mlaction:
            if self.earlyUpdate:
                raise MLTrainerWrongActionException()
        return action

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)

class MLPassiveAggressiveTrainerActionDecider:  # {{{
    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i):
        return [self.next_action(stack, deps, sent, i)]

    def next_action(self, stack, deps, sent, i):
        action = self.decider.next_action(stack, deps, sent, i)
        mlaction = self.ml.do_pa_update(
            self.fs.extract(
                stack, deps, sent, i), action, C=1.0)
        if action != mlaction:
            if self.earlyUpdate:
                raise MLTrainerWrongActionException()
        return action

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)

class MLTrainerActionDecider2:  # {{{
    """
    Like MLTrainerActionDecider but does the update itself (a little less efficient, a bit more informative)
    """

    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i, conf=None):
        return [self.next_action(stack, deps, sent, i, conf)]

    def score_deps(self, deps, sent):
        score = 0
        deps = deps.deps  # a set of (p,c) ids
        sent_deps = set()
        for tok in sent:
            if tok['id'] == 0:
                continue
            sent_deps.add((tok['parent'], tok['id']))
        for pc in sent_deps:
            if pc not in deps:
                score += 0.2
        for pc in deps:
            if pc not in sent_deps:
                score += 1
        return score

    def cum_score_of_action(self, action, conf, ml=False):
        newconf = conf.newAfter(action)
        decider = copy.deepcopy(self.decider)
        while not newconf.is_in_finish_state():
            try:
                if ml:
                    next = self.next_ml_action(newconf)
                else:
                    next = decider.next_action(
                        newconf.stack, newconf.deps, newconf.sent, newconf.i)
                newconf.do_action(next)
            except IllegalActionException:
                assert(len(newconf.sent) == newconf.i)
                break
        return self.score_deps(newconf.deps, newconf.sent)

    def next_ml_action(self, conf):
        features = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]
        return max(act_scores)[1]

    def next_action(self, stack, deps, sent, i, conf=None):
        features = self.fs.extract(stack, deps, sent, i)
        goldaction = self.decider.next_action(stack, deps, sent, i)

        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]
        pred_s, pred_a = max(act_scores)
        self.ml.tick()
        if pred_a != goldaction:
            # calculate cost of NO UPDATE:
            noupdate_cost = self.cum_score_of_action(pred_a, conf, ml=True)

            # now try to update:
            self.ml.add(features, goldaction, 1.0)
            self.ml.add(features, pred_a, -1.0)

            update_cost = self.cum_score_of_action(
                self.next_ml_action(conf), conf, ml=True)
            if noupdate_cost < update_cost:
                logging.debug("noupdate: %s, update: %s", noupdate_cost, update_cost)
                # undo prev update
                self.ml.add(features, goldaction, -1.0)
                self.ml.add(features, pred_a, 1.0)
        return goldaction

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)

class MLTrainerActionDecider3:  # {{{
    """
    Like MLTrainerActionDecider but does the update itself (a little less efficient, a bit more informative)
    """

    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i, conf=None):
        return [self.next_action(stack, deps, sent, i, conf)]

    def score_deps(self, deps, sent):
        score = 0
        deps = deps.deps  # a set of (p,c) ids
        sent_deps = set()
        for tok in sent:
            if tok['id'] == 0:
                continue
            sent_deps.add((tok['parent'], tok['id']))
        for pc in sent_deps:
            if pc not in deps:
                score += 0.2
        for pc in deps:
            if pc not in sent_deps:
                score += 1
        return score

    def cum_score_of_action(self, action, conf, ml=False):
        newconf = conf.newAfter(action)
        decider = copy.deepcopy(self.decider)
        while not newconf.is_in_finish_state():
            try:
                if ml:
                    next = self.next_ml_action(newconf)
                else:
                    next = decider.next_action(
                        newconf.stack, newconf.deps, newconf.sent, newconf.i)
                newconf.do_action(next)
            except IllegalActionException:
                logging.debug("oracle says [%s], but it is illegal, probably at end", next) 
                assert(len(newconf.sent) == newconf.i)
                break
        return self.score_deps(newconf.deps, newconf.sent)

    def next_ml_action(self, conf):
        features = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]
        return max(act_scores)[1]

    def next_action(self, stack, deps, sent, i, conf=None):
        features = self.fs.extract(stack, deps, sent, i)
        goldaction = self.decider.next_action(stack, deps, sent, i)

        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]

        pred_s, pred_a = max(act_scores)
        noupdate_cost = self.cum_score_of_action(pred_a, conf, ml=True)

        if pred_a != SHIFT:
            self.ml.add(features, SHIFT, 1.0)
            self.ml.add(features, pred_a, -1.0)
            shiftupdate_cost = self.cum_score_of_action(SHIFT, conf, ml=True)
            if shiftupdate_cost < noupdate_cost:
                self.ml.tick()
                return SHIFT
            else:  # undo
                self.ml.add(features, SHIFT, -1.0)
                self.ml.add(features, pred_a, 1.0)
                self.ml.tick()
                return pred_a
        self.ml.tick()
        return pred_a

        costs = []
        for score, act in act_scores:
            if pred_a != act:
                self.ml.add(features, act, 1.0)
                self.ml.add(features, pred_a, -1.0)
                costs.append(
                    (self.cum_score_of_action(
                        act, conf, ml=True), act))
                self.ml.add(features, act, -1.0)
                self.ml.add(features, pred_a, 1.0)
            else:
                costs.append((noupdate_cost, pred_a))
        min_cost, act = min(costs)

        if act != pred_a:
            logging.debug("min_cost [%s], noupdate_cost [%s], act [%s], goldaction [%s]", min_cost, noupdate_cost, act, goldaction)
            self.ml.add(features, act, 1.0)
            self.ml.add(features, pred_a, -1.0)
        else:
            pass

        self.ml.tick()
        return act

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)

class ArcStandardParsingOracle:  # {{{
    def __init__(self):
        self.current_sent = None
        self.childs_of_x = None
        self.connected_childs = set()
        pass

    def next_action_from_config(self, conf):
        return self.next_action(conf.stack, conf.deps, conf.sent, conf.i)

    def next_action(self, stack, deps, sent, i):
        """
        assuming sent has 'parent' information for all tokens

        need to find all childs of a token before combining it to the head
        """
        if self.current_sent != sent:
            self.current_sent = sent
            self.childs_of_x = defaultdict(set)
            self.connected_childs = set([-1])
            for tok in sent:
                self.childs_of_x[tok['parent']].add(tok['id'])

        # if stack has < 2 elements, must shift
        if len(stack) < 2:
            return SHIFT
        # else, if two items on top of stack should connect,
        # choose the correct order
        if stack[-2]['parent'] == stack[-1]['id']:
            # if found_all_childs(stack[-2]):
            if len(self.childs_of_x[stack[-2]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-2]['id'])
                return REDUCE_L
            else:
                pass
        if stack[-1]['parent'] == stack[-2]['id']:
            # if found_all_childs(stack[-1]):
            if len(self.childs_of_x[stack[-1]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-1]['id'])
                return REDUCE_R
            else:
                pass
        # else
        if len(sent) <= i:
            pass
        return SHIFT

class ArcEagerParsingOracle:  # {{{
    def __init__(self, pop_when_can=True):
        self.current_sent = None
        self.childs_of_x = None
        self.connected_childs = set()
        self.POP_WHEN_CAN = pop_when_can
        pass

    def next_actions(self, stack, deps, sent, i):
        return [self.next_action(stack, deps, sent, i)]

    def next_action(self, stack, deps, sent, i):
        """
        if top-of-stack has a connection to next token:
           do reduce_L / reduce_R based on direction
        elsif top-of-stack-minus-1 has a connection to next token:
           do pop
        else do shift

        assuming sent has 'parent' information for all tokens

        assuming several possibilities, the order is left > right > pop > shift.

        see: "An efficient algorithm for projective dependency parsing" / Joakim Nivre, iwpt 2003

        """
        if self.current_sent != sent:
            self.current_sent = sent
            self.childs_of_x = defaultdict(set)
            self.connected_childs = set([-1])
            for tok in sent:
                self.childs_of_x[tok['parent']].add(tok['id'])

        # if stack has < 1 elements, must shift
        if len(stack) < 1:
            return SHIFT

        #ext = sent[i]['extra'].split("|")
        # if len(ext)>1 and ext[1] in ['NONDET']:
        #   return SHIFT

        if stack and sent[i:] and stack[-1]['parent'] == sent[i]['id']:
            self.connected_childs.add(stack[-1]['id'])
            return REDUCE_L

        if stack and sent[i:] and stack[-1]['id'] == sent[i]['parent']:
            if deps.has_parent(sent[i]):
                logging.debug("skipping add: stack [%s], sent [%s]", stack[-1], sent[i])
                pass
            else:
                self.connected_childs.add(sent[i]['id'])
                return REDUCE_R

        if len(stack) > 1:
            # POP when you can
            # if found_all_childs(stack[-1]):
            if len(self.childs_of_x[stack[-1]['id']] -
                   self.connected_childs) < 1:
                if deps.has_parent(stack[-1]):
                    if self.POP_WHEN_CAN:
                        return POP
                    else:  # pop when must..
                        # go up to parents. if a parent has a right-child,
                        # then we need to reduce in order to be able to build it.
                        # else, why reduce?
                        par = deps.parent(stack[-1])
                        while par is not None:
                            par_childs = self.childs_of_x[par['id']]
                            for c in par_childs:
                                if c > stack[-1]['id']:
                                    return POP
                            # if the paren't parent is on the right --
                            # we also need to reduce..
                            if par['parent'] > stack[-1]['id']:
                                return POP
                            par = deps.parent(par)
                        # if we are out of the loop: no need to reduce
        if i < len(sent):
            logging.debug("defaulting: to shift..")
            return SHIFT

        assert(False)

# learners

class OldArcStandardParser:  # {{{
    """
    old code, before refactoring (might be easier to read by starting from here, before decoupling to parser and configuration..)
    """

    def __init__(self, decider):
        self.d = decider
        pass

    def decide(self, stack, deps, sent, i):
        action = self.d.next_action(stack, deps, sent, i)
        return action

    def parse(self, sent):
        stack = []
        sent = [ROOT] + sent
        deps = DependenciesCollection()
        i = 0
        while (sent[i:]) or stack[1:]:
            next_action = self.decide(stack, deps, sent, i)
            if next_action == SHIFT:
                stack.append(sent[i])
                i += 1
            elif next_action == REDUCE_R:
                tokt = stack.pop()  # tok_t
                tokt1 = stack.pop()  # tok_t-1
                deps.add(tokt1, tokt)
                stack.append(tokt1)
            elif next_action == REDUCE_L:
                tokt = stack.pop()  # tok_t
                tokt1 = stack.pop()  # tok_t-1
                deps.add(tokt, tokt1)
                stack.append(tokt)
            else:
                raise "unknown action", next_action
        return deps

class IllegalActionException(Exception):
    pass

class Configuration:  # {{{

    def __init__(self, sent):
        self.stack = []
        self.sent = sent
        self.deps = DependenciesCollection()
        self.i = 0

        self.actions = []

        self._action_scores = []

    def __deepcopy__(self, memo):
        # @@ TODO: how to create the proper Configuration in the derived class?
        c = self.__class__(self.sent)
        c.deps = copy.deepcopy(self.deps, memo)
        c.i = self.i
        c.stack = self.stack[:]
        c.actions = self.actions[:]
        return c

    def actions_map(self):
        """
        returns:
           a dictionary of ACTION -> function_name. to be provided in the derived class
           see ArcStandardConfiguration for an example
        """
        return {}

    def score(self, action): pass  # @TODO

    def is_in_finish_state(self):
        return len(self.stack) == 1 and not self.sent[self.i:]

    def do_action(self, action):
        return self.actions_map()[action]()

    def newAfter(self, action):
        """
        return a new configuration based on current one after aplication of ACTION
        """
        conf = copy.deepcopy(self)
        conf.do_action(action)
        return conf

class ArcStandardConfiguration(Configuration):  # {{{
    def actions_map(self):
        return {
            SHIFT: self.do_shift,
            REDUCE_R: self.do_reduceR,
            REDUCE_L: self.do_reduceL}

    def do_shift(self):
        if not (self.sent[self.i:]):
            raise IllegalActionException()
        self.actions.append(SHIFT)
        self._features = []
        self.stack.append(self.sent[self.i])
        self.i += 1

    def do_reduceR(self):
        if len(self.stack) < 2:
            raise IllegalActionException()
        self.actions.append(REDUCE_R)
        self._features = []
        stack = self.stack
        deps = self.deps

        tokt = stack.pop()  # tok_t
        tokt1 = stack.pop()  # tok_t-1
        deps.add(tokt1, tokt)
        stack.append(tokt1)

    def do_reduceL(self):
        if len(self.stack) < 2:
            raise IllegalActionException()
        self.actions.append(REDUCE_L)
        self._features = []
        stack = self.stack
        deps = self.deps

        tokt = stack.pop()  # tok_t
        tokt1 = stack.pop()  # tok_t-1
        deps.add(tokt, tokt1)
        stack.append(tokt)

    def valid_actions(self):
        res = []
        if self.sent[self.i:]:
            res.append(SHIFT)
        if len(self.stack) >= 2:
            res.append(REDUCE_L)
            res.append(REDUCE_R)
        return res


class Old_ArcEagerConfiguration(Configuration):  # {{{
    """
    Nivre's ArcEager parsing algorithm
    with slightly different action names:

       Nivre's        ThisCode
       ========================
       SHIFT          SHIFT
       ARC_L          REDUCE_L
       ARC_R          REDUCE_R
       REDUCE         POP

    """

    def __init__(self, sent):
        Configuration.__init__(sent)

    def is_in_finish_state(self):
        return not self.sent[self.i:]

    def actions_map(self):
        return {
            SHIFT: self.do_shift,
            REDUCE_R: self.do_reduceR,
            REDUCE_L: self.do_reduceL,
            POP: self.do_pop}

    def do_shift(self):
        logging.debug("do_shift")
        if not (self.sent[self.i:]):
            raise IllegalActionException()
        self.actions.append(SHIFT)
        self._features = []
        self.stack.append(self.sent[self.i])
        self.i += 1

    def do_reduceR(self):
        logging.debug("do_reduceR") 
        if len(self.stack) < 1:
            raise IllegalActionException()
        if len(self.sent) <= self.i:
            raise IllegalActionException()
        self.actions.append(REDUCE_R)
        self._features = []
        stack = self.stack
        deps = self.deps
        sent = self.sent

        # attach the tokens, keeping having both on the stack
        parent = stack[-1]
        child = sent[self.i]
        if deps.has_parent(child):
            raise IllegalActionException()
        deps.add(parent, child)
        self.stack.append(child)
        self.i += 1

    def do_reduceL(self):
        logging.debug("do_reduceL")
        if len(self.stack) < 1:
            raise IllegalActionException()
        if len(self.sent) <= self.i:
            raise IllegalActionException()
        self.actions.append(REDUCE_L)
        self._features = []
        stack = self.stack
        deps = self.deps
        sent = self.sent

        # add top-of-stack as child of sent, pop stack
        child = stack[-1]
        parent = sent[self.i]
        if deps.has_parent(child):
            raise IllegalActionException()
        stack.pop()
        deps.add(parent, child)

    def do_pop(self):
        stack = self.stack

        if len(stack) == 0:
            raise IllegalActionException()
        # also illegal to pop when the item to be popped does not have a
        # parent. (can this happen? yes, right after a shift..)
        if (not self.deps.has_parent(stack[-1])):
            raise IllegalActionException()

        self.actions.append(POP)
        self._features = []

        stack.pop()

    # def valid_actions(self):
    def ABC(self):
        res = [SHIFT, REDUCE_R, REDUCE_L, POP]

        if not (self.sent[self.i:]):
            res.remove(SHIFT)

        if len(stack) == 0:
            res.remove(POP)
        elif not self.deps.has_parent(stack[-1]):
            res.remove(POP)

        if len(self.stack) < 1:
            res.remove(REDUCE_L)
            res.remove(REDUCE_R)
        elif len(self.sent) <= self.i:
            res.remove(REDUCE_L)
            res.remove(REDUCE_R)
        else:
            if self.deps.has_parent(self.stack[-1]):
                res.remove(REDUCE_L)
            if self.deps.has_parent(self.sent[self.i]):
                res.remove(REDUCE_R)

        return res

class ArcEagerConfiguration(Configuration):  # {{{
    """
    Nivre's ArcEager parsing algorithm
    with slightly different action names:

       Nivre's        ThisCode
       ========================
       SHIFT          SHIFT
       ARC_L          REDUCE_L
       ARC_R          REDUCE_R
       REDUCE         POP

    """

    def is_in_finish_state(self):
        return not self.sent[self.i:]

    def actions_map(self):
        return {
            SHIFT: self.do_shift,
            REDUCE_R: self.do_reduceR,
            REDUCE_L: self.do_reduceL,
            POP: self.do_pop}

    def do_shift(self):
        logging.debug("do_shift")
        if not (self.sent[self.i:]):
            raise IllegalActionException()
        self.actions.append(SHIFT)
        self._features = []
        self.stack.append(self.sent[self.i])
        self.i += 1

    def do_reduceR(self):
        logging.debug("do_reduceR")
        if len(self.stack) < 1:
            raise IllegalActionException()
        if len(self.sent) <= self.i:
            raise IllegalActionException()
        self.actions.append(REDUCE_R)
        self._features = []
        stack = self.stack
        deps = self.deps
        sent = self.sent

        # attach the tokens, keeping having both on the stack
        parent = stack[-1]
        child = sent[self.i]
        if deps.has_parent(child):
            raise IllegalActionException()
        deps.add(parent, child)
        self.stack.append(child)
        self.i += 1

    def do_reduceL(self):
        logging.debug("do_reduceL")
        if len(self.stack) < 1:
            raise IllegalActionException()
        if len(self.sent) <= self.i:
            raise IllegalActionException()
        self.actions.append(REDUCE_L)
        self._features = []
        stack = self.stack
        deps = self.deps
        sent = self.sent

        # add top-of-stack as child of sent, pop stack
        child = stack[-1]
        parent = sent[self.i]
        if deps.has_parent(child):
            raise IllegalActionException()
        stack.pop()
        deps.add(parent, child)

    def do_pop(self):
        stack = self.stack

        if len(stack) == 0:
            raise IllegalActionException()
        # also illegal to pop when the item to be popped does not have a
        # parent. (can this happen? yes, right after a shift..)
        if not self.deps.has_parent(stack[-1]):
            if stack[-1]['parent'] != -1:
                raise IllegalActionException()

        self.actions.append(POP)
        self._features = []

        stack.pop()

    def valid_actions(self):
        res = [SHIFT, REDUCE_R, REDUCE_L, POP]

        if not (self.sent[self.i:]):
            res.remove(SHIFT)

        if len(self.stack) == 0:
            res.remove(POP)
        elif not self.deps.has_parent(self.stack[-1]):
            res.remove(POP)

        if len(self.stack) < 1:
            res.remove(REDUCE_L)
            res.remove(REDUCE_R)
        elif len(self.sent) <= self.i:
            res.remove(REDUCE_L)
            res.remove(REDUCE_R)
        else:
            if self.deps.has_parent(self.stack[-1]):
                res.remove(REDUCE_L)
            if self.deps.has_parent(self.sent[self.i]):
                res.remove(REDUCE_R)

        return res

class TransitionBasedParser:
    """
    Refactored ArcStandardParser, with a Configuration object
    """
    Configuration = None
    """
   Configuration class, defines how the parser behaves
   """

    def __init__(self, decider):
        self.d = decider
        pass

    def decide(self, conf):
        actions = self.d.next_actions(
            conf.stack, conf.deps, conf.sent, conf.i, conf)
        return actions

    def parse(self, sent):
        sent = [ROOT] + sent
        conf = self.Configuration(sent)
        while not conf.is_in_finish_state():
            next_actions = self.decide(conf)
            for act in next_actions:
                try:
                    conf.do_action(act)
                    break
                except IllegalActionException:
                    pass
        return conf.deps  # ,conf.chunks

class ArcStandardParser2(TransitionBasedParser):
    Configuration = ArcStandardConfiguration


class ArcEagerParser(TransitionBasedParser):
    Configuration = ArcEagerConfiguration


class ErrorInspectionParser(ArcStandardParser2):  # {{{
    def __init__(self, decider, oracle, confPrinter, out=sys.stdout):
        ArcStandardParser2.__init__(self, decider)
        self.oracle = oracle
        self.confPrinter = confPrinter
        self.out = out

        self.raise_on_error = False
        self.use_oracle_answer = True

    def decide(self, conf):
        action = ArcStandardParser2.decide(self, conf)
        real = self.oracle.next_action(
            conf.stack, conf.deps, conf.sent, conf.i)
        if action != real:
            self.out.write(
                "%s -> %s %s\n" %
                (real, action, self.confPrinter.format(conf)))
            if self.raise_on_error:
                raise MLTrainerWrongActionException()

        if self.use_oracle_answer:
            return real
        else:
            return action
