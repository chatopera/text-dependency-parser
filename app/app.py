#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: /Users/hain/ai/text-dependency-parser/app/app.py
# Author: Hai Liang Wang
# Date: 2018-02-22:21:57:07
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2018-02-22:21:57:07"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

import unittest

# run testcase: python /Users/hain/ai/text-dependency-parser/app/app.py Test.testExample
class Test(unittest.TestCase):
    '''
    
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testExample(self):
        print("foo")

def test():
    unittest.main()

if __name__ == '__main__':
    test()
