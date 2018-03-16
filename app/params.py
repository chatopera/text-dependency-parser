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
from optparse import OptionParser

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
    "-l",
    "--lazypop",
    dest="POP_WHEN_CAN",
    action="store_false",
    default=True)
parser.add_option(
    "-u",
    "--unlex",
    dest="UNLEX",
    action="store_true",
    default=False)
parser.add_option(
    "--features",
    dest="feature_extarctor",
    default="eager.zhang")
