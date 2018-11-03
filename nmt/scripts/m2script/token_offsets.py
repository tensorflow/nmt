#!/usr/bin/python

# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: token_offsets.py
# convert character to token offsets, tokenize sentence 
#
# usage: %prog  < input > output
#


import sys
import re
import os
from util import *
from Tokenizer import PTBTokenizer


assert len(sys.argv) == 1


# main
# loop over sentences cum annotation
tokenizer = PTBTokenizer()
sentence = ''
for line in sys.stdin:
    line = line.decode("utf8").strip()
    if line.startswith("S "):
        sentence = line[2:]
        sentence_tok = "S " + ' '.join(tokenizer.tokenize(sentence))
        print sentence_tok.encode("utf8")
    elif line.startswith("A "):
        fields = line[2:].split('|||')
        start_end = fields[0]
        char_start, char_end = [int(a) for a in start_end.split()]
        # calculate token offsets
        prefix = sentence[:char_start]
        tok_start = len(tokenizer.tokenize(prefix))
        postfix = sentence[:char_end]
        tok_end = len(tokenizer.tokenize(postfix))
        start_end = str(tok_start) + " " + str(tok_end)
        fields[0] = start_end
        # tokenize corrections, remove trailing whitespace
        corrections = [(' '.join(tokenizer.tokenize(c))).strip() for c in fields[2].split('||')]
        fields[2] = '||'.join(corrections)
        annotation =  "A " + '|||'.join(fields)
        print annotation.encode("utf8")
    else:
        print line.encode("utf8")

