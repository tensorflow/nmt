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

# file: m2scorer.py
# 
# score a system's output against a gold reference 
#
# Usage: m2scorer.py [OPTIONS] proposed_sentences source_gold
# where
#  proposed_sentences   -   system output, sentence per line
#  source_gold          -   source sentences with gold token edits
# OPTIONS
#   -v    --verbose             -  print verbose output
#   --very_verbose              -  print lots of verbose output
#   --max_unchanged_words N     -  Maximum unchanged words when extracting edits. Default 2."
#   --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and caseing. Default no."
#

import sys
import levenshtein
from getopt import getopt
from util import paragraphs
from util import smart_open



def load_annotation(gold_file):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file, 'r')
    puffer = fgold.read()
    fgold.close()
    puffer = puffer.decode('utf8')
    for item in paragraphs(puffer.splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                continue
            assert line.startswith('A ')
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == 'noop':
                start_offset = -1
                end_offset = -1
            corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in annotations.keys():
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections))
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.iteritems():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)


def print_usage():
    print >> sys.stderr, "Usage: m2scorer.py [OPTIONS] proposed_sentences gold_source"
    print >> sys.stderr, "where"
    print >> sys.stderr, "  proposed_sentences   -   system output, sentence per line"
    print >> sys.stderr, "  source_gold          -   source sentences with gold token edits"
    print >> sys.stderr, "OPTIONS"
    print >> sys.stderr, "  -v    --verbose                   -  print verbose output"
    print >> sys.stderr, "        --very_verbose              -  print lots of verbose output"
    print >> sys.stderr, "        --max_unchanged_words N     -  Maximum unchanged words when extraction edit. Default 2."
    print >> sys.stderr, "        --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and caseing. Default no."



max_unchanged_words=2
ignore_whitespace_casing= False
verbose = False
very_verbose = False
opts, args = getopt(sys.argv[1:], "v", ["max_unchanged_words=", "verbose", "ignore_whitespace_casing", "very_verbose"])
for o, v in opts:
    if o in ('-v', '--verbose'):
        verbose = True
    elif o == '--very_verbose':
        very_verbose = True
    elif o == '--max_unchanged_words':
        max_unchanged_words = int(v)
    elif o == '--ignore_whitespace_casing':
        ignore_whitespace_casing = True
    else:
        print >> sys.stderr, "Unknown option :", o
        print_usage()
        sys.exit(-1)


