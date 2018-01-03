# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility to build vocabulary."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import time
import os

def get_time():
    """Get local time"""
    return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

def basic_tokenizer(sentence, delimiter=None):
    """A very simple tokenizer with space to tokenize a sentence.
    Args:
        sentence: a token sequence, which is pre-tokened.
        dilimiter: delimiter string. If None, using default delimiter.
    Return:
        A word list.
    """
    if delimiter==None:
        words = [w for w in sentence.strip().split()]
    else:
        words = [w for w in sentence.strip().split(delimiter)]
    return words

def main():
    parser = argparse.ArgumentParser(description="Build vocabulary")
    parser.add_argument(
        "--data", default=None,
        required=True,
        help="Source text file")
    parser.add_argument(
        "--save_vocab", default=None,
        required=True,
        help="Output vocabulary file")
    parser.add_argument(
        "--min_frequency", default=1,
        type=int,
        help="Min word frequency, default 1")
    parser.add_argument(
        "--size", default=0,
        type=int,
        help="Max vocabulary size, if set 0, no limited, default 0")
    parser.add_argument(
        "--without_special_token", default=False,
        help="If set true, the vocabulary will not contain special such as '<s>','</s>' etc."
    )
    args = parser.parse_args()
    # build vocabulary
    if os.path.exists(args.data):
        print(get_time()+"  Build vocabulary...")
        with codecs.open(args.data, "r", "utf-8") as data_f:
            with codecs.open(args.save_vocab, "w", "utf-8") as vocab_f:
                vocab_dict = {}
                for line in data_f:
                    # space split
                    words = basic_tokenizer(line)
                    for w in words:
                        vocab_dict[w] = (vocab_dict[w]+1 if w in vocab_dict else 1)
                # filter low frequency words
                if args.min_frequency>1:
                    for w, v in vocab_dict.items():
                        if v<args.min_frequency:
                            vocab_dict.pop(w)
                # sort vocab by value
                vocab = sorted(vocab_dict, key=vocab_dict.get, reverse=True)
                # add special token to vocabulary
                if not args.without_special_token:
                    vocab = ["<unk>", "<s>", "</s>"] + vocab
                # cut vocabulary to max vocabulary size
                if args.size>0:
                    vocab = vocab[0:args.size]
                # write vocab to vocabulary file
                for word in vocab:
                    vocab_f.write(word+"\n")
                print(get_time()+"  %d words have been written into vocabulary file..." % len(vocab))
    else:
        raise ValueError("%s doesn't exist!" % args.data)

if __name__=="__main__":
    main()
