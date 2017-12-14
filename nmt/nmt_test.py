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
"""Tests for nmt.py, train.py and inference.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from . import inference
from . import nmt
from . import train


def _update_flags(flags, test_name):
  """Update flags for basic training."""
  flags.num_train_steps = 100
  flags.steps_per_stats = 5
  flags.src = "en"
  flags.tgt = "vi"
  flags.train_prefix = ("nmt/testdata/"
                        "iwslt15.tst2013.100")
  flags.vocab_prefix = ("nmt/testdata/"
                        "iwslt15.vocab.100")
  flags.dev_prefix = ("nmt/testdata/"
                      "iwslt15.tst2013.100")
  flags.test_prefix = ("nmt/testdata/"
                       "iwslt15.tst2013.100")
  flags.out_dir = os.path.join(tf.test.get_temp_dir(), test_name)


class NMTTest(tf.test.TestCase):

  def testTrain(self):
    """Test the training loop is functional with basic hparams."""
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    _update_flags(FLAGS, "nmt_train_test")

    default_hparams = nmt.create_hparams(FLAGS)

    train_fn = train.train
    nmt.run_main(FLAGS, default_hparams, train_fn, None)


  def testTrainWithAvgCkpts(self):
    """Test the training loop is functional with basic hparams."""
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    _update_flags(FLAGS, "nmt_train_test_avg_ckpts")
    FLAGS.avg_ckpts = True

    default_hparams = nmt.create_hparams(FLAGS)

    train_fn = train.train
    nmt.run_main(FLAGS, default_hparams, train_fn, None)


  def testInference(self):
    """Test inference is function with basic hparams."""
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    _update_flags(FLAGS, "nmt_train_infer")

    # Train one step so we have a checkpoint.
    FLAGS.num_train_steps = 1
    default_hparams = nmt.create_hparams(FLAGS)
    train_fn = train.train
    nmt.run_main(FLAGS, default_hparams, train_fn, None)

    # Update FLAGS for inference.
    FLAGS.inference_input_file = ("nmt/testdata/"
                                  "iwslt15.tst2013.100.en")
    FLAGS.inference_output_file = os.path.join(FLAGS.out_dir, "output")
    FLAGS.inference_ref_file = ("nmt/testdata/"
                                "iwslt15.tst2013.100.vi")

    default_hparams = nmt.create_hparams(FLAGS)

    inference_fn = inference.inference
    nmt.run_main(FLAGS, default_hparams, None, inference_fn)


if __name__ == "__main__":
  tf.test.main()
