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

"""Common utility functions for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from ..utils import iterator_utils
from ..utils import standard_hparams_utils


def create_test_hparams(unit_type="lstm",
                        encoder_type="uni",
                        num_layers=4,
                        attention="",
                        attention_architecture=None,
                        use_residual=False,
                        inference_indices=None,
                        num_translations_per_input=1,
                        beam_width=0,
                        init_op="uniform"):
  """Create training and inference test hparams."""
  num_residual_layers = 0
  if use_residual:
    # TODO(rzhao): Put num_residual_layers computation logic into
    # `model_utils.py`, so we can also test it here.
    num_residual_layers = 2

  standard_hparams = standard_hparams_utils.create_standard_hparams()

  # Networks
  standard_hparams.num_units = 5
  standard_hparams.num_encoder_layers = num_layers
  standard_hparams.num_decoder_layers = num_layers
  standard_hparams.dropout = 0.5
  standard_hparams.unit_type = unit_type
  standard_hparams.encoder_type = encoder_type
  standard_hparams.residual = use_residual
  standard_hparams.num_residual_layers = num_residual_layers

  # Attention mechanisms
  standard_hparams.attention = attention
  standard_hparams.attention_architecture = attention_architecture

  # Train
  standard_hparams.init_op = init_op
  standard_hparams.num_train_steps = 1
  standard_hparams.decay_scheme = ""

  # Infer
  standard_hparams.tgt_max_len_infer = 100
  standard_hparams.beam_width = beam_width
  standard_hparams.num_translations_per_input = num_translations_per_input

  # Misc
  standard_hparams.forget_bias = 0.0
  standard_hparams.random_seed = 3
  standard_hparams.language_model = False

  # Vocab
  standard_hparams.src_vocab_size = 5
  standard_hparams.tgt_vocab_size = 5
  standard_hparams.eos = "</s>"
  standard_hparams.sos = "<s>"
  standard_hparams.src_vocab_file = ""
  standard_hparams.tgt_vocab_file = ""
  standard_hparams.src_embed_file = ""
  standard_hparams.tgt_embed_file = ""

  # For inference.py test
  standard_hparams.subword_option = "bpe"
  standard_hparams.src = "src"
  standard_hparams.tgt = "tgt"
  standard_hparams.src_max_len = 400
  standard_hparams.tgt_eos_id = 0
  standard_hparams.inference_indices = inference_indices
  return standard_hparams


def create_test_iterator(hparams, mode):
  """Create test iterator."""
  src_vocab_table = lookup_ops.index_table_from_tensor(
      tf.constant([hparams.eos, "a", "b", "c", "d"]))
  tgt_vocab_mapping = tf.constant([hparams.sos, hparams.eos, "a", "b", "c"])
  tgt_vocab_table = lookup_ops.index_table_from_tensor(tgt_vocab_mapping)
  if mode == tf.contrib.learn.ModeKeys.INFER:
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_tensor(
        tgt_vocab_mapping)

  src_dataset = tf.data.Dataset.from_tensor_slices(
      tf.constant(["a a b b c", "a b b"]))

  if mode != tf.contrib.learn.ModeKeys.INFER:
    tgt_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["a b c b c", "a b c b"]))
    return (
        iterator_utils.get_iterator(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets),
        src_vocab_table,
        tgt_vocab_table)
  else:
    return (
        iterator_utils.get_infer_iterator(
            src_dataset=src_dataset,
            src_vocab_table=src_vocab_table,
            eos=hparams.eos,
            batch_size=hparams.batch_size),
        src_vocab_table,
        tgt_vocab_table,
        reverse_tgt_vocab_table)
