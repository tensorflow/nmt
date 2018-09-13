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

"""standard hparams utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_standard_hparams():
  return tf.contrib.training.HParams(
      # Data
      src="",
      tgt="",
      train_prefix="",
      dev_prefix="",
      test_prefix="",
      vocab_prefix="",
      embed_prefix="",
      out_dir="",

      # Networks
      num_units=512,
      num_encoder_layers=2,
      num_decoder_layers=2,
      dropout=0.2,
      unit_type="lstm",
      encoder_type="bi",
      residual=False,
      time_major=True,
      num_embeddings_partitions=0,
      num_enc_emb_partitions=0,
      num_dec_emb_partitions=0,

      # Attention mechanisms
      attention="scaled_luong",
      attention_architecture="standard",
      output_attention=True,
      pass_hidden_state=True,

      # Train
      optimizer="sgd",
      batch_size=128,
      init_op="uniform",
      init_weight=0.1,
      max_gradient_norm=5.0,
      learning_rate=1.0,
      warmup_steps=0,
      warmup_scheme="t2t",
      decay_scheme="luong234",
      colocate_gradients_with_ops=True,
      num_train_steps=12000,
      num_sampled_softmax=0,

      # Data constraints
      num_buckets=5,
      max_train=0,
      src_max_len=50,
      tgt_max_len=50,
      src_max_len_infer=0,
      tgt_max_len_infer=0,

      # Data format
      sos="<s>",
      eos="</s>",
      subword_option="",
      use_char_encode=False,
      check_special_token=True,

      # Misc
      forget_bias=1.0,
      num_gpus=1,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=100,
      steps_per_external_eval=0,
      share_vocab=False,
      metrics=["bleu"],
      log_device_placement=False,
      random_seed=None,
      # only enable beam search during inference when beam_width > 0.
      beam_width=0,
      length_penalty_weight=0.0,
      coverage_penalty_weight=0.0,
      override_loaded_hparams=True,
      num_keep_ckpts=5,
      avg_ckpts=False,

      # For inference
      inference_indices=None,
      infer_batch_size=32,
      sampling_temperature=0.0,
      num_translations_per_input=1,
      infer_mode="greedy",

      # Language model
      language_model=False,
  )
