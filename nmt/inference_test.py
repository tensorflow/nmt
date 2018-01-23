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

"""Tests for model inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from . import inference
from . import model_helper
from .utils import common_test_utils

float32 = np.float32
int32 = np.int32
array = np.array


class InferenceTest(tf.test.TestCase):

  def _createTestInferCheckpoint(self, hparams, name):
    # Prepare
    hparams.vocab_prefix = (
        "nmt/testdata/test_infer_vocab")
    hparams.src_vocab_file = hparams.vocab_prefix + "." + hparams.src
    hparams.tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
    out_dir = os.path.join(tf.test.get_temp_dir(), name)
    os.makedirs(out_dir)
    hparams.out_dir = out_dir

    # Create check point
    model_creator = inference.get_model_creator(hparams)
    infer_model = model_helper.create_infer_model(model_creator, hparams)
    with self.test_session(graph=infer_model.graph) as sess:
      loaded_model, global_step = model_helper.create_or_load_model(
          infer_model.model, out_dir, sess, "infer_name")
      ckpt_path = loaded_model.saver.save(
          sess, os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)
    return ckpt_path

  def testBasicModel(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="",
        attention_architecture="",
        use_residual=False,)
    ckpt_path = self._createTestInferCheckpoint(hparams, "basic_infer")
    infer_file = "nmt/testdata/test_infer_file"
    output_infer = os.path.join(hparams.out_dir, "output_infer")
    inference.inference(ckpt_path, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(5, len(list(f)))

  def testBasicModelWithMultipleTranslations(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="",
        attention_architecture="",
        use_residual=False,
        num_translations_per_input=2,
        beam_width=2,
    )
    hparams.infer_mode = "beam_search"

    ckpt_path = self._createTestInferCheckpoint(hparams, "multi_basic_infer")
    infer_file = "nmt/testdata/test_infer_file"
    output_infer = os.path.join(hparams.out_dir, "output_infer")
    inference.inference(ckpt_path, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(10, len(list(f)))

  def testAttentionModel(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="scaled_luong",
        attention_architecture="standard",
        use_residual=False,)
    ckpt_path = self._createTestInferCheckpoint(hparams, "attention_infer")
    infer_file = "nmt/testdata/test_infer_file"
    output_infer = os.path.join(hparams.out_dir, "output_infer")
    inference.inference(ckpt_path, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(5, len(list(f)))

  def testMultiWorkers(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=2,
        attention="scaled_luong",
        attention_architecture="standard",
        use_residual=False,)

    num_workers = 3

    # There are 5 examples, make batch_size=3 makes job0 has 3 examples, job1
    # has 2 examples, and job2 has 0 example. This helps testing some edge
    # cases.
    hparams.batch_size = 3

    ckpt_path = self._createTestInferCheckpoint(hparams, "multi_worker_infer")
    infer_file = "nmt/testdata/test_infer_file"
    output_infer = os.path.join(hparams.out_dir, "output_infer")
    inference.inference(
        ckpt_path, infer_file, output_infer, hparams, num_workers, jobid=1)

    inference.inference(
        ckpt_path, infer_file, output_infer, hparams, num_workers, jobid=2)

    # Note: Need to start job 0 at the end; otherwise, it will block the testing
    # thread.
    inference.inference(
        ckpt_path, infer_file, output_infer, hparams, num_workers, jobid=0)

    with open(output_infer) as f:
      self.assertEqual(5, len(list(f)))

  def testBasicModelWithInferIndices(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="",
        attention_architecture="",
        use_residual=False,
        inference_indices=[0])
    ckpt_path = self._createTestInferCheckpoint(hparams,
                                                "basic_infer_with_indices")
    infer_file = "nmt/testdata/test_infer_file"
    output_infer = os.path.join(hparams.out_dir, "output_infer")
    inference.inference(ckpt_path, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(1, len(list(f)))

  def testAttentionModelWithInferIndices(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="scaled_luong",
        attention_architecture="standard",
        use_residual=False,
        inference_indices=[1, 2])
    # TODO(rzhao): Make infer indices support batch_size > 1.
    hparams.infer_batch_size = 1
    ckpt_path = self._createTestInferCheckpoint(hparams,
                                                "attention_infer_with_indices")
    infer_file = "nmt/testdata/test_infer_file"
    output_infer = os.path.join(hparams.out_dir, "output_infer")
    inference.inference(ckpt_path, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(2, len(list(f)))
    self.assertTrue(os.path.exists(output_infer+str(1)+".png"))
    self.assertTrue(os.path.exists(output_infer+str(2)+".png"))


if __name__ == "__main__":
  tf.test.main()
