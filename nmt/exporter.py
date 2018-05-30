# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Export pre-trained model."""
import os
import time

import tensorflow as tf

from . import attention_model as attention_model
from . import gnmt_model as gnmt_model
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils


class Exporter(object):
  """Export pre-trained model and serve it by tensorflow/serving.
  """

  def __init__(self, hparams, flags):
    """Construct exporter.

    By default, the hparams can be loaded from the `hparams` file
    which saved in out_dir if you enable save_hparams. So if you want to
    export the model, you just add arguments that needed for exporting.
    Arguments are specified in ``nmt.py`` module.
    Go and check that in ``add_export_arugments()`` function.

    Args:
     hparams: Hyperparameter configurations.
     flags: extra flags used for exporting model.
    """
    self.hparams = hparams
    self._model_dir = self.hparams.out_dir
    v = flags.version_number
    self._version_number = v if v else int(round(time.time() * 1000))

    export_path = flags.export_path if flags.export_path else self.hparams.out_dir
    self._export_dir = os.path.join(export_path, str(self._version_number))

    # Decide a checkpoint path
    ckpt_path = self.hparams.out_dir
    if flags.ckpt_path:
      ckpt_path = flags.ckpt_path
    else:
      for metric in self.hparams.metrics:
        p = getattr(self.hparams, "best_" + metric + "_dir")
        if os.path.exists(p):
          ckpt_path = p
          break
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    self._ckpt_path = ckpt.model_checkpoint_path

    # A file contains sequences, used for initializing iterators.
    # A good idea is to use test or dev files as infer_file
    test_file = self.hparams.test_prefix + "." + self.hparams.src
    self._infer_file = flags.infer_file if flags.infer_file else test_file

    self._print_params()

  def _print_params(self):
    misc_utils.print_hparams(self.hparams)
    print("Model directory  : %s" % self._model_dir)
    print("Checkpoint path  : %s" % self._ckpt_path)
    print("Export path      : %s" % self._export_dir)
    print("Inference file   : %s" % self._infer_file)
    print("Version number   : %d" % self._version_number)

  def _load_infer_data(self):
    from .inference import load_data
    infer_data = load_data(self._infer_file)
    return infer_data

  def _create_infer_model(self):
    if not self.hparams.attention:
      model_creator = nmt_model.Model
    elif self.hparams.attention_architecture == "standard":
      model_creator = attention_model.AttentionModel
    elif self.hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
      model_creator = gnmt_model.GNMTModel
    else:
      raise ValueError("Unknown model architecture")
    model = model_helper.create_infer_model(model_creator=model_creator,
                                            hparams=self.hparams, scope=None)
    return model

  def export(self):
    infer_model = self._create_infer_model()

    with tf.Session(graph=infer_model.graph,
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      feature_config = {
        'input': tf.FixedLenSequenceFeature(dtype=tf.string,
                                            shape=[], allow_missing=True),
      }
      serialized_example = tf.placeholder(dtype=tf.string, name="serialized_example")
      tf_example = tf.parse_example(serialized_example, feature_config)
      inference_input = tf.identity(tf_example['input'], name="infer_input")

      saver = infer_model.model.saver
      saver.restore(sess, self._ckpt_path)

      # initialize tables
      sess.run(tf.tables_initializer())
      sess.run(
        infer_model.iterator.initializer,
        feed_dict={
          infer_model.src_placeholder: self._load_infer_data(),
          infer_model.batch_size_placeholder: self.hparams.infer_batch_size
        })

      # get outputs of model
      inference_outputs, _ = infer_model.model.decode(sess=sess)
      # get the first of the outputs as the result of inference
      inference_output = inference_outputs[0]

      # create signature def
      # key `seq_input` in `inputs` dict could be changed as your will,
      # but the client should consistent with this
      # when you make an inference request.
      # key `seq_output` in outputs dict is the same as above
      inference_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={
          'seq_input': inference_input
        },
        outputs={
          'seq_output': tf.convert_to_tensor(inference_output)
        }
      )
      legacy_ini_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

      builder = tf.saved_model.builder.SavedModelBuilder(self._export_dir)
      # key `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`
      #  (is `serving_default` actually) in signature_def_map could be changed
      # as your will. But the client should consistent with this when you make an inference request.
      builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: inference_signature,
        },
        legacy_init_op=legacy_ini_op,
        clear_devices=True,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
      builder.save(as_text=True)
      print("Done!")
