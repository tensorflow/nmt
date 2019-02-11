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
    self._model = flags.model if flags.model else "generative"

    # Decide a checkpoint path
    ckpt_path = self._get_ckpt_path(flags.ckpt_path)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    self._ckpt_path = ckpt.model_checkpoint_path

    # A file contains sequences, used for initializing iterators.
    # A good idea is to use test or dev files as infer_file
    test_file = self.hparams.test_prefix + "." + self.hparams.src
    self._infer_file = flags.infer_file if flags.infer_file else test_file

    self._print_params()

  def _print_params(self):
    misc_utils.print_hparams(self.hparams)
    print("Model to export  : %s" % self._model)
    print("Model directory  : %s" % self._model_dir)
    print("Checkpoint path  : %s" % self._ckpt_path)
    print("Export path      : %s" % self._export_dir)
    print("Inference file   : %s" % self._infer_file)
    print("Version number   : %d" % self._version_number)

  def _get_ckpt_path(self, flags_ckpt_path):
    ckpt_path = None
    if flags_ckpt_path:
      ckpt_path = flags_ckpt_path
    else:
      for metric in self.hparams.metrics:
        p = getattr(self.hparams, "best_" + metric + "_dir")
        if os.path.exists(p):
          if self._has_ckpt_file(p):
            ckpt_path = p
          break
    if not ckpt_path:
      ckpt_path = self.hparams.out_dir
    return ckpt_path

  @staticmethod
  def _has_ckpt_file(p):
    for f in os.listdir(p):
      if str(f).endswith(".meta"):
        return True
    return False

  def _load_infer_data(self):
    from .inference import load_data
    infer_data = load_data(self._infer_file)
    return infer_data

  def _create_serve_model(self):
    if not self.hparams.attention:
      model_creator = nmt_model.Model
    elif self.hparams.attention_architecture == "standard":
      model_creator = attention_model.AttentionModel
    elif self.hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
      model_creator = gnmt_model.GNMTModel
    else:
      raise ValueError("Unknown model architecture")
    if self._model == "generative":
      model = model_helper.create_serve_model(model_creator=model_creator,
                                              hparams=self.hparams, scope=None)
    elif self._model == "discriminative":
      model = model_helper.create_score_model(model_creator=model_creator,
                                              hparams=self.hparams, scope=None)
    else:
      raise NotImplementedError
    return model

  def _export_generative_model(self):
    infer_model = self._create_serve_model()

    with tf.Session(graph=infer_model.graph,
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      inference_input = infer_model.graph.get_tensor_by_name('src_placeholder:0')	  

      saver = infer_model.model.saver
      saver.restore(sess, self._ckpt_path)

      # initialize tables
      sess.run(tf.tables_initializer())
      inference_outputs = infer_model.model.sample_words
      inference_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={
          'seq_input': inference_input
        },
        outputs={
          'seq_output': tf.squeeze(tf.convert_to_tensor(inference_outputs)),
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
        clear_devices=False,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
      builder.save(as_text=False)

  def _export_discriminative_model(self):
    score_model = self._create_serve_model()

    with tf.Session(graph=score_model.graph,
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      inference_input_src = score_model.graph.get_tensor_by_name('src_placeholder:0')
      inference_input_tgt = score_model.graph.get_tensor_by_name('tgt_placeholder:0')

      saver = score_model.model.saver
      saver.restore(sess, self._ckpt_path)

      # initialize tables
      sess.run(tf.tables_initializer())
      log_likelihoods = score_model.model.log_likelihoods

      inference_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={
          'seq_input_src': inference_input_src,
          'seq_input_tgt': inference_input_tgt,
        },
        outputs={
          'seq_output': tf.squeeze(tf.convert_to_tensor(log_likelihoods)),
          'print': score_model.model.print_log_likelihoods_op
        }
      )
      legacy_ini_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
      # main_init_op = tf.group(score_model.iterator.initializer, name='main_init_op')
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
        clear_devices=False,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
      builder.save(as_text=False)

  def export(self):
    if self._model == "generative":
      self._export_generative_model()
    elif self._model == "discriminative":
      self._export_discriminative_model()
    else:
      raise NotImplementedError
    print("exported {} at {}".format(self._model, self._export_dir))
