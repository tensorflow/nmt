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

"""Utility functions for building models."""
from __future__ import print_function

import collections
import os
import time
import numpy as np
import six
import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

__all__ = [
    "get_initializer", "get_device_str", "create_train_model",
    "create_eval_model", "create_infer_model",
    "create_emb_for_encoder_and_decoder", "create_rnn_cell", "gradient_clip",
    "create_or_load_model", "load_model", "avg_checkpoints",
    "compute_perplexity"
]

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000


def get_initializer(init_op, seed=None, init_weight=None):
  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(
        -init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(
        seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(
        seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id, num_gpus):
  """Return a device string for multi-GPU setup."""
  if num_gpus == 0:
    return "/cpu:0"
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


class ExtraArgs(collections.namedtuple(
    "ExtraArgs", ("single_cell_fn", "model_device_fn",
                  "attention_mechanism_fn", "encoder_emb_lookup_fn"))):
  pass


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
  pass


def create_train_model(
    model_creator, hparams, scope=None, num_workers=1, jobid=0,
    extra_args=None):
  """Create train graph, model, and iterator."""
  src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
  tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "train"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)

    src_dataset = tf.data.TextLineDataset(tf.gfile.Glob(src_file))
    tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(tgt_file))
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        skip_count=skip_count_placeholder,
        num_shards=num_workers,
        shard_index=jobid,
        use_char_encode=hparams.use_char_encode)

    # Note: One can set model_device_fn to
    # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
    model_device_fn = None
    if extra_args: model_device_fn = extra_args.model_device_fn
    with tf.device(model_device_fn):
      model = model_creator(
          hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          scope=scope,
          extra_args=extra_args)

  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator,
      skip_count_placeholder=skip_count_placeholder)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src_file_placeholder",
                            "tgt_file_placeholder", "iterator"))):
  pass


def create_eval_model(model_creator, hparams, scope=None, extra_args=None):
  """Create train graph, model, src/tgt file holders, and iterator."""
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "eval"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)

    src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    src_dataset = tf.data.TextLineDataset(src_file_placeholder)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len_infer,
        tgt_max_len=hparams.tgt_max_len_infer,
        use_char_encode=hparams.use_char_encode)
    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        reverse_target_vocab_table=reverse_tgt_vocab_table,
        scope=scope,
        extra_args=extra_args)
  return EvalModel(
      graph=graph,
      model=model,
      src_file_placeholder=src_file_placeholder,
      tgt_file_placeholder=tgt_file_placeholder,
      iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder",
                            "batch_size_placeholder", "iterator"))):
  pass


def create_infer_model(model_creator, hparams, scope=None, extra_args=None):
  """Create inference model."""
  graph = tf.Graph()
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  with graph.as_default(), tf.container(scope or "infer"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)

    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

    src_dataset = tf.data.Dataset.from_tensor_slices(
        src_placeholder)
    iterator = iterator_utils.get_infer_iterator(
        src_dataset,
        src_vocab_table,
        batch_size=batch_size_placeholder,
        eos=hparams.eos,
        src_max_len=hparams.src_max_len_infer,
        use_char_encode=hparams.use_char_encode)
    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.INFER,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        reverse_target_vocab_table=reverse_tgt_vocab_table,
        scope=scope,
        extra_args=extra_args)
  return InferModel(
      graph=graph,
      model=model,
      src_placeholder=src_placeholder,
      batch_size_placeholder=batch_size_placeholder,
      iterator=iterator)


def _get_embed_device(vocab_size):
  """Decide on which device to place an embed matrix given its vocab size."""
  if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
    return "/cpu:0"
  else:
    return "/gpu:0"


def _create_pretrained_emb_from_txt(
    vocab_file, embed_file, num_trainable_tokens=3, dtype=tf.float32,
    scope=None):
  """Load pretrain embeding from embed_file, and return an embedding matrix.

  Args:
    embed_file: Path to a Glove formated embedding txt file.
    num_trainable_tokens: Make the first n tokens in the vocab file as trainable
      variables. Default is 3, which is "<unk>", "<s>" and "</s>".
  """
  vocab, _ = vocab_utils.load_vocab(vocab_file)
  trainable_tokens = vocab[:num_trainable_tokens]

  utils.print_out("# Using pretrained embedding: %s." % embed_file)
  utils.print_out("  with trainable tokens: ")

  emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file)
  for token in trainable_tokens:
    utils.print_out("    %s" % token)
    if token not in emb_dict:
      emb_dict[token] = [0.0] * emb_size

  emb_mat = np.array(
      [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
  emb_mat = tf.constant(emb_mat)
  emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
  with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
    with tf.device(_get_embed_device(num_trainable_tokens)):
      emb_mat_var = tf.get_variable(
          "emb_mat_var", [num_trainable_tokens, emb_size])
  return tf.concat([emb_mat_var, emb_mat_const], 0)


def _create_or_load_embed(embed_name, vocab_file, embed_file,
                          vocab_size, embed_size, dtype):
  """Create a new or load an existing embedding matrix."""
  if vocab_file and embed_file:
    embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
  else:
    with tf.device(_get_embed_device(vocab_size)):
      embedding = tf.get_variable(
          embed_name, [vocab_size, embed_size], dtype)
  return embedding


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_enc_partitions=0,
                                       num_dec_partitions=0,
                                       src_vocab_file=None,
                                       tgt_vocab_file=None,
                                       src_embed_file=None,
                                       tgt_embed_file=None,
                                       use_char_encode=False,
                                       scope=None):
  """Create embedding matrix for both encoder and decoder.

  Args:
    share_vocab: A boolean. Whether to share embedding matrix for both
      encoder and decoder.
    src_vocab_size: An integer. The source vocab size.
    tgt_vocab_size: An integer. The target vocab size.
    src_embed_size: An integer. The embedding dimension for the encoder's
      embedding.
    tgt_embed_size: An integer. The embedding dimension for the decoder's
      embedding.
    dtype: dtype of the embedding matrix. Default to float32.
    num_enc_partitions: number of partitions used for the encoder's embedding
      vars.
    num_dec_partitions: number of partitions used for the decoder's embedding
      vars.
    scope: VariableScope for the created subgraph. Default to "embedding".

  Returns:
    embedding_encoder: Encoder's embedding matrix.
    embedding_decoder: Decoder's embedding matrix.

  Raises:
    ValueError: if use share_vocab but source and target have different vocab
      size.
  """
  if num_enc_partitions <= 1:
    enc_partitioner = None
  else:
    # Note: num_partitions > 1 is required for distributed training due to
    # embedding_lookup tries to colocate single partition-ed embedding variable
    # with lookup ops. This may cause embedding variables being placed on worker
    # jobs.
    enc_partitioner = tf.fixed_size_partitioner(num_enc_partitions)

  if num_dec_partitions <= 1:
    dec_partitioner = None
  else:
    # Note: num_partitions > 1 is required for distributed training due to
    # embedding_lookup tries to colocate single partition-ed embedding variable
    # with lookup ops. This may cause embedding variables being placed on worker
    # jobs.
    dec_partitioner = tf.fixed_size_partitioner(num_dec_partitions)

  if src_embed_file and enc_partitioner:
    raise ValueError(
        "Can't set num_enc_partitions > 1 when using pretrained encoder "
        "embedding")

  if tgt_embed_file and dec_partitioner:
    raise ValueError(
        "Can't set num_dec_partitions > 1 when using pretrained decdoer "
        "embedding")

  with tf.variable_scope(
      scope or "embeddings", dtype=dtype, partitioner=enc_partitioner) as scope:
    # Share embedding
    if share_vocab:
      if src_vocab_size != tgt_vocab_size:
        raise ValueError("Share embedding but different src/tgt vocab sizes"
                         " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
      assert src_embed_size == tgt_embed_size
      utils.print_out("# Use the same embedding for source and target")
      vocab_file = src_vocab_file or tgt_vocab_file
      embed_file = src_embed_file or tgt_embed_file

      embedding_encoder = _create_or_load_embed(
          "embedding_share", vocab_file, embed_file,
          src_vocab_size, src_embed_size, dtype)
      embedding_decoder = embedding_encoder
    else:
      if not use_char_encode:
        with tf.variable_scope("encoder", partitioner=enc_partitioner):
          embedding_encoder = _create_or_load_embed(
              "embedding_encoder", src_vocab_file, src_embed_file,
              src_vocab_size, src_embed_size, dtype)
      else:
        embedding_encoder = None

      with tf.variable_scope("decoder", partitioner=dec_partitioner):
        embedding_decoder = _create_or_load_embed(
            "embedding_decoder", tgt_vocab_file, tgt_embed_file,
            tgt_vocab_size, tgt_embed_size, dtype)

  return embedding_encoder, embedding_decoder


def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

  # Cell Type
  if unit_type == "lstm":
    utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units,
        forget_bias=forget_bias)
  elif unit_type == "gru":
    utils.print_out("  GRU", new_line=False)
    single_cell = tf.contrib.rnn.GRUCell(num_units)
  elif unit_type == "layer_norm_lstm":
    utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
                    new_line=False)
    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True)
  elif unit_type == "nas":
    utils.print_out("  NASCell", new_line=False)
    single_cell = tf.contrib.rnn.NASCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # Dropout (= 1 - keep_prob)
  if dropout > 0.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
                    new_line=False)

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)
    utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

  # Device Wrapper
  if device_str:
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
    utils.print_out("  %s, device=%s" %
                    (type(single_cell).__name__, device_str), new_line=False)

  return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
  """Create a list of RNN cells."""
  if not single_cell_fn:
    single_cell_fn = _single_cell

  # Multi-GPU
  cell_list = []
  for i in range(num_layers):
    utils.print_out("  cell %d" % i, new_line=False)
    single_cell = single_cell_fn(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        device_str=get_device_str(i + base_gpu, num_gpus),
        residual_fn=residual_fn
    )
    utils.print_out("")
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):
  """Create multi-layer RNN cell.

  Args:
    unit_type: string representing the unit type, i.e. "lstm".
    num_units: the depth of each unit.
    num_layers: number of cells.
    num_residual_layers: Number of residual layers from top to bottom. For
      example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
      cells in the returned list will be wrapped with `ResidualWrapper`.
    forget_bias: the initial forget bias of the RNNCell(s).
    dropout: floating point value between 0.0 and 1.0:
      the probability of dropout.  this is ignored if `mode != TRAIN`.
    mode: either tf.contrib.learn.TRAIN/EVAL/INFER
    num_gpus: The number of gpus to use when performing round-robin
      placement of layers.
    base_gpu: The gpu device id to use for the first RNN cell in the
      returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
      as its device id.
    single_cell_fn: allow for adding customized cell.
      When not specified, we default to model_helper._single_cell
  Returns:
    An `RNNCell` instance.
  """
  cell_list = _cell_list(unit_type=unit_type,
                         num_units=num_units,
                         num_layers=num_layers,
                         num_residual_layers=num_residual_layers,
                         forget_bias=forget_bias,
                         dropout=dropout,
                         mode=mode,
                         num_gpus=num_gpus,
                         base_gpu=base_gpu,
                         single_cell_fn=single_cell_fn)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary, gradient_norm


def print_variables_in_ckpt(ckpt_path):
  """Print a list of variables in a checkpoint together with their shapes."""
  utils.print_out("# Variables in ckpt %s" % ckpt_path)
  reader = tf.train.NewCheckpointReader(ckpt_path)
  variable_map = reader.get_variable_to_shape_map()
  for key in sorted(variable_map.keys()):
    utils.print_out("  %s: %s" % (key, variable_map[key]))


def load_model(model, ckpt_path, session, name):
  """Load model from a checkpoint."""
  start_time = time.time()
  try:
    model.saver.restore(session, ckpt_path)
  except tf.errors.NotFoundError as e:
    utils.print_out("Can't load checkpoint")
    print_variables_in_ckpt(ckpt_path)
    utils.print_out("%s" % str(e))

  session.run(tf.tables_initializer())
  utils.print_out(
      "  loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt_path, time.time() - start_time))
  return model


def avg_checkpoints(model_dir, num_last_checkpoints, global_step,
                    global_step_name):
  """Average the last N checkpoints in the model_dir."""
  checkpoint_state = tf.train.get_checkpoint_state(model_dir)
  if not checkpoint_state:
    utils.print_out("# No checkpoint file found in directory: %s" % model_dir)
    return None

  # Checkpoints are ordered from oldest to newest.
  checkpoints = (
      checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])

  if len(checkpoints) < num_last_checkpoints:
    utils.print_out(
        "# Skipping averaging checkpoints because not enough checkpoints is "
        "avaliable."
    )
    return None

  avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
  if not tf.gfile.Exists(avg_model_dir):
    utils.print_out(
        "# Creating new directory %s for saving averaged checkpoints." %
        avg_model_dir)
    tf.gfile.MakeDirs(avg_model_dir)

  utils.print_out("# Reading and averaging variables in checkpoints:")
  var_list = tf.contrib.framework.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if name != global_step_name:
      var_values[name] = np.zeros(shape)

  for checkpoint in checkpoints:
    utils.print_out("    %s" % checkpoint)
    reader = tf.contrib.framework.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor

  for name in var_values:
    var_values[name] /= len(checkpoints)

  # Build a graph with same variables in the checkpoints, and save the averaged
  # variables into the avg_model_dir.
  with tf.Graph().as_default():
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
        for v in var_values
    ]

    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step_var = tf.Variable(
        global_step, name=global_step_name, trainable=False)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                             six.iteritems(var_values)):
        sess.run(assign_op, {p: value})

      # Use the built saver to save the averaged checkpoint. Only keep 1
      # checkpoint and the best checkpoint will be moved to avg_best_metric_dir.
      saver.save(
          sess,
          os.path.join(avg_model_dir, "translate.ckpt"))

  return avg_model_dir


def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step


def compute_perplexity(model, sess, name):
  """Compute perplexity of the output of the model.

  Args:
    model: model for compute perplexity.
    sess: tensorflow session to use.
    name: name of the batch.

  Returns:
    The perplexity of the eval outputs.
  """
  total_loss = 0
  total_predict_count = 0
  start_time = time.time()

  while True:
    try:
      output_tuple = model.eval(sess)
      total_loss += output_tuple.eval_loss * output_tuple.batch_size
      total_predict_count += output_tuple.predict_count
    except tf.errors.OutOfRangeError:
      break

  perplexity = utils.safe_exp(total_loss / total_predict_count)
  utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                   start_time)
  return perplexity
