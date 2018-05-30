import tensorflow as tf
import argparse
import os
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
  # flags.out_dir = os.path.join(tf.test.get_temp_dir(), test_name)
  # Need train a model and save the model to `nmt/testdata/models`
  flags.out_dir = "nmt/testdata/models"
  print(flags.out_dir)
  flags.export_path = os.path.join(flags.out_dir, "export")
  flags.version_number = None
  flags.ckpt_path = None
  flags.infer_file = "nmt/testdata/test_infer_file"


class TestExporter(tf.test.TestCase):

  def test_exporter(self):
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    _update_flags(FLAGS, "exporter_test")
    default_hparams = nmt.create_hparams(FLAGS)
    nmt.run_main(FLAGS, default_hparams, train.train, None)


if __name__ == "__main__":
  tf.test.main()
