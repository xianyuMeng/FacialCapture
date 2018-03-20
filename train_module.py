from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ModelConfig import ModelConfig
from VisualModule import Module

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

#tf.flags.DEFINE_string("input_file_pattern", "",
#                               "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "",
                               "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 500000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                                "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("save_interval_secs", 60,
                                "Frequency at which checkpoints are saved.")
tf.flags.DEFINE_integer("save_summaries_secs", 10,
                                "The frequency with which summaries are saved, in seconds.")

def main(_):
    #assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"
    model_config = ModelConfig()
#    model_config.file_pattern = FLAGS.input_file_pattern
    print("file pattern : {}".format(model_config.file_pattern))
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)
    model = Module(model_config, mode="train")
    model.build()
#    learning_rate = 0.000001
#    learning_rate = tf.train.exponential_decay(
#        0.001,
#        model.global_step,
#        decay_steps=5000,
#        decay_rate=0.1,
#        staircase=True)
    learning_rate = tf.train.polynomial_decay( \
	0.00001, \
	model.global_step, \
	20000, \
	0.001, \
	power = 0.5)
    tf.summary.scalar("learning_rate", learning_rate)
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=model_config.optimizer,
        clip_gradients=model_config.clip_gradients)
    saver = tf.train.Saver(max_to_keep=model_config.max_checkpoint_keep)

    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        saver=saver,
        save_interval_secs=FLAGS.save_interval_secs,
        save_summaries_secs=FLAGS.save_summaries_secs)


if __name__ == "__main__":
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    tf.app.run()
