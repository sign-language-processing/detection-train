"""Training command line arguments."""
from absl import flags

FLAGS = flags.FLAGS

# Training flags
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_string('device', '/GPU:0', 'Tensorflow device')
flags.DEFINE_string('model_path', 'model.h5', 'Path to save trained model')
flags.DEFINE_integer('epochs', 1, 'Maximum number of epochs')
flags.DEFINE_integer('steps_per_epoch', 1, 'Number of batches per epoch')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('stop_patience', 3, 'Patience for early stopping')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training')
flags.DEFINE_integer('test_batch_size', 1, 'Batch size for evaluation')

# Model flags
flags.DEFINE_float('input_dropout', 0.5, 'Input dropout rate')
flags.DEFINE_integer('encoder_layers', 1, 'Number of RNN layers')
flags.DEFINE_bool('encoder_bidirectional', False, 'Use a bidirectional encoder?')
flags.DEFINE_integer('hidden_size', 2 ** 6, 'RNN hidden state size')

# Augmentation flags
flags.DEFINE_float('frame_dropout_std', 0.3, 'Augmentation drop frames std')

# Dataset flags
flags.DEFINE_string('dataset_path', "examples/data.tfrecord", 'Location of tfrecord file')
flags.DEFINE_integer('input_size', 75, 'Number of pose points')

flags.mark_flag_as_required('dataset_path')
flags.mark_flag_as_required('input_size')
