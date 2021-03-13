"""Training script for sign language detection."""

import random

import tensorflow as tf
from absl import app
from sign_language_detection.args import FLAGS
from sign_language_detection.dataset import get_datasets
from sign_language_detection.model import build_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


def set_seed():
    """Set seed for deterministic random number generation."""
    seed = FLAGS.seed if FLAGS.seed is not None else random.randint(0, 1000)
    tf.random.set_seed(seed)
    random.seed(seed)


def main(unused_argv):
    """Keras training loop with early-stopping and model checkpoint."""

    set_seed()

    # Initialize Dataset
    train, dev, test = get_datasets()

    # Initialize Model
    model = build_model()

    # Train
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=FLAGS.stop_patience)
    mc = ModelCheckpoint(FLAGS.model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    with tf.device(FLAGS.device):
        model.fit(train,
                  epochs=FLAGS.epochs,
                  steps_per_epoch=FLAGS.steps_per_epoch,
                  validation_data=dev,
                  callbacks=[es, mc])

    best_model = load_model(FLAGS.model_path)
    print('Testing')
    best_model.evaluate(test)


if __name__ == '__main__':
    app.run(main)
