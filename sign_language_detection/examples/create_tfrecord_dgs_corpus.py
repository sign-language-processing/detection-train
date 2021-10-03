"""Example code to create tfrecord for training from the dgs corpus."""

import numpy as np
import itertools
import tensorflow as tf

import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

config = SignDatasetConfig(name="dgs-holistic", version="3.0.0", include_video=False, include_pose="holistic")
dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config))

with tf.io.TFRecordWriter('example.tfrecord') as writer:
    for datum in itertools.islice(dgs_corpus["train"], 0, 10):
        pose = datum["pose"]
        frames = len(pose["data"])  # Number of frames in the example video
        fps = 50  # dgs corpus os of constant fps

        print(pose.shape)

        die

        is_signing = np.random.randint(low=0, high=1, size=(frames), dtype='byte').tobytes()
        data = tf.io.serialize_tensor(tf.random.normal(shape=(frames, 1, 137, 2), dtype=tf.float32)).numpy()
        confidence = tf.io.serialize_tensor(tf.random.normal(shape=(frames, 1, 137), dtype=tf.float32)).numpy()

        features = {
            'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
            'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
            'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence])),
            'is_signing': tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_signing]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
