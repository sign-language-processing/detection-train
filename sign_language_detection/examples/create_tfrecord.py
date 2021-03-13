"""Example code to create tfrecord for training."""

import numpy as np
import tensorflow as tf

with tf.io.TFRecordWriter('example.tfrecord') as writer:
    for _ in range(5):  # Iterate over 5 examples
        frames = 100  # Number of frames in the example video
        fps = 25  # FPS in the example video

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
