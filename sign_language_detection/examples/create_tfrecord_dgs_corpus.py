"""Code to create tfrecord for training from The Public DGS Corpus."""
import itertools

import numpy as np
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
import tensorflow as tf
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from tqdm import tqdm

config = SignDatasetConfig(name="dgs-holistic", version="3.0.0", include_video=False, include_pose="holistic")
dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config))


def time_frame(ms, fps):
    return int(fps * (ms / 1000))


# Body and two hands, ignoring the face
body_points = list(range(33)) + list(range(33 + 468, 33 + 468 + 21 * 2))

with tf.io.TFRecordWriter('data.tfrecord') as writer:
    for datum in tqdm(dgs_corpus["train"]):
        elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
        sentences = get_elan_sentences(elan_path)

        for person in ["a", "b"]:
            frames = len(datum["poses"][person]["data"])
            fps = int(datum["poses"][person]["fps"].numpy())

            pose_data = datum["poses"][person]["data"].numpy()[:, :, body_points, :]
            pose_conf = datum["poses"][person]["conf"].numpy()[:, :, body_points]

            if pose_data.shape[0] > 0: # Remove odd, 0 width examples
                is_signing = np.zeros(pose_data.shape[0], dtype='byte')

                for sentence in sentences:
                    if sentence["participant"].lower() == person:
                        for gloss in sentence["glosses"]:
                            start_frame = time_frame(gloss["start"], fps)
                            end_frame = time_frame(gloss["end"], fps)

                            is_signing[start_frame:end_frame + 1] = 1  # Sign detected

                is_signing = is_signing.tobytes()
                pose_data = tf.io.serialize_tensor(pose_data).numpy()
                pose_conf = tf.io.serialize_tensor(pose_conf).numpy()

                features = {
                    'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
                    'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_data])),
                    'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_conf])),
                    'is_signing': tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_signing]))
                }

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
