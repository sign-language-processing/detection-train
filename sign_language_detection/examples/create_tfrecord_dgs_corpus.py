"""Code to create tfrecord for training from The Public DGS Corpus."""
import itertools
import os

import numpy as np
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
import tensorflow as tf
import tensorflow_datasets as tfds
from pose_format import Pose, PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from tqdm import tqdm

config = SignDatasetConfig(name="dgs-holistic", version="3.0.0", include_video=False, include_pose="holistic")
dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config))


def time_frame(ms, fps):
    return int(fps * (ms / 1000))


def get_pose_header():
    """Get pose header with components description."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    header_path = os.path.join(dir_path, '..', "holistic.poseheader")
    f = open(header_path, "rb")
    reader = BufferReader(f.read())
    header = PoseHeader.read(reader)
    header.components.pop(1)  # Remove face from holistic pose
    return header


# Body and two hands, ignoring the face
body_points = list(range(33)) + list(range(33 + 468, 33 + 468 + 21 * 2))

pose_header = get_pose_header()

with tf.io.TFRecordWriter('data.tfrecord') as writer:
    for datum in tqdm(itertools.islice(dgs_corpus["train"], 10)):
        elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
        sentences = get_elan_sentences(elan_path)

        for person in ["a", "b"]:
            frames = len(datum["poses"][person]["data"])
            fps = int(datum["poses"][person]["fps"].numpy())

            pose_data = datum["poses"][person]["data"].numpy()[:, :, body_points, :]
            pose_conf = datum["poses"][person]["conf"].numpy()[:, :, body_points]

            pose_body = NumPyPoseBody(fps=fps, data=pose_data, confidence=pose_conf)
            pose = Pose(header=get_pose_header(), body=pose_body)

            try:
                pose.normalize(pose.header.normalization_info(
                    p1=("POSE_LANDMARKS", "LEFT_SHOULDER"),
                    p2=("POSE_LANDMARKS", "RIGHT_SHOULDER")
                ))
            except:
                print("Skipping for empty pose")
                continue

            if pose_data.shape[0] > 0:  # Remove odd, 0 width examples
                is_signing = np.zeros(pose_data.shape[0], dtype='byte')

                for sentence in sentences:
                    if sentence["participant"].lower() == person:
                        for gloss in sentence["glosses"]:
                            start_frame = time_frame(gloss["start"], fps)
                            end_frame = time_frame(gloss["end"], fps)

                            is_signing[start_frame:end_frame + 1] = 1  # Sign detected

                print(pose.body.data.shape)

                if pose.body.data.shape[2] != 75:
                    print("Skipping for data out of shape", pose.body.data.shape)

                is_signing = is_signing.tobytes()

                pose = pose.tensorflow()

                features = {"is_signing": tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_signing]))}
                features.update(pose.body.as_tfrecord())

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
