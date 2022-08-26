"""Code to create tfrecord for training from The Public DGS Corpus."""
import itertools

import numpy as np
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
import tensorflow as tf
import tensorflow_datasets as tfds
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from tqdm import tqdm
from sign_language_detection.dataset import get_pose_header

config = SignDatasetConfig(name="dgs-holistic", version="3.0.0", include_video=False, include_pose="holistic")
dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config))

pose_header = get_pose_header()


def time_frame(ms, fps):
    return int(fps * (ms / 1000))


def hide_legs(pose: Pose):
    point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
    # pylint: disable=protected-access
    points = [pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
              for n in point_names for side in ["LEFT", "RIGHT"]]
    pose.body.confidence[:, :, points] = 0
    pose.body.data[:, :, points, :] = 0


def load_pose(tf_pose):
    fps = int(tf_pose["fps"].numpy())

    pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
    pose = Pose(pose_header, pose_body)

    # Get subset of components
    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    # Normalize to shoulderwidth
    pose = pose.normalize(pose.header.normalization_info(
        p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
        p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
    ))

    # Remove legs
    hide_legs(pose)

    # Data without Z axis
    data = pose.body.data.data[:, :, :, :2]
    conf = pose.body.confidence

    return data, conf, fps


with tf.io.TFRecordWriter('data.tfrecord') as writer:
    for datum in tqdm(dgs_corpus["train"]):
        elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
        sentences = list(get_elan_sentences(elan_path))

        for person in ["a", "b"]:
            pose_data, pose_conf, fps = load_pose(datum["poses"][person])
            frames = len(pose_data)

            if pose_data.shape[0] > 0:  # Remove odd, 0 width examples
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
