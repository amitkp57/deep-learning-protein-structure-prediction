# Modified code from https://github.com/aqlaboratory/proteinnet compatible with the latest version of tensorflow

import json
import sys

import tensorflow as tf
from path import Path

NUM_AAS = 20
NUM_DIMENSIONS = 3
NUM_ATOMS = 3
NUM_SS = 8
SS_MAP = {v: k for k, v in dict(enumerate(list('LHBEGITS'))).items()}


def decode_fn(serialized_example):
    """
    Used to parse protein data in tensorflow
    :param serialized_example:
    :return:
    """
    return tf.io.parse_single_sequence_example(serialized_example,
                                               context_features={'id': tf.io.FixedLenFeature((1,), tf.string)},
                                               sequence_features={
                                                   'primary': tf.io.FixedLenSequenceFeature((1,), tf.int64),
                                                   'evolutionary': tf.io.FixedLenSequenceFeature(
                                                       (21,), tf.float32,
                                                       allow_missing=True),
                                                   'secondary': tf.io.FixedLenSequenceFeature((1,), tf.int64,
                                                                                              allow_missing=True),
                                                   'tertiary': tf.io.FixedLenSequenceFeature(
                                                       (NUM_DIMENSIONS), tf.float32, allow_missing=True),
                                                   'mask': tf.io.FixedLenSequenceFeature((1,), tf.float32,
                                                                                         allow_missing=True)})


def ss_to_int(ss):
    """
    Convert letters in secondary structure to int labels
    :param ss:
    :return:
    """
    return [SS_MAP[s] for s in ss]


def get_protein_to_ss(f_path):
    """
    Returns protein id to secondary structure mapping
    :param f_path:
    :return:
    """
    with open(f_path, 'r') as f:
        ss = json.load(f)
    id_to_ss = {k: ss_to_int(v['DSSP']) for k, v in ss.items()}
    return id_to_ss


def masking_matrix(mask):
    """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding.

    Args:
        mask: 0/1 vector indicating whether a position should be masked (0) or not (1)

    Returns:
        A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
    """

    mask = tf.expand_dims(mask, 0)
    base = tf.ones([tf.size(mask), tf.size(mask)])
    matrix_mask = base * mask * tf.transpose(mask)
    return matrix_mask


def read_protein(file_path, ss_file_path, max_length=512, min_length=64):
    """ Reads and parses a ProteinNet TF Record.

        Primary sequences are mapped onto 20-dimensional one-hot vectors.
        Evolutionary sequences are mapped onto num_evo_entries-dimensional real-valued vectors.
        Secondary structures are mapped onto ints indicating one of 8 class labels.
        Tertiary coordinates contains NUM_ATOMS * NUM_DIMENSIONS values for each residue
        Evolutionary, secondary, and tertiary entries are optional.

    Returns:
        id: string identifier of record
        one_hot_primary: AA sequence as one-hot vectors
        evolutionary: PSSM sequence as vectors
        on_hot_secondary: DSSP sequence as one-hot vectors
        tertiary: 3D coordinates of structure
        matrix_mask: Masking matrix to zero out pairwise distances in the masked regions
    """

    id, primary, evolutionary, secondary, tertiary, ter_mask = [], [], [], [], [], []
    id_to_ss = get_protein_to_ss(ss_file_path)
    for batch in tf.data.TFRecordDataset([file_path]).map(
            decode_fn):
        id_ = batch[0]['id'][0]
        evolutionary_ = batch[1]['evolutionary']
        mask_ = batch[1]['mask'][:, 0]
        primary_ = tf.cast(batch[1]['primary'][:, 0], tf.int32)
        p_id = id_.numpy().decode("utf-8")[3:]
        if p_id not in id_to_ss:
            continue
        secondary_ = tf.cast(id_to_ss[p_id], tf.int32)  # tf.cast(batch[1]['secondary'][:, 0], tf.int32)
        tertiary_ = batch[1]['tertiary'][2::3]  # select beta carbons

        pri_length = primary_.shape[0]
        keep = (min_length <= pri_length <= max_length)
        primary_ = tf.one_hot(primary_, NUM_AAS)
        secondary_ = tf.one_hot(secondary_, NUM_SS)
        # Generate tertiary masking matrix--if mask is missing then assume all residues are present
        mask_ = tf.cond(tf.not_equal(tf.size(mask_), 0), lambda: mask_, lambda: tf.ones([pri_length]))
        ter_mask_ = masking_matrix(mask_)

        if keep:
            padding = max_length - pri_length
            id.append(id_)
            primary.append(tf.pad(primary_, tf.constant([[0, padding], [0, 0]])))
            evolutionary.append(tf.pad(evolutionary_, tf.constant([[0, padding], [0, 0]])))
            secondary.append(tf.pad(secondary_, tf.constant([[0, padding], [0, 0]])))
            tertiary.append(tf.pad(tertiary_, tf.constant([[0, padding], [0, 0]])))
            ter_mask.append(tf.pad(ter_mask_, tf.constant([[0, padding], [0, padding]])))

    return id, primary, secondary, tertiary, evolutionary, ter_mask
    # return tf.stack(id, axis=0), tf.stack(primary, axis=0), tf.stack(evolutionary, axis=0), tf.stack(
    #     secondary, axis=0), tf.stack(tertiary, axis=0), tf.stack(ter_mask, axis=0)


if __name__ == '__main__':
    ss_path = sys.argv[1]
    protein_dir = sys.argv[2]
    id, primary, evolutionary, secondary, tertiary, ter_mask = [], [], [], [], [], []
    for f_path in Path(protein_dir).walkfiles():
        id_, primary_, evolutionary_, secondary_, tertiary_, ter_mask_ = read_protein(f_path, ss_path, min_length=64,
                                                                                      max_length=512)
        id.extend(id_)
        primary.extend(primary_)
        evolutionary.extend(evolutionary_)
        secondary.extend(secondary_)
        tertiary.extend(tertiary_)
        ter_mask.extend(ter_mask_)
    results = tf.stack(id), tf.stack(primary), tf.stack(secondary), tf.stack(tertiary).tf.stack(evolutionary), tf.stack(
        ter_mask)
