#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : cifar10-tfrecord.py
# Desc  :
# Author: Len-Xu
# Date  : 2020/12/16 9:39
# Site  : China

import tensorflow as tf
import numpy as np
import os
import cv2


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def convert_to_tf_example(image, label):
    image = image.astype(np.uint8)
    img_h, img_w = image.shape[:2]
    image_string = image.tostring()
    feature = {
        'height': _int64_feature(img_h),
        'width': _int64_feature(img_w),
        'label': _int64_feature(int(label)),
        'image_raw': _bytes_feature(image_string)
    }
    # Create a tf.train.Features
    features = tf.train.Features(feature=feature)
    # Create example protocol
    example = tf.train.Example(features=features)
    return example


train_data, valid_data = tf.keras.datasets.cifar10.load_data()

all_data = {"train": train_data, 'valid': valid_data}
save_dir = "./data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for key in ["valid", 'train']:
    with tf.io.TFRecordWriter(os.path.join(save_dir, "cifar10_%s.tfrecord" % key)) as writer:
        images, labels = all_data[key]
        for image, label in zip(images, labels):
            tf_example = convert_to_tf_example(image, label)
            writer.write(tf_example.SerializeToString())

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, image_feature_description)
    return image_features


dataset = tf.data.TFRecordDataset("./data/cifar10_valid.tfrecord")
dataset = dataset.map(_parse_image_function)
dataset.as_numpy_iterator()
for image_feature in dataset.take(3):
    image = tf.io.decode_raw(image_feature['image_raw'], tf.uint8).numpy()
    img_h = image_feature['height']
    img_w = image_feature['width']
    image = image.reshape((img_h, img_w, -1))
    cv2.imshow("image", image)
    cv2.waitKey()
    break
