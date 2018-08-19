# coding=utf-8

import tensorflow as tf
import os
import cv2

'''
本文件由图片生成tfrecord文件，保存的信息有图片的像素信息，图片的高和宽，图片的通道数，
图片对应的label，也就是图片是1还是0.

会生成训练的tfrecord文件和验证的tfrecord文件，
还会生成一个用来展示卷积过程中图片经过卷积变化的图片的tfrecord文件。
'''


def gen_tfrecord(zero_dir, left_dir, inversion_dir, right_dir, output_tfrecord_file):
    '''
    读取文件夹中的图片数据， 生成tfrecord格式的文件

    Args:
      zero_dir: 保存图片旋转0度的文件夹
      inversion_dir: 保存图片旋转180度的文件夹
      left_dir: 保存图片旋转90度的文件夹
      right_dir: 保存图片旋转270度的文件夹
      output_tfrecord_file: 输出的tfrecord文件

    Return:

    '''
    tf_writer = tf.python_io.TFRecordWriter(output_tfrecord_file)

    # 为0度的图片
    for file in os.listdir(zero_dir):
        file_path = os.path.join(zero_dir, file)
        image_data = cv2.imread(file_path)
        image_bytes = image_data.tostring()
        rows = image_data.shape[0]
        cols = image_data.shape[1]
        channels = image_data.shape[2]
        label_data = 0

        example = tf.train.Example()

        feature = example.features.feature
        feature['height'].int64_list.value.append(rows)
        feature['width'].int64_list.value.append(cols)
        feature['channels'].int64_list.value.append(channels)
        feature['image_data'].bytes_list.value.append(image_bytes)
        feature['label'].int64_list.value.append(label_data)

        tf_writer.write(example.SerializeToString())

    # 为90度的图片
    for file in os.listdir(left_dir):
        file_path = os.path.join(left_dir, file)
        image_data = cv2.imread(file_path)
        image_bytes = image_data.tostring()
        rows = image_data.shape[0]
        cols = image_data.shape[1]
        channels = image_data.shape[2]
        label_data = 1

        example = tf.train.Example()

        feature = example.features.feature
        feature['height'].int64_list.value.append(rows)
        feature['width'].int64_list.value.append(cols)
        feature['channels'].int64_list.value.append(channels)
        feature['image_data'].bytes_list.value.append(image_bytes)
        feature['label'].int64_list.value.append(label_data)

        tf_writer.write(example.SerializeToString())

    # 为180度的图片
    for file in os.listdir(inversion_dir):
        file_path = os.path.join(inversion_dir, file)
        image_data = cv2.imread(file_path)
        image_bytes = image_data.tostring()
        rows = image_data.shape[0]
        cols = image_data.shape[1]
        channels = image_data.shape[2]
        label_data = 2

        example = tf.train.Example()

        feature = example.features.feature
        feature['height'].int64_list.value.append(rows)
        feature['width'].int64_list.value.append(cols)
        feature['channels'].int64_list.value.append(channels)
        feature['image_data'].bytes_list.value.append(image_bytes)
        feature['label'].int64_list.value.append(label_data)

        tf_writer.write(example.SerializeToString())

    # 为270度图片
    for file in os.listdir(right_dir):
        file_path = os.path.join(right_dir, file)
        image_data = cv2.imread(file_path)
        image_bytes = image_data.tostring()
        rows = image_data.shape[0]
        cols = image_data.shape[1]
        channels = image_data.shape[2]
        label_data = 3

        example = tf.train.Example()

        feature = example.features.feature
        feature['height'].int64_list.value.append(rows)
        feature['width'].int64_list.value.append(cols)
        feature['channels'].int64_list.value.append(channels)
        feature['image_data'].bytes_list.value.append(image_bytes)
        feature['label'].int64_list.value.append(label_data)

        tf_writer.write(example.SerializeToString())

    tf_writer.close()


def gen_tfrecord_data(train_data_dir, test_data_dir):
    '''
    生成训练和测试的tfrecord格式的数据

    Args:
      train_data_dir: 训练图片数据的文件夹
      test_data_dir:  测试图片数据的文件夹

    Return:
    '''
    train_data_zero_dir = os.path.join(train_data_dir, "0")
    train_data_inversion_dir = os.path.join(train_data_dir, "180")
    train_data_left_dir = os.path.join(train_data_dir, "90")
    train_data_right_dir = os.path.join(train_data_dir, "270")

    test_data_zero_dir = os.path.join(test_data_dir, "0")
    test_data_inversion_dir = os.path.join(test_data_dir, "180")
    test_data_left_dir = os.path.join(test_data_dir, "90")
    test_data_right_dir = os.path.join(test_data_dir, "270")

    output_dir = "./tfrecord_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_tfrecord_file = os.path.join(output_dir, "train.tfrecord")
    test_tfrecord_file = os.path.join(output_dir, "test.tfrecord")

    gen_tfrecord(train_data_zero_dir, train_data_left_dir, train_data_inversion_dir, train_data_right_dir, train_tfrecord_file)
    gen_tfrecord(test_data_zero_dir, test_data_left_dir, test_data_inversion_dir, test_data_right_dir, test_tfrecord_file)


if __name__ == "__main__":
    gen_tfrecord_data("./data/train/", "./data/test/")
    print("generate tfrecord data complete!")
