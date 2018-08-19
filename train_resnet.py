#coding=utf-8

import tensorflow as tf
import os
from ResNet50 import ResNet50

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.95, 'gpu占用内存比例')
tf.app.flags.DEFINE_integer('batch_size', 2, 'batch_size大小')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, '优化过程中的学习率')
tf.app.flags.DEFINE_integer('reload_model', 0, '是否reload之前训练好的模型')
tf.app.flags.DEFINE_string('model_dir', "./savemodel/", '保存模型的文件夹')
tf.app.flags.DEFINE_string('event_dir', "./event/", '保存event数据的文件夹,给tensorboard展示用')
min_dequeue = FLAGS.batch_size*3


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'channels': tf.FixedLenFeature([], tf.int64),
          'image_data': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_data'], tf.uint8)
  image = tf.reshape(image, [224, 224, 3])

  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  # image = tf.cast(image, tf.float32)

  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(filename, batch_size):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=None)

    image, label = read_and_decode(filename_queue)

    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=1,
        capacity=10*min_dequeue,
        min_after_dequeue=min_dequeue)

    return images, labels


def train():
    '''
    训练过程
    '''
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False, dtype=tf.int32)

    batch_size = FLAGS.batch_size
    train_images, train_labels = inputs("./tfrecord_data/train.tfrecord", batch_size)
    test_images, test_labels = inputs("./tfrecord_data/test.tfrecord", batch_size)
    verification_images, verification_labels = inputs("./tfrecord_data/ver.tfrecord", 1)

    train_labels_one_hot = tf.one_hot(train_labels, 4, on_value=1.0, off_value=0.0)
    test_labels_one_hot = tf.one_hot(test_labels, 4, on_value=1.0, off_value=0.0)

    learning_rate = FLAGS.learning_rate

    with tf.variable_scope("ResNet50") as scope:
        train_y_conv = ResNet50(train_images, num_classes=4).logits
        scope.reuse_variables()
        test_y_conv = ResNet50(test_images, num_classes=4).logits
        ver_prediction = ResNet50(verification_images, num_classes=4).predictions

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_one_hot, logits=train_y_conv))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)

    train_correct_prediction = tf.equal(tf.argmax(train_y_conv, 1), tf.argmax(train_labels_one_hot, 1))
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

    test_correct_prediction = tf.equal(tf.argmax(test_y_conv, 1), tf.argmax(test_labels_one_hot, 1))
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()

    saver = tf.train.Saver()
    tf.summary.scalar('cross_entropy_loss', cross_entropy)
    tf.summary.scalar('train_acc', train_accuracy)
    summary_op = tf.summary.merge_all()

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        if FLAGS.reload_model == 1:
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            save_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("reload model from %s, save_step = %d" % (ckpt.model_checkpoint_path, save_step))
        else:
            print("Create model with fresh paramters.")
            sess.run(init_op)
            sess.run(local_init_op)

        summary_writer = tf.summary.FileWriter(FLAGS.event_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                #train_images, train_labels_one_hot = sess.run([train_images, train_labels_one_hot])
                #test_images, test_labels_one_hot = sess.run([test_images, test_labels_one_hot])
                _, g_step = sess.run([train_op, global_step])
                if g_step % 2 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, g_step)

                if g_step % 100 == 0:
                    train_accuracy_value, loss = sess.run([train_accuracy, cross_entropy])
                    print("step %d training_acc is %.2f, loss is %.8f" % (g_step, train_accuracy_value, loss))
                if g_step % 500 == 0:
                    test_accuracy_value = sess.run(test_accuracy)
                    print("step %d test_acc is %.2f" % (g_step, test_accuracy_value))
                    if test_accuracy_value > 0.95:
                        for i in range(5):
                            ver_predict = sess.run(ver_prediction)
                            print("the prediction of verification image is:")
                            print(ver_predict)
                            ver_label = sess.run(verification_labels)
                            print("the verification label is %s" % ver_label)
                if g_step % 2000 == 0:
                    # 保存一次模型
                    print("save model to %s" % FLAGS.model_dir + "model.ckpt." + str(g_step))
                    saver.save(sess, FLAGS.model_dir + "model.ckpt", global_step=global_step)

        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == "__main__":
    if not os.path.exists("./tfrecord_data/train.tfrecord") or \
            not os.path.exists("./tfrecord_data/test.tfrecord"):
        gen_tfrecord_data("./data/train", "./data/test/")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # test()
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    train()
