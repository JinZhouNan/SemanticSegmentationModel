import argparse
import os
import tensorflow as tf
from MulGpumodel import Model
"""
This script defines hyperparameters.
"""


def configure():
    flags = tf.app.flags

    # training
    flags.DEFINE_integer('num_steps', 200000, 'maximum number of iterations')
    flags.DEFINE_integer('save_interval', 1000, 'number of iterations for saving and visualization')
    flags.DEFINE_integer('random_seed', 1234, 'random seed')
    flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
    flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate')
    flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
    flags.DEFINE_float('momentum', 0.9, 'momentum')
    flags.DEFINE_boolean('is_training', False,
                         'whether to updates the running means and variances of BN during the training')
    flags.DEFINE_string('pretrain_file', None, 'pre-trained model filename')
    flags.DEFINE_string('data_list', 'train.txt',
                        'training data list filename')

    # testing / validation
    flags.DEFINE_integer('valid_step', 10000, 'checkpoint number for testing/validation')
    flags.DEFINE_integer('valid_num_steps', 19000, '= number of testing/validation samples')
    flags.DEFINE_string('valid_data_list', 'val.txt',
                        'testing/validation data list filename')

    # data
    flags.DEFINE_string('data_dir', '', 'data directory')
    flags.DEFINE_integer('batch_size', 1, 'training batch size')
    flags.DEFINE_integer('input_height', 276, 'input image height')
    flags.DEFINE_integer('input_width', 276, 'input image width')
    flags.DEFINE_integer('num_classes', 28, 'number of classes')
    flags.DEFINE_integer('ignore_label', 0, 'label pixel value that should be ignored')
    flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
    flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
    flags.DEFINE_integer('n_gpu', 1, "number of used gpu")
    flags.DEFINE_integer('num', 500, "number of used batch get loss")

    # log
    flags.DEFINE_string('modeldir', 'model', 'model directory')
    flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
    flags.DEFINE_string('logdir', 'log', 'training log directory')
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()

    if args.option not in ['train', 'test', 'predict']:

        print('invalid option: ', args.option)
        print("Please input a option: train, test, or predict")



    else:
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        model = Model(sess, configure())
        getattr(model, args.option)()


if __name__ == '__main__':
    # Choose which gpu or cpu to us
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    tf.app.run()
