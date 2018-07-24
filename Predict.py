import argparse
import os
import tensorflow as tf
#from model import Model
from ModelPredict import Model



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
	flags.DEFINE_boolean('is_training', False, 'whether to updates the running means and variances of BN during the training')
	flags.DEFINE_string('pretrain_file', None, 'pre-trained model filename')
	flags.DEFINE_string('data_list', 'train.txt', 'training data list filename')

	# testing / validation
	flags.DEFINE_integer('valid_step', 7000, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('valid_num_steps', 10, '= number of testing/validation samples')
	flags.DEFINE_string('valid_data_list', 'train.txt', 'testingvalidationdatalistfilename')

	# data
	flags.DEFINE_string('data_dir', '', 'data directory')
	flags.DEFINE_integer('batch_size', 1, 'training batch size')
	flags.DEFINE_integer('input_height', 800, 'input image height')
	flags.DEFINE_integer('input_width', 1000, 'input image width')
	flags.DEFINE_integer('num_classes', 28, 'number of classes')
	flags.DEFINE_integer('ignore_label', 0, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')

	# log
	flags.DEFINE_string('modeldir', 'model', 'model directory')
	flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log', 'training log directory')
	flags.DEFINE_string('outimage', 'outimage/', 'outimage directory')
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='predict',
		help='actions: train, test, or predict')
	args = parser.parse_args()

	if args.option not in ['train', 'test', 'predict']:


		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")



	else:



		# Set up tf session and initialize variables. 
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)
		sess = tf.Session()
		# Run
		model = Model(sess, configure())
		getattr(model, args.option)()


if __name__ == '__main__':
	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	tf.app.run()
