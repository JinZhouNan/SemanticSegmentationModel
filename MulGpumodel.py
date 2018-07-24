from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from libs.nets import deeplabv3
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log

"""
This script trains or evaluates the model on augmented PASCAL VOC 2012 dataset.
The training set contains 10581 training images.
The validation set contains 1449 validation images.

Training:
'poly' learning rate
different learning rates for different layers
"""

IMG_MEAN = np.array([125.03455], dtype=np.float32)


class Model(object):
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf

    # train
    def train(self):
        self.train_setup()
        self.train_summary()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)

        # Load the pre-trained model if provided
        if self.conf.pretrain_file is not None:
            self.load(self.loader, self.conf.pretrain_file)
            original_step = int(self.conf.pretrain_file.split("-")[-1])
        else:
            original_step = 0

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # Train!
        num_steps = self.conf.num_steps + original_step
        # print "1"*33
        # print  self.conf.num_steps
        # print  step+original_step
        losstrans = 0
        for step in range(num_steps + 1):
            # print  step + original_step
            start_time = time.time()
            feed_dict = {self.curr_step: step + original_step}
            # print feed_dict
            if step % self.conf.save_interval == 0:
                loss_values, images, labels, preds, summary = self.sess.run(
                    [self.reduced_loss,
                     self.image_batch,
                     self.label_batch,
                     self.pred,
                     self.total_summary],
                    feed_dict={self.curr_step: step + original_step, self.loss_trans: losstrans})
                loss_value = loss_values
                self.summary_writer.add_summary(summary, step + original_step)
                self.save(self.saver, step + original_step)
            else:
                loss_collect = []
                # print self.conf.num
                for i in range(self.conf.num):
                    loss_value = self.sess.run(self.reduced_loss,
                                               feed_dict=feed_dict)
                    loss_collect.append(loss_value)
                loss_value = loss_value
                loss_values = np.sum(loss_collect) / len(loss_collect)
                _ = self.sess.run(self.train_op, {self.curr_step: step + original_step, self.loss_trans: loss_values})

            duration = time.time() - start_time
            print(
            'step {:d} \t loss = {:.3f}, {:.3f},({:.3f} sec/step)'.format(step + original_step, loss_values, loss_value,
                                                                          duration))
            write_log('{:d}, {:.3f}'.format(step, loss_values), self.conf.logfile)

        # finish
        self.coord.request_stop()
        self.coord.join(threads)
        self.coord.join(threads)

    # evaluate
    def test(self):
        self.test_setup()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # load checkpoint
        checkpointfile = self.conf.modeldir + '/model.ckpt-' + str(self.conf.valid_step)
        self.load(self.loader, checkpointfile)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # Test!
        for step in range(self.conf.valid_num_steps):
            preds, _, _ = self.sess.run([self.pred, self.accu_update_op, self.mIou_update_op])
            if step % 100 == 0:
                print('step {:d}'.format(step))
        print('Pixel Accuracy: {:.3f}'.format(self.accu.eval(session=self.sess)))
        print('Mean IoU: {:.3f}'.format(self.mIoU.eval(session=self.sess)))

        # finish
        self.coord.request_stop()
        self.coord.join(threads)

    def train_setup(self,reuse=False):
        tf.set_random_seed(self.conf.random_seed)
        num_layers=50#-----------------------------------------------------------------------------------------

        # Create queue coordinator.
        self.coord = tf.train.Coordinator()
        self.n_gpu=self.conf.n_gpu

        # Input size
        self.input_size = (self.conf.input_height, self.conf.input_width)
        j_step =0
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                self.conf.data_dir,
                self.conf.data_list,
                self.input_size,
                self.conf.random_scale,
                self.conf.random_mirror,
                self.conf.ignore_label,
                IMG_MEAN,
                self.coord)
            # print "1"*22
            # print reader
        image_data, image_label = reader.dequeue(self.conf.batch_size)
        self.image_data=  image_data
        if tf.__version__.startswith('1.'):
            split_train_data_node = tf.split(image_data, self.n_gpu)
            split_train_labels_node = tf.split(image_label, self.n_gpu)
        else:
            split_train_data_node = tf.split(0, self.n_gpu, image_data)
            split_train_labels_node = tf.split(0,self.n_gpu, image_label)
        with tf.variable_scope(tf.get_variable_scope()):
            all_loss=[ ]
            for device_index, (i, self.image_batch, self.label_batch) in enumerate(
                        zip([1], split_train_data_node, split_train_labels_node)):
                with tf.device('/gpu:%d' % i):
                    #print i
                    with tf.name_scope('%s_%d' % ("gpu", i)) as scope:
                        if j_step == 0:
                            j_step = 1
                            pass
                        else:
                            reuse = True
                        # net = DeepLab_v2_Network(self.image_batch, num_classes=self.conf.num_classes,
                        #                          is_training=self.conf.is_training ,reuse=reuse)
                        net, end_points = deeplabv3(self.image_batch,num_classes=self.conf.num_classes,depth=num_layers,is_training=True, reuse=reuse)
                        self.raw_output = end_points['gpu_{}/resnet{}/logits'.format(i,num_layers)]
                        # Network raw output
                        # [batch_size, 41, 41, 21]
                        output_size = (self.raw_output.shape[1].value, self.raw_output.shape[2].value)

                        label_proc = prepare_label(self.label_batch, output_size, num_classes=self.conf.num_classes,
                                                   one_hot=False)  # [batch_size, 41, 41]
                        raw_gt = tf.reshape(label_proc, [-1, ])
                        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
                        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                        raw_prediction = tf.reshape(self.raw_output, [-1, self.conf.num_classes])
                        # print raw_prediction
                        # print gt
                        prediction = raw_prediction
                        # prediction = tf.expand_dims(raw_prediction,  3)
                        # prediction = tl.act.pixel_wise_softmax(prediction)
                        # print prediction
                        # print label_proc
                        # loss = 1 - tl.cost.dice_coe(prediction, label_proc, axis=[1, 2, 3, 4])
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
                        l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                                     'weights' in v.name]
                        # Loss function
                        all_loss .append(tf.reduce_mean(loss) + tf.add_n(l2_losses))
                        tf.get_variable_scope().reuse_variables()
                        
        # Output size
        #output_size = (self.raw_output.shape[1].value, self.raw_output.shape[2].value)

        # Variables that load from pre-trained model.
        # For training, last few layers should not be loaded.
        if self.conf.pretrain_file is not None:
            restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
            original_step = int(self.conf.pretrain_file.split("-")[-1])
        else:
            original_step = 0
        num_steps=self.conf.num_steps+original_step
        # Trainable Variables
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # So we remove them from the list.
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
        # Fine-tune part
        conv_trainable = [v for v in all_trainable if 'fc' not in v.name]  # lr * 1.0
        # ASPP part
        fc_trainable = [v for v in all_trainable if 'fc' in v.name]
        # fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
        # fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
        # check
        #assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        #assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

        # Groud Truth: ignoring all labels greater or equal than n_classes
        #label_proc = prepare_label(self.label_batch, output_size, num_classes=self.conf.num_classes,
                                   #one_hot=False)  # [batch_size, 41, 41]
        #raw_gt = tf.reshape(label_proc, [-1, ])
        #indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
                        #gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                        #raw_prediction = tf.reshape(self.raw_output, [-1, self.conf.num_classes])
                        #prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax_cross_entropy loss
                        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        # L2 regularization
        #l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        # Loss function
        self.reduced_loss=tf.add_n(all_loss)/self.n_gpu
        #self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # Define optimizers
        # 'poly' learning rate
        base_lr = tf.constant(self.conf.learning_rate)
        self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
        self.loss_trans = tf.placeholder(dtype=tf.float32, shape=())
        self.final_loss = (self.reduced_loss + self.loss_trans) / 2

        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / num_steps), self.conf.power))
        #print self.conf.power
        self.learning_rate =learning_rate 
        #print  learning_rate
        # We have several optimizers here in order to handle the different lr_mult
        # which is a kind of parameters in Caffe. This controls the actual lr for each
        # layer.
        opt = tf.train.AdamOptimizer(learning_rate, self.conf.momentum,0.98)
        #opt= tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
        #opt_fc_w = tf.train.AdamOptimizer(learning_rate , self.conf.momentum,0.98)
        #opt_fc_b = tf.train.AdamOptimizer(learning_rate , self.conf.momentum,0.98)
        #opt_conv = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
        #opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.conf.momentum)
        #opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.conf.momentum)
        # To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
        # Instead, we separate the steps compute_grads+update_params.
        # Compute grads
        grads_conv = tf.gradients(self.final_loss, conv_trainable)
        # train_op = opt.apply_gradients(zip(grads_conv, conv_trainable))
        #grads = tf.gradients(self.reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
        grads_conv = grads_conv[:len(conv_trainable)]
        # grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
        # grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]
        # Update params
        train_op_conv=opt.apply_gradients(zip(grads_conv, conv_trainable))
        # train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
        # train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
        # train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
        # Finally, get the train_op!
        self.train_op = train_op_conv
        # Saver for storing checkpoints of the model
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

    # Loader for loading the pre-trained model
        if self.conf.pretrain_file is not None:
            self.loader = tf.train.Saver(var_list=restore_var)

    def train_summary(self):
        # Processed predictions: for visualisation.
        raw_output_up = tf.image.resize_bilinear(self.raw_output, self.input_size)
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        # print (self.pred)
        self.pred = tf.expand_dims(raw_output_up, dim=3)
        # Image summary.
        images_summary = tf.py_func(inv_preprocess, [self.image_batch, 1, IMG_MEAN], tf.uint8)
        #print  np.shape(self.label_batch)
        labels_summary = tf.py_func(decode_labels, [self.label_batch, 1, self.conf.num_classes], tf.uint8)
        #print  np.shape(self.pred)
        preds_summary = tf.py_func(decode_labels, [self.pred, 1, self.conf.num_classes], tf.uint8)

        self.total_summary = tf.summary.image('images',
                                              tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                              max_outputs=2)  # Concatenate row-wise.

        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        self.summary_writer = tf.summary.FileWriter(self.conf.logdir,
                                                    graph=tf.get_default_graph())

    def test_setup(self):
        # Create queue coordinator.
        num_layers = 50
        self.coord = tf.train.Coordinator()

        # Load reader
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                self.conf.data_dir,
                self.conf.valid_data_list,
                None,  # the images have different sizes
                False,  # no data-aug
                False,  # no data-aug
                self.conf.ignore_label,
                IMG_MEAN,
                self.coord)
            image, label = reader.image, reader.label  # [h, w, 3 or 1]
        # Add one batch dimension [1, h, w, 3 or 1]
        self.image_batch, self.label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)

        # Create network
        net, end_points = deeplabv3(self.image_batch, num_classes=self.conf.num_classes, depth=num_layers, is_training=True)
        raw_output = end_points['resnet{}/logits'.format(num_layers)]

        # predictions
        #raw_output = net.o  # [batch_size, 41, 41, 21]
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.image_batch)[1:3, ])
        raw_output = tf.argmax(raw_output, axis=3)
        pred = tf.expand_dims(raw_output, dim=3)
        self.pred = tf.reshape(pred, [-1, ])
        # labels
        gt = tf.reshape(self.label_batch, [-1, ])
        # Ignoring all labels greater than or equal to n_classes.
        temp = tf.less_equal(gt, self.conf.num_classes - 1)
        weights = tf.cast(temp, tf.int32)

        # fix for tf 1.3.0
        gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

        # Pixel accuracy
        self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
            self.pred, gt, weights=weights)

        # mIoU
        self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
            self.pred, gt, num_classes=self.conf.num_classes, weights=weights)

        # Loader for loading the checkpoint
        self.loader = tf.train.Saver(var_list=tf.global_variables())

    def save(self, saver, step):
        '''
        Save weights.
        '''
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(self.conf.modeldir, model_name)
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        saver.save(self.sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')

    def load(self, saver, filename):
        '''
        Load trained weights.
        '''
        saver.restore(self.sess, filename)
        print("Restored model parameters from {}".format(filename))
