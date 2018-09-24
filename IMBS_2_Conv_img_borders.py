import tensorflow as tf
import numpy as nmp
import PIL.Image as Image
from datetime import datetime
import pickle as pickle

right_now_time = datetime.now()
PATH_SAVE_MODEL = "C:/TensFlow/Image_Borders/Save_var/vol_1/"
PATH_LOAD_DATA = "C:/TensFlow/Image_Borders/Input_Images/Ready/Train_data/"
LOGDIR = 'C:/TensFlow/Log/{0}_{1}_{2} ({3}-{4})__'.format(right_now_time.day, right_now_time.month, right_now_time.year, right_now_time.hour, right_now_time.minute)

dat_img = []
dat_ans = []

with open(PATH_LOAD_DATA + "Rdy_inp.txt", "rb") as f:
    dat_img = pickle.load(f)
with open(PATH_LOAD_DATA + "Rdy_ans.txt", "rb") as f:
    dat_ans = pickle.load(f)

xx = dat_img
yy = dat_ans

xx = nmp.reshape(xx, [-1, 28, 28, 3])
yy = nmp.reshape(yy, [-1, 2])

class Train_Net:
    def __init__(self, start_state):
        self.tr = start_state
        self.chk = start_state

    def next_dat(self, batch_sz):
        coun = self.tr
        Xx = xx[coun * batch_sz : (coun + 1) * batch_sz]
        Yy = yy[coun * batch_sz : (coun + 1) * batch_sz]
        self.tr += 1
        if (self.tr + 1) * batch_sz > len(yy):
            self.tr = 0
        return Xx, Yy

    def next_chk(self, batch_sz):
        coun = self.chk
        Xx = xx[coun * batch_sz : (coun + 1) * batch_sz]
        Yy = yy[coun * batch_sz : (coun + 1) * batch_sz]
        self.chk += 1
        if (self.chk + 1) * batch_sz > len(yy):
            self.chk = 0
        return Xx, Yy

tr_next_dat = Train_Net(0)

#=====================================

def add_conv_lay(input, size_stride, size_in, size_out, use_pool, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, [1, size_stride, size_stride, 1], padding='SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        if use_pool:
            return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            return act

def add_full_con_lay(input, size_in, size_out, name='fully_con', act_f=tf.nn.relu):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        m_mul = tf.matmul(input, w)
        act = act_f(m_mul + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def MNIST_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    X = tf.placeholder(tf.float32, [None, 28, 28, 3], name="X")
    tf.summary.image('a_input', X, 3)

    Y_ = tf.placeholder(tf.float32, [None, 2], name="labels")

    lr = tf.placeholder(tf.float32, 1, name="learning_rate")

    if use_two_conv:
        conv1 = add_conv_lay(X, 1, 3, 16, False, "conv1")
        conv2 = add_conv_lay(conv1, 2, 16, 32, False, "conv2")
        conv_out = add_conv_lay(conv2, 2, 32, 64, False, "conv3")
    else:
        conv1 = add_conv_lay(X, 2, 3, 64, False, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    loc_im1, _ = tf.split(conv1, [1, 15], 3)
    tf.summary.image('conv_img_1', loc_im1, 3)
    loc_im2, _ = tf.split(conv2, [1, 31], 3)
    tf.summary.image('conv_img_2', loc_im2, 3)
    loc_im3, _ = tf.split(conv_out, [1, 63], 3)
    tf.summary.image('conv_img_3', loc_im3, 3)

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if use_two_fc:
        fc1 = add_full_con_lay(flattened, 7*7*64, 1024, "fc1")
        #dropout
        fc1_dr = tf.nn.dropout(fc1, 0.75)
        logits_w = add_full_con_lay(fc1_dr, 1024, 2, "fc2", tf.nn.softmax)
    else:
        logits_w = add_full_con_lay(flattened, 7*7*64, 2, "fc")

    #logits = tf.nn.dropout(logits_w, 0.95)
    logits = logits_w

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        ler_rt = tf.scalar_mul(learning_rate, lr)
        train_step = tf.train.AdamOptimizer(ler_rt[0]).minimize(xent)
        tf.summary.scalar("learning_rate", ler_rt[0])

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    print("\nПриступаю к обучению...")

    strt_time = datetime.now()

    for i in range(6001):
        loc_lear = 2/nmp.math.log1p((i/100)+1)
        batch_X, batch_Y = tr_next_dat.next_dat(100)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={X: batch_X, Y_: batch_Y, lr: [loc_lear]})
            writer.add_summary(s, i)
        
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: [loc_lear]})

    print("\nОбучение завершено!\nВремя работы: {0}".format(datetime.now() - strt_time))

    saver.save(sess, PATH_SAVE_MODEL + "model.ckpt")
    print("\nМодель сохранена!")

    sess.close()
    #del dat_ans
    #del dat_img
    #del xx
    #del yy
    #del tr_next_dat

#**********************************************************************************************************


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
  conv_param = "conv=2" if use_two_conv else "conv=1"
  fc_param = "fc=2" if use_two_fc else "fc=1"
  return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def go():
  for learning_rate in [5E-5]:

    for use_two_fc in [True]:
      for use_two_conv in [True]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
        hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
        print('Starting run for %s' % hparam)

        MNIST_model(learning_rate, use_two_fc, use_two_conv, hparam)

#+++++++++++++++++++++++++++++++++++++

#go()
