import tensorflow as tf
import numpy as nmp
import PIL.Image as Image
from datetime import datetime
import pickle as pickle
import os as OS

#   Присвоения и инициализация
PATH_LOAD_IMG = "C:/TensFlow/Image_Borders/Input_Images/Ready/Test_img/"
PATH_LOAD_GRAF = "C:/TensFlow/Image_Borders/Save_var/vol_1/"
PATH_SAVE_IMG = "C:/TensFlow/Image_Borders/Output_Images/Binary_Format/"

#   Класс выдачи кусочков для обучения
class Train_Net:
    def __init__(self, start_state):
        self.test = start_state

    def next_test(self, batch_sz, loc_arr):
        coun = self.test
        Xx = loc_arr[coun * batch_sz : (coun + 1) * batch_sz]
        self.test += 1
        if (self.test + 1) * batch_sz > len(loc_arr):
            self.test = 0
        return Xx

#   Функции создания слоёв
def add_conv_lay(input, size_stride, size_in, size_out, use_pool, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, [1, size_stride, size_stride, 1], padding='SAME')
        act = tf.nn.relu(conv + b)
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
        return act

#   Функция загрузки изображения
def load_img_from_bin(file_post_name = "Rdy_test_0.txt"):
    with open(PATH_LOAD_IMG + file_post_name, "rb") as f:
        ret_img = pickle.load(f)
    return nmp.reshape(ret_img, [-1, 28, 28, 3])

#   Работа с файлами
def get_file_list():
    list_of_files = OS.listdir(PATH_LOAD_IMG)
    print("Список файлов получен!")
    return list_of_files

#   Функция создания модели
def MNIST_model(learning_rate, use_two_conv, use_two_fc, hparam):
    print("\nПриступаю к созданию модели...\n")

    #   Параметры сессии
    tf.reset_default_graph()
    sess = tf.Session()

    #   Местозаполнитель входных значений
    X = tf.placeholder(tf.float32, [None, 28, 28, 3], name="X")

    #   Инициализация свёрочных слоёв
    if use_two_conv:
        conv1 = add_conv_lay(X, 1, 3, 16, False, "conv1")
        conv2 = add_conv_lay(conv1, 2, 16, 32, False, "conv2")
        conv_out = add_conv_lay(conv2, 2, 32, 64, False, "conv3")
    else:
        conv1 = add_conv_lay(X, 2, 3, 64, False, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #   Линеаризация выхода свёртки
    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    #   Инициализация полносвязных слоёв
    if use_two_fc:
        fc1 = add_full_con_lay(flattened, 7*7*64, 1024, "fc1")
        logits_w = add_full_con_lay(fc1, 1024, 2, "fc2", tf.nn.softmax)
    else:
        logits_w = add_full_con_lay(flattened, 7*7*64, 2, "fc")

    #   Инициализация переменных
    sess.run(tf.global_variables_initializer())

    #   Модуль сохранения/загрузки
    saver = tf.train.Saver()

    #   Старт таймера
    strt_time = datetime.now()

    #   Загрузка модели
    saver.restore(sess, PATH_LOAD_GRAF + "model.ckpt")
    print("\nМодель загружена!")

    #   Обработка каждого изображения отдельно
    file_list = get_file_list()
    num_save = 1

    print("\nОбработка изображений...")

    for loc_file in file_list:
        #   Инициализация
        tr_next_dat = Train_Net(0)
        glob_output = []

        print(" Загружается изображение {0}...".format(num_save))
        dat_test = load_img_from_bin(loc_file)

        upp_b = len(dat_test)//100

        print(" Обрабатывается изображение {0}...".format(num_save))

        for i in range(upp_b):
            batch_X = tr_next_dat.next_test(100, dat_test)
            loc_ext = sess.run(logits_w, feed_dict={X: batch_X})
            glob_output.extend(loc_ext)

        print("     Изображение {0} обработано! Длина массива: {1}".format(num_save, len(glob_output)))

        with open(PATH_SAVE_IMG + "Rdy_out_{0}.txt".format(num_save), "wb") as f:
            pickle.dump(glob_output, f)
            print("     Массив данных изображения {0} - сохранён!".format(num_save))

        num_save += 1

    print("\nГотово!".format(num_save))
    print("\nВремя работы: {0}\n".format(datetime.now() - strt_time))

    sess.close()

    #**********************************************************************************************************

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
  conv_param = "conv=2" if use_two_conv else "conv=1"
  fc_param = "fc=2" if use_two_fc else "fc=1"
  return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def go():
  for learning_rate in [1E-6]:

    for use_two_fc in [True]:
      for use_two_conv in [True]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
        hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
        print('Starting run for %s' % hparam)

        MNIST_model(learning_rate, use_two_fc, use_two_conv, hparam)

#+++++++++++++++++++++++++++++++++++++

#go()
