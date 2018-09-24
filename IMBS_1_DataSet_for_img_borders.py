import PIL.Image as Image
import numpy as nmp
import pickle as pickle
import os as OS
import random as rnd

#   Присвоения и инициализация
PATH_TRAIN_DATA = "C:/TensFlow/Image_Borders/Input_Images/Datas/"
PATH_TRAIN_MAP = "C:/TensFlow/Image_Borders/Input_Images/Maps/"
PATH_TRANS = "C:/TensFlow/Image_Borders/Input_Images/Trans/"
PATH_SAVE_DATA = "C:/TensFlow/Image_Borders/Input_Images/Ready/Train_data/"
PATH_SAVE_TRANS = "C:/TensFlow/Image_Borders/Input_Images/Ready/Test_img/"

#   Обработка комманды
def set_params():
    mode_choose = int(input("   1 - Датасет для обучения\n   2 - Преобразование для обработки\nВыерите режим: "))
    imp_num = int(input("Сколько изображений для импорта?: "))
    batch_size = int(input("Размер кусочка: "))
    stide_size = int(input("Шаг: "))

    return mode_choose, imp_num, batch_size, stide_size

#   Работа с файлами
def get_file_list(inp_num, loc_path):
    list_of_files = OS.listdir(loc_path)
    print("\nСписок файлов получен!")
    if inp_num == -1:
        return list_of_files[:len(list_of_files)]
    else:
        return list_of_files[:inp_num] 
    

#   Загрузка изображений для обучения
def get_train_image_list(list_of_files):
    img_data = []
    img_map = []
    img_size = []

    for loc_path in list_of_files:
        with Image.open(PATH_TRAIN_DATA + loc_path) as im:
            img_data.append(im.load())
            img_size.append([im.width, im.height])
        with Image.open(PATH_TRAIN_MAP + loc_path) as im:
            img_map.append(im.load())
    print("Изображения загружены!")
    return img_data, img_map, img_size

#   Загрузка изображений для сканирования
def get_test_image_list(list_of_files):
    img_data = []
    img_size = []
    
    for loc_path in list_of_files:
        with Image.open(PATH_TRANS + loc_path) as im:
            img_data.append(im.load())
            img_size.append([im.width, im.height])
    print("Изображения загружены!")
    return img_data, img_size

#   Подготовка массива данных для обучения
def get_train_data(img_data, img_map, img_size, batch_size, stide_size):
    data_train = []
    answer_train = []

    print("\nОбработка массива данных:")

    batch_border = batch_size // 2
    right_count = 0
    wrong_count = 0
    save_sum = 1

    for loc_img in img_data:
        loc_data = [[0]*(img_size[0][0]//1) for i in range(img_size[0][1])]

        for loc_y in range(img_size[0][1]):
            for loc_x in range(img_size[0][0]):
                loc_data[loc_y][loc_x] = loc_img[loc_x, loc_y]

        loc_img_lv2 = nmp.array(loc_data)

        for loc_y in range(batch_border, img_size[0][1] - batch_border, stide_size):
            for loc_x in range(batch_border, img_size[0][0] - batch_border, stide_size):
                l_x_start = loc_x-batch_border
                l_x_stop = loc_x-batch_border+batch_size
                l_y_start = loc_y-batch_border
                l_y_stop = loc_y-batch_border+batch_size

                lev_3 = loc_img_lv2[l_y_start:l_y_stop, l_x_start:l_x_stop]

                lev_3 = lev_3 / 255

                if (img_map[save_sum-1][loc_x, loc_y][0] == 255) & (img_map[save_sum-1][loc_x, loc_y][1] == 0) & (img_map[save_sum-1][loc_x, loc_y][2] == 0):
                    answer_train.append([1, 0])
                    data_train.append(lev_3)
                    right_count +=1
                else:
                    if (wrong_count / (right_count + 1) < 1.8) & (rnd.getrandbits(3) == 0):
                        answer_train.append([0, 1])
                        data_train.append(lev_3)
                        wrong_count +=1

        print("Изображение [ {0} / {1} ] - готово...".format(save_sum, len(img_data)))
        save_sum += 1

    print("\nМассив данных готов! Элементов в массиве: {0}".format(len(data_train)))

    print("\nЭлементов массива данных: {0}\nОтмеченных кусочков: {1}\nПустых кусочков: {2}\nОтношение пустые/отмеченные: {3:.2f}\n".format(len(data_train), 
                                                                                                                                           right_count, wrong_count, 
                                                                                                                                           wrong_count / right_count ))

    with open(PATH_SAVE_DATA + "Rdy_inp.txt", "wb") as f:
        pickle.dump(data_train, f)
        print("Массив данных сохранён!")

    with open(PATH_SAVE_DATA + "Rdy_ans.txt", "wb") as f:
        pickle.dump(answer_train, f)
        print("Массив ответов сохранён!")


#   Подготовка массива данных для сканирования
def get_trans_data(img_data, img_size, batch_size, stide_size):
    data_test = []

    print("\nОбработка массива данных:")

    batch_border = batch_size // 2
    loc_h = 0
    loc_w = 0

    save_num = 1

    for loc_img in img_data:
        data_test = []
        loc_data = [[0]*(img_size[0][0]//1) for i in range(img_size[0][1])]

        for loc_y in range(img_size[0][1]):
            for loc_x in range(img_size[0][0]):
                loc_data[loc_y][loc_x] = loc_img[loc_x, loc_y]

        loc_img_lv2 = nmp.array(loc_data)

        loc_h = 0
        for loc_y in range(batch_border, img_size[0][1] - batch_border, stide_size):
            lev_2 = []
            loc_h += 1
            loc_w = 0
            for loc_x in range(batch_border, img_size[0][0] - batch_border, stide_size):
                l_x_start = loc_x-batch_border
                l_x_stop = loc_x-batch_border+batch_size
                l_y_start = loc_y-batch_border
                l_y_stop = loc_y-batch_border+batch_size

                lev_3 = loc_img_lv2[l_y_start:l_y_stop, l_x_start:l_x_stop]
                loc_w += 1

                lev_3 = lev_3 / 255

                data_test.append(lev_3)
                #----
            #data_test.extend(lev_2)

        print("     Изображение [ {0} / {1} ] - готово...".format(save_num, len(img_data)))

        with open(PATH_SAVE_TRANS + "Rdy_test_{0}.txt".format(save_num), "wb") as f:
            pickle.dump(data_test, f)
        print("         Массив данных изображения {0} - сохранён!".format(save_num))
        save_num += 1

    print("\nИтоговые изображения размера: {0} х {1}\n".format(loc_h, loc_w))
    print("Массив для сканирования - сохранён!")

#   Интерфейс
def init_dataset_binarization(hparams = 0):
    if hparams == 0:
        #   Получение параметров работы
        mode_choose, imp_num, batch_size, stide_size = set_params()

        #   Выбор режима
        if mode_choose == 1:
            list_of_files = get_file_list(imp_num, PATH_TRAIN_DATA)
            img_data, img_map, img_size = get_train_image_list(list_of_files)
            get_train_data(img_data, img_map, img_size, batch_size, stide_size)
        elif mode_choose == 2:
            list_of_files = get_file_list(imp_num, PATH_TRANS)
            img_data, img_size = get_test_image_list(list_of_files)
            get_trans_data(img_data, img_size, batch_size, stide_size)
        else:
            print("Выбран неверный режим!")
    elif hparams == 1:
        list_of_files = get_file_list(-1, PATH_TRANS)
        img_data, img_size = get_test_image_list(list_of_files)
        get_trans_data(img_data, img_size, 28, 1)
    elif hparams == 2:
        list_of_files = get_file_list(-1, PATH_TRAIN_DATA)
        img_data, img_map, img_size = get_train_image_list(list_of_files)
        get_train_data(img_data, img_map, img_size, 28, 1)

    print("\nГотово!\n")

#   Простой старт
#init_dataset_binarization()
