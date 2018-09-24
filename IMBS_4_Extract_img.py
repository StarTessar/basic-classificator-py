import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import numpy as nmp
import pickle as pickle
import os as OS

#   Присвоения и инициализация
PATH_DIR = "C:/TensFlow/Image_Borders/Output_Images/Binary_Format/"
PATH_SAVE = "C:/TensFlow/Image_Borders/Output_Images/Image_Format/"
PATH_TRANS = "C:/TensFlow/Image_Borders/Input_Images/Trans/"
#list_of_files = []
#dat_img = []
height = 212
width = 332
Iter_num = 1
ORIGIN_SIZE = (360, 240)

#   Работа с файлами
def get_file_list(loc_path_dir):
    list_of_files = []
    list_of_files.extend(OS.listdir(loc_path_dir))
    print("Список файлов получен!")
    return list_of_files

def get_bin_image_list(list_of_files, loc_path_dir):
    dat_img = []
    for loc_path in list_of_files:
        with open(loc_path_dir + loc_path, "rb") as f:
            dat_img.append(pickle.load(f))
    print("Изображения загружены!")
    return dat_img

def get_true_image_list(list_of_files, loc_path_dir):
    dat_img = []
    for loc_path in list_of_files:
        with Image.open(loc_path_dir + loc_path) as f:
            dat_img.append(f.load())
    print("Изображения загружены!")
    return dat_img

def save_true_image(loc_path_save, is_image, name_param):
    is_image.save(loc_path_save + "{0}{1}.png".format(Iter_num, name_param), "PNG")
    print("     Изображение: {0}{1}{2}.png - сохранено!".format(loc_path_save,Iter_num,name_param))

#   Преобразование файлов
def get_reshaped_bin_img_list(dat_img):
    for loc_img_num in range(len(dat_img)):
        dat_img[loc_img_num] = nmp.reshape(dat_img[loc_img_num], [-1, 2])

        locloc = nmp.reshape([0,0]*(height*width - len(dat_img[loc_img_num])), [-1, 2])
        dat_img[loc_img_num] = nmp.append(dat_img[loc_img_num], locloc)

        for num in range(len(dat_img[loc_img_num])):
            dat_img[loc_img_num][num]*=255

        dat_img[loc_img_num] = nmp.reshape(dat_img[loc_img_num], [-1, 2])
        dat_img[loc_img_num] = nmp.reshape(dat_img[loc_img_num], [-1, height, width, 2])
    print("Преобразование данных завершено!")

#   Создание изображений
def get_image_from_binary(dat_img):
    loc_image_storage = []
    save_num = 1
    for loc_img in dat_img:
        img_proto = Image.new("RGB",(width,height), (255,255,255))
        draw = ImageDraw.Draw(img_proto)

        for loc_y in range(height):
            for loc_x in range(width):
                ans_tr = loc_img[0, loc_y, loc_x, 0]
                ans_fl = loc_img[0, loc_y, loc_x, 1]
                if (ans_tr == 0) & (ans_fl == 0):
                    draw.point((loc_x, loc_y), (0, 0, 0))
                elif (ans_tr == 255) & (ans_fl == 255):
                    draw.point((loc_x, loc_y), (0, 0, 0))
                else:
                    # Режим
                    #loc_pix = (ans_tr + (255 - ans_fl))/2
                    loc_pix = ans_tr - ans_fl
                    loc_pix = loc_pix if loc_pix < 255 else 255
                    loc_pix = loc_pix if loc_pix > 0 else 0
                    draw.point((loc_x, loc_y), (int(loc_pix), int(loc_pix), int(loc_pix)))

        img_4_paste = Image.new("RGB", ORIGIN_SIZE)
        img_4_paste.paste(img_proto, ((ORIGIN_SIZE[0]-width)//2, (ORIGIN_SIZE[1]-height)//2))

        loc_image_storage.append(img_4_paste)

        save_true_image(PATH_SAVE, img_4_paste, save_num)

        #img_4_paste.save(PATH_SAVE + "{0}{1}.png".format(Iter_num, save_num), "PNG")
        #print("     Изображение: {0}{1}{2}.png - сохранено!".format(PATH_SAVE,Iter_num,save_num))
        save_num+=1
        del draw
    print("Выгрузка изображений завершена!")
    return loc_image_storage

def image_con(image_storage):
    list_of_files = get_file_list(PATH_TRANS)
    orig_image_storage = get_true_image_list(list_of_files, PATH_TRANS)
    conv_image_storage = []

    for loc_img in image_storage:
        conv_image_storage.append(loc_img.load())

    save_num = 1
    for loc_img in range(len(conv_image_storage)):
        img_proto = Image.new("RGB",ORIGIN_SIZE, (255,255,255))
        draw = ImageDraw.Draw(img_proto)

        for loc_y in range(ORIGIN_SIZE[1]):
            for loc_x in range(ORIGIN_SIZE[0]):
                pix_R = (orig_image_storage[loc_img][loc_x, loc_y][0] + conv_image_storage[loc_img][loc_x, loc_y][0]) // 2
                pix_G = (orig_image_storage[loc_img][loc_x, loc_y][1] + conv_image_storage[loc_img][loc_x, loc_y][1]) // 2
                pix_B = (orig_image_storage[loc_img][loc_x, loc_y][2] + conv_image_storage[loc_img][loc_x, loc_y][2]) // 2

                draw.point((loc_x, loc_y), (pix_R, pix_G, pix_B))

        save_true_image(PATH_SAVE, img_proto, str(save_num) + "_rdy")
        save_num+=1
        del draw
    print("Выгрузка результирующих изображений завершена!\n")

#   Интерфейс
def init_extract_image():
    print("\nНачинаю извлечение...")
    list_of_files = get_file_list(PATH_DIR)
    dat_img = get_bin_image_list(list_of_files, PATH_DIR)
    get_reshaped_bin_img_list(dat_img)
    image_storage = get_image_from_binary(dat_img)
    print("Извлечение завершено!\n")

    print("Начинаю слияние изображений...\n")
    image_con(image_storage)

#   Простой запуск
#init_extract_image()
