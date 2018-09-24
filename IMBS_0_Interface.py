import IMBS_1_DataSet_for_img_borders as IMBS_DATASET
import IMBS_2_Conv_img_borders as IMBS_TRAIN
import IMBS_3_Ready_model as IMBS_SCAN
import IMBS_4_Extract_img as IMBS_IMG_PROC
import winsound as WinSo

def more_opt():
    ch_p = input("\nБольше опций:\n    1 - Загруить новые изображения, просканировать и преобразовать.\n    2 - Переобучить сеть на новых входных данных.\n    0 - Обратно.\nВыберите опцию: ")
    print()
    if ch_p == '1':
        IMBS_DATASET.init_dataset_binarization(1)
        IMBS_SCAN.go()
        IMBS_IMG_PROC.init_extract_image()
    elif ch_p == '2':
        IMBS_DATASET.init_dataset_binarization(2)
        IMBS_TRAIN.go()
    elif ch_p == '0':
        return

def main_c():
    while True:
        ch_p = input("Главное меню:\n  1 - Создание обучающей выборки или преобразование для сканирования.\n  2 - Обучение нейросети.\n  3 - Сканирование изображений.\n  4 - Преобразование после сканирования.\n  5 - Больше опций.\n  0 - Выход.\nВыберите опцию: ")
        print()
        if ch_p == '1':
            IMBS_DATASET.init_dataset_binarization()
        elif ch_p == '2':
            IMBS_TRAIN.go()
        elif ch_p == '3':
            IMBS_SCAN.go()
        elif ch_p == '4':
            IMBS_IMG_PROC.init_extract_image()
        elif ch_p == '5':
            more_opt()
        elif ch_p == '0':
            return

        WinSo.MessageBeep()

main_c()
