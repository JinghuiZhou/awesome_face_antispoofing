# coding: utf8
import os
import numpy as np
from PIL import Image
import argparse 
def traversalDir_FirstDir(path):
    dir_list = []
# 判断路径是否存在
    if (os.path.exists(path)):
        # 获取该目录下的所有文件或文件夹目录
        files = os.listdir(path)
        for file in files:
            # 得到该文件下所有目录的路径
            m = os.path.join(path, file)
            # 判断该路径下是否是文件夹
            if (os.path.isdir(m)):
                dir_list.append(m)
    return dir_list


def require_img_label(path):
    img_label = []
    for i in traversalDir_FirstDir(path):
        for root, dirs, files in os.walk(i):
            # print(files)
            #s = root.split("\\")     #windows
            #s = root.split("/")       #linux
            # print(s[-1])
            for filename in files:
                if filename.endswith("jpg") or filename.endswith("JPG"):
                    f1 = root + '\\' + filename
                    f1 = f1.replace('\\', '/')
                    #tup = tuple((f1, s[-1]))
                    # img_list.append(f1)
                    # lable_list.append(s[-1])
                    img_label.append(f1)
                    # print(tup)
    return img_label


def changeDir_name(path):
    # 定义一个列表，用来存储结果
    dir_list = []
# 判断路径是否存在
    if (os.path.exists(path)):
        # 获取该目录下的所有文件或文件夹目录
        files = os.listdir(path)
        for dir_name in files:
            if dir_name in ndict.keys():
                #print(path + '\\' + dir_name)
                if os.path.exists(path + '\\' + ndict[dir_name]):
                    pass
                else:
                    os.rename(path + '\\' + dir_name,
                              path + '\\' + ndict[dir_name])


def write_txt(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            #print(item)
            line = '%s' % item+'\n'
            fout.write(line)


def make_txt(path, rand_shuff, chunks, tra_ratio, tes_ratio):
    imglab_list = require_img_label(path)
    #print(imglab_list)
    if rand_shuff is True:
        np.random.seed(100)
        np.random.shuffle(imglab_list)
    # print(imglab_list)
    N = len(imglab_list)

    chunk_size = (N + chunks - 1) // chunks
    print(chunk_size)

    for i in range(chunks):
        chunk = imglab_list[i * chunk_size:(i + 1) * chunk_size]

        if chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''

        sep = int(chunk_size * tra_ratio)
        sep_test = int(chunk_size * tes_ratio)
        #if tes_ratio == 1.0:
            #print(chunk)
        write_txt(path+str_chunk + 'test.txt', chunk[:sep_test])
       



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str,\
                        default='AgriculturalDisease_testA/', 
                        help='img firtst address')
    parser.add_argument('--shuffle', type=bool,\
                        default=False, 
                        help='file shuffle')
    parser.add_argument('--chunks', type=int,\
                        default=1, 
                        help='how many img to deal every step')

    args = parser.parse_args()
    make_txt(args.filepath, args.shuffle, args.chunks, 0, 1.0)
