"""
数据集coco转txt
"""

import json
import os
from collections import defaultdict

#-------------------------------------------------------#
#   指向了COCO训练集与验证集图片的路径
#-------------------------------------------------------#
train_datasets_path     = "/home/chenshengjia/data/DAUB/"
val_datasets_path       = "/home/chenshengjia/data/DAUB/"

#-------------------------------------------------------#
#   指向了COCO训练集与验证集标签的路径
#-------------------------------------------------------#
train_annotation_path   = "/home/chenshengjia/data/DAUB/annotations/instances_train2017.json"
val_annotation_path     = "/home/chenshengjia/data/DAUB/annotations/instances_test2017.json"

#-------------------------------------------------------#
#   生成的txt文件路径
#-------------------------------------------------------#
train_output_path       = "coco_train.txt"
val_output_path         = "coco_val.txt"

def get_path(images, id):
    for image in images:
        if id == image["id"]:
            return image['file_name']
    

if __name__ == "__main__":
    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = open(train_annotation_path, encoding='utf-8')
    data        = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(train_datasets_path, get_path(images, id))
        cat = ant['category_id'] - 1
        name_box_id[name].append([ant['bbox'], cat])

    f = open(train_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()

    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = open(val_annotation_path, encoding='utf-8')
    data        = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(train_datasets_path, get_path(images, id))
        cat = ant['category_id']
        cat = cat - 1
        name_box_id[name].append([ant['bbox'], cat])

    f = open(val_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()