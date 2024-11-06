# -*- coding:utf-8 -*-
"""
@Author  :
@Time    :
@FileName: json2yolo_txt.py
@Software: PyCharm
@Question: json格式标注文件转化为yolo的txt格式
"""

import os
import json
from pathlib import Path

json_path = '/home/qjc/DataSet/FLIR/FLIR_Aligned/flir_test_ir.json'  # json文件路径
out_path = '/home/qjc/DataSet/FLIR/FLIR_Aligned/labels_test/data/'  # 输出 txt 文件路径
Path(out_path).mkdir(parents=True, exist_ok=True)  # 若路径不存在则创建
# 读取 json 文件数据
with open(json_path, 'r') as load_f:
    content = json.load(load_f)


def process_classes():
    '''
    classes.txt文件处理
    '''
    # 处理class
    categories = content['categories']
    classes_txt = os.path.join(out_path, 'classes.txt')
    for categorie in categories:
        if os.path.exists(classes_txt):
            with open(classes_txt, mode="r+", encoding="utf-8") as fp:
                file_str = str(categorie['id'] - 1) + ':' + str(categorie['name'])
                line_data = fp.readlines()

                if len(line_data) != 0:
                    if file_str not in line_data:
                        fp.write('\n' + file_str)
                else:
                    fp.write(file_str)
        else:
            with open(classes_txt, mode="w+", encoding="utf-8") as fp:
                file_str = str(categorie['id'] - 1) + ':' + str(categorie['name'])
                fp.write(file_str)


def process_label():
    '''
    创建label.txt
    '''
    images = {image['id']: image for image in content['images']}
    annotations = content['annotations']

    for label in annotations:
        image_id = label['image_id']
        category_id = label['category_id'] - 1  # YOLO 格式中的类别从 0 开始

        if image_id in images:
            image = images[image_id]
            file_name = image['file_name']
            image_width = image['width']
            image_height = image['height']

            img_file_name = os.path.splitext(file_name)[0]
            label_txt = os.path.join(out_path, img_file_name + '.txt')

            x_center = (label['bbox'][0] + label['bbox'][2] / 2) / image_width
            y_center = (label['bbox'][1] + label['bbox'][3] / 2) / image_height
            width = label['bbox'][2] / image_width
            height = label['bbox'][3] / image_height

            file_str = f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            with open(label_txt, mode="a+", encoding="utf-8") as fp:
                fp.write(file_str + '\n')


process_classes()
process_label()
