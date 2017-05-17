# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import image_helper
from CaffeCls import CaffeCls
import os

normal_width, normal_height = (28, 28)

# 载入图像
image_color = cv2.imread("1.jpg")

# 灰度图像
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# 二值图像
ret, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)

# bit图像
ret, image_bit = cv2.threshold(image_gray, 127, 1, cv2.THRESH_BINARY_INV)

# 水平投影分割
horizontal_sum = np.sum(image_bit, axis=1)
peek_ranges = image_helper.extract_peek_ranges_from_array(horizontal_sum, minimun_val=20, minimun_range=3)
sub_images_binary = image_helper.split_from_peek_ranges(image_binary, peek_ranges, horizontal=True)
sub_images_bit = image_helper.split_from_peek_ranges(image_bit, peek_ranges, horizontal=True)
sub_images_color = image_helper.split_from_peek_ranges(image_color, peek_ranges, horizontal=True)
num_top, num_bottom = peek_ranges[3]

# 对第三行做形态学闭运算
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sub_images_bit3 = cv2.morphologyEx(sub_images_bit[3], cv2.MORPH_CLOSE, rectKernel)

# 借助闭运算做垂直投影分割
vertical_sum = np.sum(sub_images_bit3, axis=0)
peek_ranges = image_helper.extract_peek_ranges_from_array(vertical_sum, minimun_val=1, minimun_range=10)
num_left, num_right = peek_ranges[3]

# cv2.rectangle(image_color, (num_left-3, num_top-3), (num_right+3, num_bottom+3), (0, 255, 0), 3)
# cv2.imshow("win1", image_color)
# cv2.waitKey()

# 分割出票号图像
image_num_color = image_color[num_top:num_bottom, num_left:num_right+2]  # 此处加2不知道原因
image_num_binary = image_binary[num_top:num_bottom, num_left:num_right+2]
image_num_bit = image_bit[num_top:num_bottom, num_left:num_right+2]
# cv2.imshow("win1", image_num_binary)
# cv2.waitKey()

# 垂直投影分割票号数字
vertical_sum = np.sum(image_num_bit, axis=0)
peek_ranges = image_helper.extract_peek_ranges_from_array(vertical_sum, minimun_val=1, minimun_range=1)
sub_images_num_binary = image_helper.split_from_peek_ranges(image_num_binary, peek_ranges, horizontal=False)

# 保存分割出的票号数字
for i in xrange(len(sub_images_num_binary)):
    sub_images_num_binary[i] = image_helper.preprocess_crop_zeros(sub_images_num_binary[i])
    sub_images_num_binary[i] = image_helper.preprocess_resize_keep_ratio(sub_images_num_binary[i],
                                                                         normal_width, normal_height)
    # cv2.imwrite("chars/char{}.jpg".format(i.zfill), sub_images_num_binary[i])


# # 字符分割
# ret, image_bin = cv2.threshold(image_gray, 127, 1, cv2.THRESH_BINARY_INV)
# vertical_sum = np.sum(image_bin, axis=0)
# peek_ranges = image_helper.extract_peek_ranges_from_array(vertical_sum, minimun_val=1, minimun_range=1)
#
# # 画字符分割线
# image_helper.draw_lines_from_peek_ranges(image_color, peek_ranges, horizontal=False)
# # cv2.imshow("win1", image_color)
# # cv2.waitKey()
#
#
# chars = []
# for i in xrange(len(peek_ranges)):
#     char = image_threshold_inv[:, peek_ranges[i][0]:peek_ranges[i][1]+1]
#     char = image_helper.preprocess_crop_zeros(char)
#     char = image_helper.preprocess_resize_keep_ratio(char, normal_width, normal_height)
#     chars.append(char)


#for i in xrange(len(chars)):
#    cv2.imwrite("char%s.jpg" % i, chars[i])

# 字符识别
char_w = 28
char_h = 28
path_y_tag = "/home/leo/project/code/deep_ocr/workspace/caffe_dataset_digits/y_tag.json"
path_model_def = "/home/leo/project/code/deep_ocr/workspace/caffe_dataset_digits/lenet.prototxt"
model_weights = "/home/leo/project/code/deep_ocr/workspace/caffe_dataset_digits/lenet_iter_10000.caffemodel"
images = np.asarray(sub_images_num_binary)
images = images / 255.0
cls = CaffeCls(path_model_def, model_weights, path_y_tag,
               width=char_w, height=char_h)
ret = cls.predict_cv2_imgs(images)
result = ''
for i, item in enumerate(ret):
    s = item[0][0]
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    print ("%s , %s" % (s, item[0][1]))

