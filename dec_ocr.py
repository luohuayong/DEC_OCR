# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import image_helper
# from CaffeCls import CaffeCls
import os

normal_width, normal_height = (28, 28)

# 载入图像
image_color = cv2.imread("2.jpg")

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

# 画分割线
# image_helper.draw_lines_from_peek_ranges(image_color, peek_ranges)
# cv2.imshow("win1", image_color)
# cv2.waitKey()

# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
# image_binary1 = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, rectKernel)
# cv2.imshow("win1", image_binary1)
# cv2.waitKey(0)


# 出票日期 1
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sub_images_bit1 = cv2.morphologyEx(sub_images_bit[1], cv2.MORPH_CLOSE, rectKernel)
vertical_sum1 = np.sum(sub_images_bit1, axis=0)
peek_ranges1 = image_helper.extract_peek_ranges_from_array(vertical_sum1, minimun_val=1, minimun_range=10)
num_top, num_bottom = peek_ranges[1]
num_left, num_right = peek_ranges1[1]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

# 汇票到期日/票号 2
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sub_images_bit2 = cv2.morphologyEx(sub_images_bit[2], cv2.MORPH_CLOSE, rectKernel)
vertical_sum2 = np.sum(sub_images_bit2, axis=0)
peek_ranges2 = image_helper.extract_peek_ranges_from_array(vertical_sum2, minimun_val=1, minimun_range=10)
num_top, num_bottom = peek_ranges[2]
num_left, num_right = peek_ranges2[1]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

num_left, num_right = peek_ranges2[3]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

# 出票人全称/收票人全称 3
# rectKernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
# sub_images_bit3 = cv2.morphologyEx(sub_images_binary[3], cv2.MORPH_CLOSE, rectKernel1)
# cv2.imshow("win1", sub_images_bit3)
# cv2.waitKey()

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sub_images_bit3 = cv2.morphologyEx(sub_images_bit[3], cv2.MORPH_CLOSE, rectKernel)
vertical_sum3 = np.sum(sub_images_bit3, axis=0)
peek_ranges3 = image_helper.extract_peek_ranges_from_array(vertical_sum3, minimun_val=1, minimun_range=10)
num_top, num_bottom = peek_ranges[3]
num_left, num_right = peek_ranges3[1]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

num_left, num_right = peek_ranges3[3]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

# 出票人账号/收票人账号 4
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
sub_images_bit4 = cv2.morphologyEx(sub_images_bit[4], cv2.MORPH_CLOSE, rectKernel)
vertical_sum4 = np.sum(sub_images_bit4, axis=0)
peek_ranges4 = image_helper.extract_peek_ranges_from_array(vertical_sum4, minimun_val=1, minimun_range=10)
num_top, num_bottom = peek_ranges[4]
num_left, num_right = peek_ranges4[3]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

num_left, num_right = peek_ranges4[7]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

# 出票人开户行/收票人开户行 5
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
sub_images_bit5 = cv2.morphologyEx(sub_images_bit[5], cv2.MORPH_CLOSE, rectKernel)
vertical_sum5 = np.sum(sub_images_bit5, axis=0)
peek_ranges5 = image_helper.extract_peek_ranges_from_array(vertical_sum5, minimun_val=1, minimun_range=10)
num_top, num_bottom = peek_ranges[5]
num_left, num_right = peek_ranges5[2]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

num_left, num_right = peek_ranges5[5]
cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)
# 票据金额 7
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
# sub_images_bit7 = cv2.morphologyEx(sub_images_bit[7], cv2.MORPH_CLOSE, rectKernel)
# vertical_sum7 = np.sum(sub_images_bit7, axis=0)
# peek_ranges7 = image_helper.extract_peek_ranges_from_array(vertical_sum7, minimun_val=1, minimun_range=10)
# num_top, num_bottom = peek_ranges[7]
# num_left, num_right = peek_ranges7[2]
# cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

# 承兑人全称/承兑人开户行行号 8
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
# sub_images_bit8 = cv2.morphologyEx(sub_images_bit[8], cv2.MORPH_CLOSE, rectKernel)
# vertical_sum8 = np.sum(sub_images_bit8, axis=0)
# peek_ranges8 = image_helper.extract_peek_ranges_from_array(vertical_sum8, minimun_val=1, minimun_range=10)
# num_top, num_bottom = peek_ranges[8]
# num_left, num_right = peek_ranges8[2]
# cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)
#
# num_left, num_right = peek_ranges8[4]
# cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

# 承兑人开户行名称 10
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
# sub_images_bit10 = cv2.morphologyEx(sub_images_bit[10], cv2.MORPH_CLOSE, rectKernel)
# vertical_sum10 = np.sum(sub_images_bit10, axis=0)
# peek_ranges10 = image_helper.extract_peek_ranges_from_array(vertical_sum10, minimun_val=1, minimun_range=10)
# num_top, num_bottom = peek_ranges[10]
# num_left, num_right = peek_ranges10[2]
# cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

# num_left, num_right = peek_ranges10[3]
# cv2.rectangle(image_color, (num_left-1, num_top-1), (num_right+1, num_bottom+1), (0, 255, 0), 1)

cv2.imshow("win1", image_color)
cv2.waitKey()

# vertical_sum = np.sum(image_binary1, axis=0)
# peek_ranges = image_helper.extract_peek_ranges_from_array(vertical_sum, minimun_val=1, minimun_range=10)


# 画分割线
# image_helper.draw_lines_from_peek_ranges(sub_images_color[3], peek_ranges, horizontal=False)
# cv2.imshow("win1", sub_images_color[3])
# cv2.waitKey()

# sub_images_bit4 = image_helper.split_from_peek_ranges(sub_images_bit[3], peek_ranges, horizontal=True)




# # 垂直投影分割
# chars2d_binary = []
# for i in xrange(len(sub_images_bit)):
#     # 字符分割
#     vertical_sum = np.sum(sub_images_bit[i], axis=0)
#     peek_ranges = image_helper.extract_peek_ranges_from_array(vertical_sum, minimun_val=1, minimun_range=1)
#     chars_binary = image_helper.split_from_peek_ranges(sub_images_binary[i], peek_ranges, horizontal=False)
#     for j in xrange(len(chars_binary)):
#         chars_binary[j] = image_helper.preprocess_crop_zeros(chars_binary[j])
#         chars_binary[j] = image_helper.preprocess_resize_keep_ratio(chars_binary[j], normal_width, normal_height)
#     chars2d_binary.append(chars_binary)
#
# for i in xrange(len(chars2d_binary)):
#     chars_binary = chars2d_binary[i]
#     for j in xrange(len(chars_binary)):
#         cv2.imwrite("chars/char%s_%s.jpg" % (i, j), chars_binary[j])
#

