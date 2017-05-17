# -*- coding: utf-8 -*-
import cv2
import numpy as np
"""
以通栏表格线横向分割图像
"""
# def extract_peek_ranges_from_lines(image_binary, minimun_val=10, horizontal=True):
#     y, x = image_binary.shape
#     x = int(x)
#     y = int(y)
#     lines = np.array([0, 0, 0])
#     for i in xrange(y):
#         line = image_threshold[i, :]
#         # print np.sum(line)/255
#         if np.sum(line) / 255 < int(x * 0.1):
#             lines = np.row_stack((lines, [i, np.sum(line) / 255, np.sum(line) / 255 / x]))

"""
根据投影图获取分割数组
"""
def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val >= minimun_val and start_i is None:
            start_i = i
        elif val >= minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i-1))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
        if i == len(array_vals) - 1 and val >= minimun_val:
            end_i = i
            peek_ranges.append((start_i, end_i-1))
    return peek_ranges

"""
根据投影图获取分割数组
minimun_val 分割线穿墙值，小于此值穿过
minimun_foreground 前景有效宽度，大于此值算有效前景
minimun_background 背景有效宽度，大于此值算有效背景
"""
# def extract_peek_ranges_from_array1(array_vals, minimun_val=10, minimun_foreground=2, minimun_background=1)
#     array_temp = np.array(array_vals)
#     for i in xrange(len(array_temp)):
#         if array_temp[i] >= minimun_val:
#             array_temp[i]=1
#         else:
#             array_temp[i]=0
#     for i in xrange(len(array_temp)):
#         if array_temp[i]==0:
#
#
#     start_i = None
#     end_i = None
#     peek_ranges = []
#     for i, val in enumerate(array_vals):
#         if val < minimun_val and :
#             if end_i is None or start_i - end_i >= minimun_background:
#                 start_i = i
#                 is_foreground == True
#         elif val < minimun_val and is_foreground == True:
#             if end_i - start_i >= minimun_foreground:
#
#             end_i = i
#             if end_i - start_i >= minimun_foreground:
#                 peek_ranges.append((start_i, end_i-1))
#             start_i = None
#             end_i = None
#         elif val < minimun_val and start_i is None:
#             pass
#         else:
#             raise ValueError("cannot parse this case...")
#         if i == len(array_vals) - 1 and val >= minimun_val:
#             end_i = i
#             peek_ranges.append((start_i, end_i-1))
#     return peek_ranges

"""
根据分割数组划线
"""
def draw_lines_from_peek_ranges(image, peek_ranges, horizontal=True):
    green = (0, 255, 0)
    x = int(image.shape[1])
    y = int(image.shape[0])
    for i, peek_range in enumerate(peek_ranges):
        if horizontal:
            pt1 = (0, peek_range[0]-1)
            pt2 = (x, peek_range[0]-1)
            pt3 = (0, peek_range[1]+1)
            pt4 = (x, peek_range[1]+1)
        else:
            pt1 = (peek_range[0]-1, 0)
            pt2 = (peek_range[0]-1, y)
            pt3 = (peek_range[1]+1, 0)
            pt4 = (peek_range[1]+1, y)
        cv2.line(image, pt1, pt2, green, 1)
        cv2.line(image, pt3, pt4, green, 1)

"""
根据分割数组分割图像
"""
def split_from_peek_ranges(image, peek_ranges, horizontal=True):
    sub_images = []
    if horizontal:
        for i in xrange(len(peek_ranges)):
            item = image[peek_ranges[i][0]:peek_ranges[i][1] + 1, :]
            sub_images.append(item)
    else:
        for i in xrange(len(peek_ranges)):
            item = image[:, peek_ranges[i][0]:peek_ranges[i][1] + 1]
            sub_images.append(item)
    return sub_images

    green = (0, 255, 0)
    x = int(image.shape[1])
    y = int(image.shape[0])
    for i, peek_range in enumerate(peek_ranges):
        if horizontal:
            pt1 = (0, peek_range[0]-1)
            pt2 = (x, peek_range[0]-1)
            pt3 = (0, peek_range[1]+1)
            pt4 = (x, peek_range[1]+1)
        else:
            pt1 = (peek_range[0]-1, 0)
            pt2 = (peek_range[0]-1, y)
            pt3 = (peek_range[1]+1, 0)
            pt4 = (peek_range[1]+1, y)
        cv2.line(image, pt1, pt2, green, 1)
        cv2.line(image, pt3, pt4, green, 1)

"""
去除边缘空白
"""
def preprocess_crop_zeros(image_bin):
    height = image_bin.shape[0]
    width = image_bin.shape[1]
    v_sum = np.sum(image_bin, axis=0)
    h_sum = np.sum(image_bin, axis=1)
    left = 0
    right = width - 1
    top = 0
    low = height - 1

    for i in range(width):
        if v_sum[i] > 0:
            left = i
            break

    for i in range(width - 1, -1, -1):
        if v_sum[i] > 0:
            right = i
            break

    for i in range(height):
        if h_sum[i] > 0:
            top = i
            break

    for i in range(height - 1, -1, -1):
        if h_sum[i] > 0:
            low = i
            break
    if not (top < low and right > left):
        return image_bin

    return image_bin[top: low + 1, left: right + 1]


"""
增加内边距
"""
def add_padding(image_bin, value=1):
    height = image_bin.shape[0]
    width = image_bin.shape[1]

    image_new = np.zeros((height + value * 2, width + value * 2), np.uint8)
    image_new[value:height + value, value:width + value] = image_bin
    return image_new

"""
图片等比变换为给定尺寸，取宽高中的大值匹配
"""
def preprocess_resize_keep_ratio(image, width, height, is_fill_bg = True):
    cur_height, cur_width = image.shape

    ratio_w = float(width)/float(cur_width)
    ratio_h = float(height)/float(cur_height)
    ratio = min(ratio_w, ratio_h)

    new_size = (min(int(cur_width*ratio), width),
                min(int(cur_height*ratio), height))

    new_size = (max(new_size[0], 1),
                max(new_size[1], 1),)

    resized_img = cv2.resize(image, new_size)
    resized_height, resized_width = resized_img.shape

    if is_fill_bg:
        norm_img = np.zeros((height, width), np.uint8)
        start_height = 0 if height - resized_height == 0 else (height - resized_height) / 2
        end_height = start_height + resized_height
        start_width = 0 if width - resized_width == 0 else (width - resized_width) / 2
        end_width = start_width + resized_width
        norm_img[start_height:end_height, start_width:end_width] = resized_img
        return norm_img
    else:
        return resized_img