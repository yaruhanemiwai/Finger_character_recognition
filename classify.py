#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ctypes import *
import math
import random
import glob
import os
import gc
import sys
args = sys.argv

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def nparray_to_image(img):

    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image
    #yield image

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im, thresh=.1, hier_thresh=.5, nms=.45):
    im = load_image(im, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    #free_detections(dets, num)
    #free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]

    f = open("../../data/yubimoji_yolo/class.txt","r")
    img_dir = "../caffenet/data/img_new/"
    weight_dir = "../../data/yubimoji_yolo/result_" + str(args[1]) + "/"
    #weight_dir = "../../dataset/yubimoji_yolo/backup/"
    #weight_dir = "../../dataset/yubimoji_yolo/result_" + str(2) + "/"
    class_num = range(44)
    img_list = list()
    #data_dir = 
    line = f.readline()
    count = 0
    dic = {}
    weight_max = ""
    while line:
        line = line[:-1]
        #line.rstrip()        
        dic[line] = count
        #print(line," ",len(line),"\n")
        #print(line)
        line = f.readline()
        count += 1
    f.close()
    #for i in os.listdir()
    #print(dic)
    for i in class_num:
        if i != 43:
            cross_dir = img_dir + str(i) + "/4/*"
            img_list.extend(glob.glob(cross_dir))
        else:
            cross_dir = img_dir + str(i) + "/2/*"
            img_list.extend(glob.glob(cross_dir))
    #print(img_list)
    count_max = 1
    count_rate_max = 0
                        
    for i in os.listdir(weight_dir):
        if i[-6:] == "backup":
            pass
        else:
            count = 0
            count_rate = 0
            meta = load_meta("../../data/yubimoji_yolo/datasets_resize.data")
            net = load_net("../../data/yubimoji_yolo/yolov3-tiny_test_resize.cfg",weight_dir + i, 0)
            for ii in img_list:    
                r = detect(net, meta, ii)
                
                #print(dic[r[0][0]]," ",ii[21:23].rstrip("/"))
                if str(len(r)) == str(0) and str(ii[25:27].rstrip("/")) == str(43):
                    count += 1
                    count_rate += float(1)
                elif str(len(r)) == str(0):
                    pass
                elif str(dic[r[0][0]]) == str(ii[25:27].rstrip("/")):
                    count += 1
                    count_rate += float(r[0][1])
                else:
                    pass
            #print(str(count))
            if count > count_max and count_rate*1.0/count > count_rate_max*1.0/count_max:
                weight_max = weight_dir + i
                count_max = count
                count_rate_max = count_rate
                
            print(weight_max)
            #del meta
            #del net
            #gc.collect()
    #print(weight_max)
    
    count = 0          
    count_rate = 0  
    f = open("../../data/yubimoji_yolo/result_" + str(args[1]) + ".txt","a+")
    #f = open("../../dataset/yubimoji_yolo/result_new.txt","a+")
    #f = open("../../dataset/yubimoji_yolo/result_" + str(0) + ".txt","a+")
    meta = load_meta("../../data/yubimoji_yolo/datasets_resize.data")            
    net = load_net("../../data/yubimoji_yolo/yolov3-tiny_test_resize.cfg",weight_max,0)
    #net = load_net("../../dataset/yubimoji_yolo/yolov3-tiny_test.cfg",weight_dir + "yolov3-tiny_20000.weights",0)
    for i in img_list:    
        r = detect(net, meta, i)
        
        if str(len(r)) == str(0) and str(i[25:27]) == str(42):
            count += 1
            count_rate += float(1)
            f.write(str(i) + " " + str(r) + "\n")
            
        elif str(len(r)) == str(0):
            f.write(str(i) + " " + str(r) + "\n")
            
        elif str(dic[r[0][0]]) == str(i[25:27].rstrip("/")):
            count += 1
            count_rate += float(r[0][1])
            f.write(str(i) + " " + str(r[0][0]) + " " + str(r[0][1:]) + "\n")

        else:
            f.write(str(i) + " " + str(r[0][0]) + " " + str(r[0][1:]) + "\n")   
        
    f.write(str(weight_max) + " score: " + str(count*1.0/len(img_list)) + " rate_score: " + str(count_rate*1.0/count))
    f.close()
    #print(r)
    #print(dic[r[0][0]])
    #print(r[0][0].decode('utf-8'))
    #r[0] = r[0].decode('utf-8')
    #print r[0][0],r[0][1:]
