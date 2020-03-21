#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import *
import cv2
import numpy as np
import os,glob,sys
import darknet as dn
#import time
#import collections
import matplotlib.pyplot as plt

def main():

    path_img = sys.argv[1]
    #path_img = "../sample/ku.bmp"

    if sys.version_info[0] == 2:
        net = dn.load_net("../model/yolov3-tiny_test.cfg",\
            "../model/yolov3-tiny_130000.weights",0)
        meta = dn.load_meta("../model/datasets.data")
    else:
        net = dn.load_net(b"../model/yolov3-tiny_test.cfg",\
            b"../model/yolov3-tiny_130000.weights",0)
        meta = dn.load_meta(b"../model/datasets.data")

    f = open("../model/class.txt","r")
    
    list_class = f.readlines()
    list_class = [x.strip("\n") for x in list_class]

    f.close()

    class_colors = dn.get_colors(list_class)

    img = cv2.imread(path_img)
    h,w,_= img.shape
    img = img[:,(w-h)//2:(w+h)//2,:]
    img = cv2.resize(img,(416,416))
    img_ = dn.nparray_to_image(img)   
    r = dn.detect(net, meta, img_)

    try:

        cv2.rectangle(img, (int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
            (int(r[0][2][0]+r[0][2][2]/2), int(r[0][2][1]+r[0][2][3]/2)),\
                class_colors[list_class.index(r[0][0].decode())], 2)
        cv2.rectangle(img,(int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
            (int(r[0][2][0]-r[0][2][2]/2+75), int(r[0][2][1]-r[0][2][3]/2+20)),\
                class_colors[list_class.index(r[0][0].decode())], -1)
        cv2.putText(img,r[0][0].decode(),(int(r[0][2][0]-r[0][2][2]/2+30),int(r[0][2][1]-r[0][2][3]/2+15)),\
            cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,0),1)
            
        #cv2.imwrite(path_img[:-4] + "_result.bmp",img)
        plt.imshow(img)
        plt.show()

    except:
        print("There is no finger characters!!")

if __name__ == "__main__":
    main()
