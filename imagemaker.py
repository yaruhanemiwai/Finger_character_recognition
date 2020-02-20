#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import cv2
import sys,os
from tqdm import tqdm
sys.path.append('/home/es1video4/workspace/iwamine/module/')
import copy
import random
import shutil
import matplotlib.pyplot as plt
from myfunc import *

def blur_gaussian_randomly(img,value):

    rand_rate = random.randint(0,value//2)    
    if (rand_rate == int(0)):                    
        pass
    else:
        img = cv2.GaussianBlur(img,(2*rand_rate+1,2*rand_rate+1),0)
        #img = cv2.fastNlMeansDenoisingColored(img.astype(np.uint8),None,rand_rate,10,7,21)

    return img

def rotation_randomly(img,seg,value,list_angle):

    if (random.randint(0,value-1) == 0):
        list_angle = [random.choice(list_angle)]
        img = rotation_usual(img,list_angle)
        seg = rotation_usual(seg,list_angle)
    
    return img,seg

def control_hsv(img,saturation,brightness):
    #input:bgr return:bgr
    img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2HSV) 
    img = img.astype(np.float32)
    rand_s = random.uniform(saturation[0],saturation[1])
    rand_b = random.uniform(brightness[0],brightness[1])
    img[:,:,1] = np.clip(rand_s * img[:,:,1],0,255)
    img[:,:,2] = np.clip(rand_b * img[:,:,2],0,255)
    img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_HSV2BGR)

    return img

def binarizing(seg):

    label = np.where(seg[:,:,0],0,1)    
    label_1 = np.where(seg[:,:,2]==255,1,0)
    label_2 = np.where(seg[:,:,1],0,1)
    label = (label * label_1 * label_2).astype(np.float32)
    
    return label

def get_back(img_h,img_w):
    #return all background data(type:array)

    back_dir_yes = "./data/20191218/0/0/*.bmp"
    back_dir_no = "./data/20191218/0/1/*.bmp"
     
    back_list_yes = glob.glob(back_dir_yes)
    back_list_no = glob.glob(back_dir_no)

    back_no = len(back_list_yes)
    back_list_yes.extend(back_list_no)
    back_list = back_list_yes

    back_all = np.zeros((len(back_list),img_h,img_w,3)).astype(np.uint8)

    for ii,iii in enumerate(back_list):
        back_all[ii] = cv2.resize(cv2.imread(str(iii)),(img_w,img_h))

    return back_all,back_no

def crop_considered(img,seg,a,b,c,d):
    #(a,b,c,d)=(left,right,top,bottom)
    #crop that including all hand regions
    h, w, _ = img.shape
    count = 0

    if a < 0:
        a = 0
    if b > w:
        b = w
    if c < 0:
        c = 0
    if d > h:
        d = h

    while(1):
    
        if b - a < d - c:
            rand = random.randint(d - c, h)
        else:
            rand = random.randint(b - a, h)
            
        if b - rand < 0:
            b = rand
        if d - rand < 0:
            d = rand

        left = np.random.randint(b - rand, a + 1)
        top = np.random.randint(d - rand, c + 1)
        
        count += 1
        
        if count == 10:
            if top > h - rand:
                top = h - rand
            if left > w - rand:
                left = w - rand
            break
        
        if top + rand < h and left + rand < w:
            break            
    
    bottom = top + rand
    right = left + rand

    img = img[top:bottom, left:right, :]
    seg = seg[top:bottom, left:right] 
    
    return img,seg
    
def hamerun(hand,seg,back):
    
    while(1):
        rand = random.uniform(0.7,1.3)
        hand_1 = cv2.resize(hand.copy(),None,fx=rand,fy=rand)
        if hand_1.shape[0] < back.shape[0] and hand_1.shape[1] < back.shape[0]:
            break
    
    hand = cv2.resize(hand,None,fx=rand,fy=rand)
    seg = cv2.resize(seg,None,fx=rand,fy=rand,interpolation=cv2.INTER_NEAREST)
    a1 = np.zeros(back.shape[0:2])

    img_0 = seg.copy()
    img_1 = cv2.bitwise_not((seg.copy()*255).astype(np.uint8))/255

    top = np.random.randint(0*back.shape[0], back.shape[0]-hand.shape[0])
    bottom = top + hand.shape[0]
    left = np.random.randint(0*back.shape[1], back.shape[1]-hand.shape[1])    
    right = left + hand.shape[1]

    back[top:bottom, left:right] = \
        hand * np.tile(img_0,(3,1,1)).transpose((1,2,0)) + \
            back[top:bottom,left:right] * np.tile(img_1,(3,1,1)).transpose((1,2,0))

    a1[top:bottom, left:right] = seg
    
    return back,a1

def main():
    #Setting
    img_num = 10
    
    dir_in = './data/20191218/'
    dir_out = "./data/proposal/"
    size_input = 256
    num_class = range(1,21)
    num_type = range(6)
    
    img_h = 480
    img_w = 640

    back_all,num_no = get_back(img_h,img_w)

    list_angle = range(-15,16)
    saturation = (0.7,1.2)
    brightness = (0.7,1.2)
        
    for i in num_class:
        
        for ii in num_type:

            for iii in os.listdir(os.path.join(dir_in,str(i),str(ii),"0")):

                if not(os.path.exists(os.path.join(dir_out,str(i-1),str(ii),str(iii[:-4]).zfill(5),"0"))):
                    os.makedirs(os.path.join(dir_out,str(i-1),str(ii),str(iii[:-4]).zfill(5),"0"))
                    os.makedirs(os.path.join(dir_out,str(i-1),str(ii),str(iii[:-4]).zfill(5),"1"))
            
            for iii in os.listdir(os.path.join(dir_in,str(i),str(ii),"0")):
                print(iii)
                img_ori = cv2.resize(cv2.imread(os.path.join(dir_in,str(i),str(ii),"0",str(iii))),(img_w,img_h))            
                seg = cv2.resize(cv2.imread(os.path.join(dir_in,str(i),str(ii),"1",str(iii))),(img_w,img_h),\
                    interpolation=cv2.INTER_NEAREST)
                seg_binarized = binarizing(seg)
                seg_binarized = closing(seg_binarized*255)//255
                seg_ori = seg_binarized
                       
                for inum in tqdm(range(img_num)):
                            
                    rand_num = random.randint(0,3)
                    img = img_ori.copy()
                    seg = seg_ori.copy()
                    
                    if rand_num == 0:
                                            
                        img_hand = (np.tile(seg,(3,1,1)).transpose((1,2,0)) * img)
                        img_back = (np.tile(np.where(seg == 1,0,1),(3,1,1)).transpose((1,2,0)) * img)
                        img_hand = control_hsv(img_hand,saturation,brightness)
                        img_back = control_hsv(img_back,saturation,brightness)

                        img = (img_hand + img_back).astype(np.uint8)                       
                        
                        img,seg = rotation_randomly(img,seg,2,list_angle)
                        
                        img = blur_gaussian_randomly(img,7)
                    
                        x,y,w,h = cv2.boundingRect(seg)
                        #(img,seg,left,right,top,bottom)
                        img,seg = crop_considered(img,seg,x-3,x+w+3,y-3,y+h+3)
                    
                    else:
                       
                        hand = np.zeros(img.shape).astype(np.float32)  
                    
                        hand = img * np.tile(seg,(3,1,1)).transpose((1,2,0))
                        
                        rand_back = random.randint(0,back_all.shape[0]-1)
                        
                        back = back_all[rand_back].copy()

                        hand = control_hsv(hand,saturation,brightness)

                        back = control_hsv(back,saturation,brightness)
                                        
                        if random.randint(0,1) == 1:
                            back = horizontal_flip(back)
                            
                        if random.randint(0,1) == 1 and rand_back < num_no - 1:
                            back = vertical_flip(back)
                            
                        """
                        if int(dir_new) == int(25):
                            #angle = range(-20,60,10)
                            angle = range(-20,60)
                        if int(dir_new) == int(14):
                            #angle = range(-50,40,10)
                            angle = range(-50,40)
                        if int(dir_new) == int(21):
                            #angle = range(-50,0,10)
                            angle = range(-50,0)                    
                        """

                        back = rotation_usual(back,list_angle)
                        
                        hand,seg = rotation_randomly(hand,seg,2,list_angle)
                        
                        #(left,top,width,height)
                        a,b,c,d = cv2.boundingRect(seg)

                        hand = hand[b:b+d,a:a+c,:]

                        seg = seg[b:b+d,a:a+c]

                        hand,seg = hamerun(hand,seg,back)

                        hand = blur_gaussian_randomly(hand,7)

                        x,y,w,h = cv2.boundingRect(seg.astype(np.uint8))
                        #(img,seg,left,right,top,bottom)
                        img,seg = crop_considered(hand,seg,x-3,x+w+3,y-3,y+h+3)
                     
                    cv2.imwrite(os.path.join(dir_out,str(i-1),str(ii),str(iii[:-4]).zfill(5),"0",str(inum).zfill(2)+".bmp"),\
                        cv2.resize(img,(size_input,size_input)))
                    seg = cv2.resize(seg,(size_input,size_input),interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(dir_out,str(i-1),str(ii),str(iii[:-4]).zfill(5),"1",str(inum).zfill(2)+".bmp"),\
                        np.where(seg>0,255,0))
            
if __name__ == '__main__':
    main()
