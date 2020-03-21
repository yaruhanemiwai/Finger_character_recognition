#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import *
import cv2
import darknet as dn
import numpy as np
import os,glob
import time
import collections
import matplotlib.pyplot as plt

if __name__ == "__main__":

    net = dn.load_net(b"../../data/yubimoji_yolo/yolov3-tiny_test.cfg",\
        b"../../data/yubimoji_yolo/result_mo_both/yolov3-tiny_130000.weights",0)
    meta = dn.load_meta(b"../../data/yubimoji_yolo/datasets_1.data")

    f = open("../../data/yubimoji_yolo/class_test.txt","r")
    
    list_class = f.readlines()
    list_class = [x.strip("\n") for x in list_class]

    f.close()

    class_colors = []

    for i in range(len(list_class)):
        hue = 255*i / len(list_class)
        col = np.zeros((1,1,3)).astype(np.uint8)
        col[0][0][0] = hue
        col[0][0][1] = 128
        col[0][0][2] = 255
        cvcolor = cv2.cvtColor(col,cv2.COLOR_HSV2BGR)
        col = (int(cvcolor[0][0][0]),int(cvcolor[0][0][1]),int(cvcolor[0][0][2]))
        class_colors.append(col)
        
    #capture = cv2.VideoCapture(0)
    #list_folder = glob.glob("../motion_test/19/*")
    list_folder = ["../motion_test/19/1"]
    #list_folder.sort()
    
    for i in list_folder:

        list_img = glob.glob(i + "/*.bmp")

        s_0 = 0
        coord_0 = [0,0]
        coord_1 = [0,0]
        word_first = ""
        word_second = ""

        if len(list_img) == 0:
            pass
        else:
            list_img.sort()

            for num,ii in enumerate(list_img):
                img_ori = cv2.imread(ii)
                h,w,_=img_ori.shape
                img_ori = img_ori[:,(w-h)//2:(w+h)//2,:]
                #img_ori = img_ori[:,-1*h:,:]
                #img_ori = img_ori[:,:h,:]
                img = dn.nparray_to_image(img_ori)   
                r = dn.detect(net, meta, img)
                print(ii)
                #print(r)
                
                try:
                    """
                    cv2.rectangle(img_ori, (int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
                        (int(r[0][2][0]+r[0][2][2]/2), int(r[0][2][1]+r[0][2][3]/2)),\
                            class_colors[list_class.index(r[0][0])], 2)
                    """
                    cv2.rectangle(img_ori, (int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
                        (int(r[0][2][0]+r[0][2][2]/2), int(r[0][2][1]+r[0][2][3]/2)),\
                            [0,0,255], 2)

                    cv2.imwrite(ii[:-4] + "_0.bmp",img_ori)
                    #plt.imshow(img_ori)
                    #plt.show()
                
                    print(r)

                    if num != 0:

                        min_1 = min(r[0][2][2],r[0][2][3])

                        if word_first == "mo_0" and r[0][0] == "mo_1":
                        
                            print("mo")

                        if word_first == "hi":

                            if word_second == "so":

                                min_2 = min(r[0][2][2],r[0][2][3])

                                if coord_1[1] - r[0][2][1] > (min_0 + min_1 + min_2)/6 and \
                                    coord_1[0] - r[0][2][0] > (min_0 + min_1 + min_2)/6:
                                #if np.dot(- coord_1[1],np.array()) - r[0][2][1] > (min_0 + min_1 + min_2)/6 and \
                                    #coord_1[0] - r[0][2][0] > (min_0 + min_1 + min_2)/6:

                                    print("nn 右上")

                            else:
                                if r[0][2][0] - coord_0[0] > (min_0 + min_1)/4 and r[0][2][1] - coord_0[1] > (min_0 + min_1)/4:

                                    print("no 左下")

                                elif r[0][2][1] - coord_0[1] > (min_0 + min_1)/4 and abs(r[0][2][0] - coord_0[0]) < (min_0 + min_1)/4:

                                    print("- 下")

                                    word_second = r[0][0]
                                    coord_1 = [r[0][2][0],r[0][2][1]]
                                
                                elif coord_0[1] - r[0][2][1]> (min_0 + min_1)/4 and abs(r[0][2][0] - coord_0[0]) < (min_0 + min_1)/4:

                                    print("上")

                        if word_first in ["shi","wa"]:
                            if r[0][2][0] - coord_0[0] > (min_0 + min_1)/4 and r[0][2][1] - coord_0[1] > (min_0 + min_1)/4:

                                print("ri 左下")


                        if word_first == "o":

                            if r[0][2][2] * r[0][2][3] * 1.0/ s_0 < 0.8:

                                print("wo 引")

                        if word_first in ["ka","ki","ku","ke","ko"]:

                            if coord_0[0] - r[0][2][0] > (min_0 + min_1)/4 and abs(r[0][2][1] - coord_0[1]) < (min_0 + min_1)/4:

                                print("右")

                        if word_first in ["ha","hu","he","ho"]:

                            if abs(coord_0[0] - r[0][2][0]) < (min_0 + min_1)/4 and coord_0[1] - r[0][2][1] > (min_0 + min_1)/4:

                                print("上")
                                            
                        if word_first in ["ya","yu","yo","tsu","chi"]:

                            if r[0][2][2] * r[0][2][3] * 1.0/ s_0 < 0.8:

                                print("引")

                    
                    if num == 0:

                        word_first = r[0][0]

                        s_0 = r[0][2][2] * r[0][2][3]
                        min_0 = min(r[0][2][2],r[0][2][3])
                        coord_0 = [r[0][2][0],r[0][2][1]]
                except:
                    pass


    #print(list_img)
    
    """
    frame1 = frame2 = np.zeros((480, 480))
    count_static = 0
    count_move = 0
    thr_pix = 400
    thr_static = 3
    thr_move = 10
    list_word = list()
    finalword = list()
    status_static = False

    while(capture.isOpened()):
        start = time.time()
        ret, frame = capture.read()  

        img_ori = frame[:,(frame.shape[1] - frame.shape[0])//2:(frame.shape[1] + frame.shape[0])//2,:]

        frame1 = frame2
        frame2 = cv2.cvtColor(img_ori,cv2.COLOR_RGB2GRAY)

        img = dn.nparray_to_image(img_ori)   
        r = dn.detect(net, meta, img)
        img_diff = dn.frame_diff(frame1,frame2,thr = 100)

        if len(r) != 0 and np.sum(img_diff)/255 <= thr_pix and not(status_static) and not(count_static >=\
            thr_static and word_best[0][1] >= thr_static):
            list_word.append(r[0][0].decode("UTF-8"))
            count_static += 1
            list_word_new = collections.Counter(list_word)
            word_best = list_word_new.most_common()
            count_move = 0
            
            try:
                cv2.rectangle(img_ori, (int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
                    (int(r[0][2][0]+r[0][2][2]/2), int(r[0][2][1]+r[0][2][3]/2)),\
                        class_colors[dic[r[0][0].decode("UTF-8")]], 2)
                text = r[0][0].decode("UTF-8") + " " + "{:.2g}".format(r[0][1])
                cv2.rectangle(img_ori,(int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
                    (int(r[0][2][0]-r[0][2][2]/2+75), int(r[0][2][1]-r[0][2][3]/2+20)),\
                        class_colors[dic[r[0][0].decode("UTF-8")]], -1)
                cv2.putText(img_ori,text,(int(r[0][2][0]-r[0][2][2]/2),int(r[0][2][1]-r[0][2][3]/2+15)),\
                    cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,0),1)
            except:
                pass
            
     
        elif count_static >= thr_static and word_best[0][1] >= thr_static:
            
            if not(status_static):                
                status_static = True
                list_word.clear()
                print("in")
                print(word_best[0][0])
            
            if len(r) != 0
                cv2.rectangle(img_ori, (int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
                    (int(r[0][2][0]+r[0][2][2]/2), int(r[0][2][1]+r[0][2][3]/2)),\
                        class_colors[dic[word_best[0][0]]], 2)
                text = word_best[0][0] + " " + "{:.2g}".format(r[0][1])
                cv2.rectangle(img_ori,(int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)),\
                    (int(r[0][2][0]-r[0][2][2]/2+75), int(r[0][2][1]-r[0][2][3]/2+20)),\
                        class_colors[dic[word_best[0][0]]], -1)
                cv2.putText(img_ori,text,(int(r[0][2][0]-r[0][2][2]/2),int(r[0][2][1]-r[0][2][3]/2+15)),\
                    cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,0),1)
                finalcoord = [int(r[0][2][0]),int(r[0][2][1])]
            else:
                pass
            
            if not(np.sum(img_diff)/255 <= thr_pix):
                status_static = False
                count_static = 0                
                word_best.clear()
                print("out")
        
        elif not(np.sum(img_diff)/255 <= thr_pix):
            count_move += 1

        else:
            pass

        cv2.putText(img_ori,str(1/(time.time()-start)),(300,460),\
            cv2.FONT_HERSHEY_DUPLEX,2.0,(0,0,255),1)

        cv2.imshow("frame",img_ori)
        cv2.imshow("frame_diff", img_diff)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    """