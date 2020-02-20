#!/usr/bin/env sh

./darknet detector train ../../dataset/yubimoji_yolo/datasets.data ../../dataset/yubimoji_yolo/yolov3.cfg 2>&1 | tee ./result/output.txt
