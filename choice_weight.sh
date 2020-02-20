#!usr/bin/bash sh

for i in `seq 0 1`
do
python python/classify.py $i
done
