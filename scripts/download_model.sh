#!/bin/bash
echo "Creating folder $1 to store model"
mkdir -p $1

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz -P $1
tar -xzf $1/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz -C $1
rm $1/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz