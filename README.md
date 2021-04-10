# Automated Tensorflow Object Detection
# Default model used:
<a href=http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz>ssd_resnet50_v1_fpn_640x640_coco17_tpu-8</a>

## Steps to train:
1. Generate annotation files using LabelIMG in pascal VOC format and place them in ds directory(sample images and respective xml are provided)
2. Edit label_map.pbtxt according to the number of categories you want the model to be trained on
3. In pipeline.config edit
   1. <a href=https://github.com/spsc0894/automated_tfod/blob/main/pipeline.config#L3>num_classes</a>
   2. you may also change the batch size: <a href=https://github.com/spsc0894/automated_tfod/blob/main/pipeline.config#L131>batch size</a>
4. Edit test_sample.py as per your requirement.


## Run setup:
''
make setup
''

## Run dockerfile to start training
''
make run_tfod
''
