# Object Detection in an Urban Environment

This repository holds the code for Udacity's Self Driving Car Engineer Nanodegree Computer Vision project - Object Detection in an Urban Environment. It details the steps in retraining a SSD ResNet50 model on the open source Waymo camera dataset to perform object detection on a typical road scene, detecting 3 classes of objects: vehicles, pedestrains and cyclists. 

## Prerequisites

### Local docker setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

### Download the Waymo dataset

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records).

```
cd src/data_processing
python3 download_process.py --data_dir <folder_to_save_dataset> --size <number of files you want to download>
```

An example command is `python3 download_process.py --data_dir /app/project/data --size 100`

The time taken to process 100 files is around 3 hours.

### Exploratory Data Analysis

This is the most important task of any machine learning project. After the data is downloaded to the folder specified in the previous step, we will explore and visualise a sample of the dataset to have a sense of what the data looks like.

To do so, open the `Exploratory Data Analysis` notebook and run the cells to get some visualisation charts and figures. An example is shown below.

![Initial data visualisation](/assets/initial_data_exploratory.png "Initial data visualisation")

From the exploratory dataset, we can immediately note down 2 things. First, there are many more cars compared to pedestrains and cyclists. Pedestrains and cyclists will form our subset of "rare classes" that we need to pay closer attention to. Second, the lighting and weather conditions vary across datasets. We need to ensure that there is proper representation of images with good and bad lighting, as well as good and adverse weather conditions.

For additional exploratory data analysis, we will check the spread of classes across each image to see how heavily the number of cars skew the dataset. We will also need to see check the distribution of lighting conditions. For that, we will find the average pixel value and covert to perceived brightness for each image.

#### Class distribution

![Initial data visualisation](/assets/cls_distribution.png "Initial data visualisation")

As "cyclists" are the rare class in this dataset, we cannot use a random split as this class might be underpresented in the training dataset and perform very poorly on the test and evaluation dataset. We need to proactively sieve out the tfrecords with cyclists and split them such that they are representative of the dataset.

#### Brightness distribution

![Initial data visualisation](/assets/brightness_distribution.png "Initial data visualisation")

From the data above, there are many more cars than other classes in the dataset, and the brightness values are skewed towards the brighter area. In the data augmentation step, these issues need to be addressed.

### Split the dataset

The dataset is split randomly in the ratio 80% for training, 20% for validation.

The three subfolders are:

* /app/project/data/processed/train

* /app/project/data/processed/val

* /app/project/data/processed/test

```
cd src/data_processing
python3 create_splits.py --source <source_folder_of_tfrecords> --destination <destination_folder_of_tfrecords>
```

An example command is `python3 create_splits.py --source /app/project/data/processed --destination /app/project/data/processed`

## Model

### Download base model

The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz).

```
chmod +x download_model.sh
cd scripts
./download_model.sh /app/project/src/experiments/pretrained_model
```

### Experiments folder structure
The experiments folder will be organized as follow:

```
experiments/
    - experiment_0/ - create a new folder for each experiment you run
    - experiment_1/ - create a new folder for each experiment you run
    - pretrained_model/
    - reference - reference training with the unchanged config file
    - edit_config.py - edit pipeline.config
    - exporter_main_v2.py - to create an inference model
    - label_map.pbtxt - label to class mapping file
    - model_main_tf2.py - to launch training
```

### Creating a new config file for training

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:

```
cd experiments
python3 edit_config.py \
--train_dir /app/project/data/processed/train \
--eval_dir /app/project/data/processed/val \
--batch_size 4 \
--checkpoint /app/project/src/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 \
--label_map /app/project/src/experiments/label_map.pbtxt
```

A new config file `pipeline_new.config` will be created, which will be put in a newly created `experiment_N` folder, where N is the experiment number.

## Training

### Start training

After the new config file is put to a new experiments folder, the training can begin with the below command.

```
cd src/experiments
python3 model_main_tf2.py \ 
--model_dir=<path_to_experiment_folder>/ \
--pipeline_config_path=<path_to_experiment_folder>/pipeline_new.config
```

Example command:

```
python3 model_main_tf2.py \
--model_dir=experiment_0/ \
--pipeline_config_path=experiment_0/pipeline_new.config
```

### Viewing Tensorboard

We can visualise the loss and learning rate trends by using Tensorboard.

```
tensorboard --logdir <path_to_experiment_folder>
```

Example command:

```
tensorboard --logdir experiment_0/train
```

### Evaluation

Once the training is finished, launch the evaluation process:

```
python3 model_main_tf2.py \
--model_dir=<path_to_experiment_folder>/ \
--pipeline_config_path=<path_to_experiment_folder>/pipeline_new.config \
--checkpoint_dir=<path_to_experiment_folder>/
```

Example command: 

```
python3 model_main_tf2.py \
--model_dir=experiment_0/ \
--pipeline_config_path=experiment_0/pipeline_new.config \
--checkpoint_dir=experiment_0/
```

### Creating an animation

#### Export the trained model

Modify the arguments of the following function to adjust it to your models:

```
python3 exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path <path_to_experiment_folder>/pipeline_new.config \
--trained_checkpoint_dir <path_to_experiment_folder>/ \
--output_directory <path_to_experiment_folder>/exported/
```

Example command:

```
python3 exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path experiment_0/pipeline_new.config \
--trained_checkpoint_dir experiment_0/ \
--output_directory experiment_0/exported/
```

#### Creating a gif

```
python3 inference_video.py \
--labelmap_path ../experiments/label_map.pbtxt \
--model_path ../experiments/<path_to_experiment_folder>/exported/saved_model \
--tf_record_path ../../data/processed/test/<tfrecord_file> \
--config_path ../experiments/<path_to_experiment_folder>/pipeline_new.config \
--output_path animation.gif
```

Example command:

```
python3 inference_video.py \
--labelmap_path ../experiments/label_map.pbtxt \
--model_path ../experiments/experiment_0/exported/saved_model \
--tf_record_path ../../data/processed/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord \
--config_path ../experiments/experiment_0/pipeline_new.config \
--output_path animation.gif
```

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.



## Results of reference pipeline

### Tensorboard

![loss](/assets/experiment_0/loss.jpeg "loss")

### Evaluation

![map](/assets/experiment_0/map.jpeg "map")

### Animation

![Test data 0](/assets/experiment_0/animation.gif "Test data 0")

### Discussion

Using the default pipeline config file, we can begin our first training job. From the Tensorboard graphs, the losses (classification, localization and regularization) can all be seen to decrease with the number of epoches. The final overall loss is a little lower than 15 after 10,000 epoches.

Looking at the evaluation numbers, the average precision and average recall are all extremely low (around < 0.25) at all area and IoU values. This means that the model does not generalise well to new data, and the training job has lots of room of improvement. The performance can be seen in the animation above, where none of the vehicles are detected (detection threshold set at 0.5). 

### Improvements to next iteration

It is clear that the default values in the pipeline config has room for improvement. From the exploratory data analysis above, we can see class imbalances (many more vehicles than cyclists) and inconsistent image data.

Regarding the class imbalance, we have already taken steps to ensure that the "cyclist" class is properly represented in the training/evaluation datasets when doing the data splits.

For inconsistent image data, we can make use of Tensorflow image augmentations to increase the quantity and quality of training/evaluation image data. From the exploratory data analysis above, we can see that is significant variation in image contrast and brightness, as well as the scale of detected classes. Hence, we can use the following image augmentations to improve our data representation:

* random_adjust_brightness

* random_image_scale

* random_adjust_hue

* random_adjust_contrast

* random_adjust_saturation

* random_distort_color

The new augmentations can be visualized in the notebook `Explore augmentations.ipynb`.

The number of steps will also be increased from 10,000 to 25,000 to allow the loss to converge more. Other modifications include increasing batch size from 2 to 4, increasing max total detections from 100 to 200 in non max supression.

## Results of improved pipeline

### Tensorboard

![loss](/assets/experiment_1/loss.jpeg "loss")

### Evaluation

![map](/assets/experiment_1/map.jpeg "map")

### Animation

![Test data 0](/assets/experiment_1/animation.gif "Test data 0")

### Discussion

From the Tensorboard graphs, the final total loss improved from around 15 to around 0.7. This can be reflected in the higher average recall and average precision values. However, the values for small areas remain relatively low compared to larges areas. The animation shows the improvements in detections, and shows the deficiencies in detecting very small objects.

### Further work

The number of training steps could be increased futher for even lower losses since it has not yet fully converged. To further improve the model, we could also differentiate the training/evaluation dataset by size of detected classes since there is a marked difference in model performance between small and large areas.
