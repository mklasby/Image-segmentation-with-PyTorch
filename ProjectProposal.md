# Project proposal for *Exploring Object Instance Segmentation with PyTorch*
Author: Michael Lasby

## 1. Why: Question/Topic being investigated 2pts
In this project, we will investigate the process required to use transfer learning to fine-tune an existing convolutional neural network (CNN) model for object instance segmentation. 

## 2. How: Plan of attack 2pts
The project will follow the [TorchVision Object Detection Finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). We will follow the tutorial to gain knowledge of the Object Instance Segmentation workflow in PySpark before extending the tutorial by using one of the following datasets: 

1. [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html)
2. [SpaceNet Buildings Dataset V2](https://spacenetchallenge.github.io/datasets/spacenetBuildings-V2summary.html)

The Open Images Dataset contains annotated photographs of 600 different class categories. We will select a sample of annotated data for one class to test our object instance segmentation model.  

If time permits, we will also try to use the SpaceNet Buildings Dataset to identify roof areas from the annotated satellite images. This is more of a challenge as the data is encoded with GeoJSON data and care must be taken to convert between latitude / longitude pairs and pixel data within a specific image. 

The final deliverable will be an interactive python notebook with visualizations depicting sample predictions as well as a summary of various performance metrics on the best performing model created. 

## 3. What: Dataset, models, framework, components 2pts
* [TorchVision Object Detection Finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
* [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html)
* [SpaceNet Buildings Dataset V2](https://spacenetchallenge.github.io/datasets/spacenetBuildings-V2summary.html)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [TorchVision Documentation](https://pytorch.org/docs/stable/torchvision/index.html)
* [TorchVision Segmentation Models](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation)
