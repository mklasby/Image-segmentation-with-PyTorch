# ENSF 611 - Final Report: Exploring Object Instance Segmentation with PyTorch
## Submitted By: Mike Lasby
## Submitted On: Dec 9, 2020

## Requirements: 
* A significant amount of image data is required to run the `./Object_Instance_Segmentation_with_PyTorch.ipynb` notebook. Since these files are too large for GitHub, please download them from this link and extract in the root folder as `./data/`: https://www.dropbox.com/s/mq7vwkg2590jyee/data.zip?dl=0 <br>
Where possible, the notebooks will download the data directly if it is not already saved. 
* Install PyTorch & related packages: 
  * Find the right version for your system: https://pytorch.org/get-started/locally/
  * Eg. `conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch`
  * **NOTE:** An Nvidia GPU is required to take advantage of CUDA.   
* Install `openimages` using `pip install openimages`. This is a utility library for downloading specific classes from the [Google Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html). 
* Several `TorchVision` utility classes have been included in the root folder for convenience. These may be ignored for the purposes of this assignment. 


## Introduction
Implementing this project proved to be a significant challenge. As such, I have written this report in the context of my learning journey. 

## Background
PyTorch is a python-based open source deep learning framework. Designed with python and flexibility in mind, the framework is intended to leverage TPUs/GPUs to perform tensor operations on deep learning neural networks. Having had no prior introduction to deep learning or computer vision, I was perhaps somewhat ambitious in tackling this project. My primary motivation in exploring computer vision is its potential for developing transformative technologies in a myriad of industries, many of which require rapid advances to meet our climate change targets. 

For example, the buildings sector is responsible for approximately [30% of Canada's CO<sub>2</sub> emissions](https://www.canada.ca/en/environment-climate-change/services/environmental-indicators/greenhouse-gas-emissions.html). By recording image data in both the visible and infrared spectrum, we can use machine learning to improve the construction and operation of buildings. As a proof of concept, I recommend readers refer to [MyHeat.ca](https://www.myheat.ca), where aerial thermographs are being used to assist building owners in making informed thermal envelope upgrade decisions. 

Main benefits of PyTorch: 
* Imperative programming -> Computations are performed as they are defined at the cost of some performance compared to symbolic programming (eg., such as in TensorFlow). However, imperative programming is more immediately familiar to most programmers and researchers. 
* Dynamic Computation Graphing -> Graph is defined at runtime; therefore, we can use networks with variable length inputs and outputs. This also helps with debugging as the traceback will traceback to the exact line where the exception was generated. Having just spent many weeks learning PySpark, this was a particularly welcome change. 
* Autograd -> The PySpark library uses the `autograd` package to perform differentiation on any operation performed on a tensor. This is where the `define-by-run` magic comes in to redefine the back propagation function during each iteration. We can take advantage of this functionality by setting the `.requires_grad` attribute to `True` on any tensor to track all differential operations applied to it. 

Fundamentally, neural networks are used to transform an input tensor into some desired output tensor. For the basic multi-layer perceptron network, each neuron can be considered as a single node with a numeric activation value. For example, the MNIST dataset contains low resolution greyscale images of hand-written numbers. The numbers are encoded as 28x28 pixel images; therefore, the input layer would have 28x28 = 784 neurons. Since we only have one colour channel in this dataset, each neuron would represent the apparent tone of a single pixel (eg., 1.0 for white, 0.0 for black). 

One or more *hidden layer* of neurons may be added between the input and output tensors. Each successive layer of neurons influences the activation of neurons in the next layer. Neurons are typically connected to many neurons in the subsequent layer. These connections are weighted to determine the respective influence of each preceding neuron activation on the activations of the next layer. So, we can surmise that the activation of any particular neuron is the weighted sum of the activations of the neurons in the preceding layer, plus or minus some bias. Typically, the activation values are normalized using a sigmoid of ReLU (rectified linear unit) function. 

During training, successive iterations are run with labelled data. A loss function, for example log loss or cross-entropy, can be defined which determines how well a given iteration has performed on the input data. By employing gradient descent, the model will tweak the weight and bias parameters by a small amount, known as the learning rate, to incrementally improve the model performance. The input dataset may be learned on more than once, in successive iterations known as epochs.  

These networks are capable of producing remarkably complex and accurate models. In fact, the [Universal Approximation Theorem](http://neuralnetworksanddeeplearning.com/chap4.html) proves that a sufficiently complex neural network can be used to represent *any* function. 

Convolutional neural networks are particularly well suited for image classification and segmentation. In this report, we will investigate the process required to use transfer learning to fine-tune an existing convolutional neural network (CNN) model for object instance segmentation of a previously unknown class. 

## Approach 
To learn PyTorch, I followed the first 5 hours of the free tutorial from Free Code Camp[<sup>1</sup>](https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=11944s&ab_channel=freeCodeCamp.org). This was a great introduction to the library as we implemented some familiar models within a neural net context. 

See `/PyTorch_Zero-to_GANs/` for the following tutorial workbooks: 
* `/PyTorch_Zero-to_GANs/LinearRegression.ipynb`
* `/PyTorch_Zero-to_GANs/LogisticRegression.ipynb`
* `/PyTorch_Zero-to_GANs/FeedForward.ipynb`

### Linear Regression: 
We implemented a regression model from scratch. It was comforting to see the classic model form `y = w * x + b`. We defined a custom MSE loss function. The gradient decent (d<sub>loss</sub>/d<sub>w</sub> & d<sub>loss</sub>/d<sub>b</sub>) was easily computed in PyTorch by calling `w.grad` and `b.grad`. We multiplied these gradients by a small learning rate to move the model toward the minimum loss value for each parameter. Of course, PyTorch also has a stochastic gradient descent optimizer available that we implemented in a neat and tidy function to demonstrate the built-in function doing the same optimization. 

### Logistic Regression 
The class MNIST dataset was used to explore how PyTorch works with image data. One of the more enticing features of PyTorch is its well thought out OOP design choices. Extending a model or dataset class is encouraged as the de facto way to deploy models while avoiding *most* of the boilerplate code. We extended the [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class, the base class for all neural network models. All subclasses of nn.Module must implement the inherited method `foward()` to define the computation to be performed on each iteration. We also defined some custom steps for training, validation and an epoch end report function. The model achieved a modest accuracy of around 83% by the end of 8 epochs. 

### Feed Forward Neural Network
Again returning to the MNIST dataset, we improved on the logistic regression model by building a feed forward neural net. This is a relatively simple network and was one of the first neural networks designed. Specifically, we designed a two-layer network with a 32 node hidden layer. We used the ReLU activation function to only allow a given node to be active if its activation value is > 0. This was a great way to visualize how the image data is transformed from an an image into the input array by unfurling the pixel. The output layer in our case is 10 nodes, one for each target class. The model did much better than the logistic regressor, achieving a score of 96%. I was surprised to see how quickly the model achieved this score, with the vast majority of loss eliminated within the first epoch. I was also astonished at the number of connections between nodes, even on such a small input dataset. In total, we had 784x32 + 32*10 =25,408 weights in the network. To attempt to work with such higher resolution images, we would need to move to convolutional neural networks.  

## Results
Following the PyTorch Image Segmentation Tutorial[<sup>2</sup>](https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=bpqN9t1u7B2J), I attempted to extend this tutorial by adapting it to a new dataset from the Google Open Images Dataset[<sup>3</sup>](https://storage.googleapis.com/openimages/web/index.html). I had originally hoped to use building rooftops as the subject, but settled for Dogs since there is much more readily available annotated data. As Dr. Pauchard says, we do our best work when we are joyful, so I figured I should look at images of dogs while attempting this challenge. 

See `./Object_Instance_Segmentation_with_PyTorch.ipynb`. 

### Data Collection: 
The Google Open Images Dataset contains more than 2.7 million instance segmentations on 350 categories. Unfortunately, the dataset lacks utilities tools to download specific classes. A variety of utility libraries have been developed by third parties to improve the import experience. I used the `openimages` library; however, it does not import the data object segmentation masks, so those must be imported directly from the website and filtered. It is my intent that you, dear reader, may avoid this experience by simply extracting the `/data` folder included with the dropbox link above. 

The first task was to find the masks for my images. I downloaded 500 Dog images and using their IDs, found the available object segmentation masks. Only a fraction of the total Open Images Dataset has been annotated with segmentations masks, so only about 10% of the originally imported images proved to be useable. 

The mask and image files were processed and any unmatched files were deleted. Many images include multiple instances of the dog class, but these were stored in separate mask .png files. As such, I simply copied the image files to ensure that each mask had a correct corresponding images. A more challenging but accurate method is to combine the masks, since the method described below is capable of segmenting multiple instances and multiple classes per image. 

The dataset was wrapped into a custom Dataset class by extending `torch.utils.data.Dataset`. See the notebook for details. In short, the data is processed into an dictionary containing a variety of tensors which describe various features present in both the training image and the corresponding mask. 

### Model Selection 
Classic multilayer perceptron models are unable to effectively account for pixel proximity in comparison to convolutional neural networks (CNN). CNNs were developed to tackle computer vision problems with improved performance compared to traditional networks. CNNs depend on down sampling the image data by computing feature maps whose nodes depend on multiple input nodes. Ie., we reduce the dimensionality of the input tensor from each convolutional layer. Due to these characteristics, CNNs are currently the best performing models for image classification and object segmentation tasks in computer vision. 

Due to the time required to train a CNN model, it was recommended to use a pre-trained model. `TorchVision` has a few models available that are suitable for image segmentation tasks, we used the [faster RCNN model](https://arxiv.org/abs/1506.01497) for this project.  

### Poor Results  
Unfortunately, I was unable to train a reasonable model in this task. The resulting CNN model predicted several dogs per image with very low confidence. The resulting masks were always rectangular with several overlapping masks. During training epochs, the loss function metrics were reducing between iterations, but the average precision calculated by average precision calculated by Intersection-over-Union remained at 0.0 in all cases. 

After spending a significant amount of time bug hunting, I was unable to solve the problem. I suspect that the input masks are not being processed properly by the model. The tutorial uses a colour mask whereas the dataset I imported uses a black and white mask. Further, the masks should be combined where multiple dog instances appear on the same image. 

## Interpretation
Ultimately, I learned a lot about the PyTorch library but fell short of my goal is tuning an existing model via transfer learning. More work is required to achieve satisfactory results in this task. 

My next steps are to re attempt the problem with one of the many datasets natively available within the PyTorch package. This will help me assess the sensitivity of the pre-trained PyTorch models on the format of the input data. I was able to correctly follow the PyTorch tutorial using the recommended data, so I suspect I have missed an important step in preparing my custom dataset. Nevertheless, I feel that I know much more about deep learning and computer vision than previously and have a greater appreciation for what's happening under the hood in supervised and unsupervised learning tasks.

## Reflection & Conclusion  
This was a very positive learning experience. While I failed to achieve my initial goal, I believe I am well on my way to achieving the proposed task. I ended up deviating greatly from my proposal once I had a closer look at PyTorch and realized its complexity. I suspect it will take months to learn this framework rather than a few weeks. 

I found that many aspects of this challenge were surprisingly tough. For instance, just getting CUDA up and running or finding good quality images with masks to train on. Not to mention trying to wrap my head around deep learning. In total, I spent about 20 hours learning and experimenting. I am excited to learn more about the computer vision field and continue exploring with PyTorch. 

Throughout this experience, I greatly benefited from the knowledged imparted in 611. I am frustrated by my lack of quality results, but hope that the effort I put in to this project is apparent through this summary report and the enclosed notebooks. 

Shoot for the moon and land among the stars as they say. 

Sincerely,<br>
-Mike Lasby

## References
1. [PyTorch for Deep Learning - Free Code Camp](https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=11944s&ab_channel=freeCodeCamp.org)
2. [PyTorch Fine Tuning Instance Segmentation](https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=bpqN9t1u7B2J)
3. [Google Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)


