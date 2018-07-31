# **Behavioral Cloning**

## Overview


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidiaNN.png "Model Visualization"
[image2]: ./examples/center.jpg "Normal Image"
[image3]: ./examples/recovery1.jpg "Recovery Image 1"
[image4]: ./examples/recovery2.jpg "Recovery Image 2"
[image5]: ./examples/recovery3.jpg "Recovery Image 3"
[image6]: ./examples/recovery4.jpg "Recovery Image 4"
[image7]: ./examples/recovery5.jpg "Recovery Image 5"
[image8]: ./examples/recovery6.jpg "Recovery Image 6"
[image9]: ./examples/recovery7.jpg "Recovery Image 7"


### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files and repository

#### 1. Files include all required ones to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md and README.md summarizing the results

#### 2. Files include functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For this project I am using the Nvidia neural network to train the model. It has 5 convolutional layers and 4 fully connected layers. It starts with three 5x5 convolution layers, followed by two 3x3 convolution layers, then four fully connected layers (model.py lines 82-99)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 80).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 93,95,97).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 104-108). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving smoothly on sharp curves. Also I am using the images and measurements data from all 3 cameras, i.e. the center, left, right cameras. For the left and right measurements, I give it a correction of steering angle @0.25.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create the model, then train and validate the model, then run the vehicle autonomously and observe and analyze the behavior, then come back to modify the model and repeat the steps until it drives well autonomously.

My first step was to use a convolution neural network model similar to the nvidia neural network, I thought this model might be appropriate because Nvidia is using this network for real driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I recorded more data, added Dropout layers in my neural network, decreased my epochs from 10 to 3.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like when it turns sharply. To improve the driving behavior in these cases, I recorded more curve turning data to make the model learn the behavior.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 79-99) consisted of a convolution neural network with the following layers and layer sizes.

<<<<<<< HEAD
* normalization
* 5x5 convolution layer, depth 3
* 5x5 convolution layer, depth 24
* 5x5 convolution layer, depth 36
* 3x3 convolution layer, depth 48
* 3x3 convolution layer, depth 64
* Flatten layer, 1164 neurons
* Fully connected layer, 100 neurons
* Fully connected layer, 50 neurons
* Fully connected layer, 10 neurons
* Fully connected layer, 1 neurons
=======
#normalization
#5x5 convolution layer, depth 3
#5x5 convolution layer, depth 24
#5x5 convolution layer, depth 36
#3x3 convolution layer, depth 48
#3x3 convolution layer, depth 64
#Flatten layer, 1164 neurons
#Fully connected layer, 100 neurons
#Fully connected layer, 50 neurons
#Fully connected layer, 10 neurons
#Fully connected layer, 1 neurons
>>>>>>> abeb9ecfaf539e7bddbf6e2e8efcdbd1772476a6

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from side of the road back to the center when it drives itself. Below images show what a recovery looks like:

![alt text][image3]
<<<<<<< HEAD

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

=======
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
>>>>>>> abeb9ecfaf539e7bddbf6e2e8efcdbd1772476a6
![alt text][image9]


After the collection process, I had 55947 number of data points. I then preprocessed this data by cropping the top and bottom part, with only the interest road section left, then applied GaussianBlur, then resize it to match the nvidia neural network input size. Also change the colorspace from BGR to YUV.

Because the data size is so big, I created a generator which is much more memory-efficient. The generator is a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, a generator pulls pieces of the data and process them on the fly only when they are needed.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the decreasing loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
