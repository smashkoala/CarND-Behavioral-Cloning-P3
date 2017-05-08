#**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./network.png "Model Visualization"
[image3]: ./bridge.jpg "Recovery Image"
[image4]: ./curve1.jpg "Recovery Image"
[image5]: ./curve2.jpg "Recovery Image"
[image6]: ./flip_left.jpg "Normal Image"
[image7]: ./flip_right.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

The files except model.h5 can be also found from the GitHub [repository](https://github.com/smashkoala/CarND-Behavioral-Cloning-P3) too.


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I chose the NVIDIA network architecture introduced in the course.
Information on the following page is used as reference.
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

It consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 179-200)

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 177).

Five fully connected layers are implemented before the final output. (code line 196-199)

####2. Attempts to reduce overfitting in the model
No special mechanism is implemented in the model to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting such as using center, left and right camera images randomly. The images are randomly flipped too.
The model was tested by running it through the simulator and ensured that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 202).

####4. Appropriate training data

For training, sample data from Udacity and some specially collected data from the simulation are used.
The special data is collected mainly to overcome the situation where the car goes off the track on the bridge and two steep curves.
For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach
The NVIDIA network was chosen as a proven model architecture.
The model has not been changed so much until the end since I knew the the NVIDIA network was a proven architecture according to the course and the web page [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
In stead of tuning the model, I spent a lot of time to fine tune the image pre-processing and to balance the distribution of samples in terms of steering angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on both the training set and on the validation set. Since I chose the proven architecture, it seems that I did not have to spend so much time to overcome overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected sample data on these spots where the vehicle fell of the track, and used the samples for training and validation.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 175-200) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

I used the Udacity sample data for general driving behavior training.
Since the Udacity samples contain a lot of data whose steering angles are near 0, the car tend to go straight at curves. In order to avoid this, I reduced the number of data whose angles are near 0 by 85 %.

In addition, I used images on the left and right cameras by randomly selecting them.
The ratio of center, left and right are 40%, 30% and 30%.

To augment the data set, I randomly flipped images and angles since the track has more left turn curves than right turn curves.
![alt text][image6]
![alt text][image7]

With the process above, the vehicle could somehow run the one full lap, although at three specific spots such as on the bridge, and two steep curves after the bridge the vehicle became off the track.

In order to solve this, I collected samples on these spots on the simulator. I used only images of the center camera of these samples. In these samples, I recorded the vehicle going straight and recovering from left side and right sides of the road back to center.

![alt text][image3]
![alt text][image4]
![alt text][image5]


After the collection process, I had 6082 number of data points. I then preprocessed this data by
1) Cropped top 70 x 320 and bottom 25 x 320
2) Converted the BGR format to YUB
3) Applied gaussian blur
4) Resized the image to 64 x 64

I randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 since the validation loss did not change so much after 4 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
