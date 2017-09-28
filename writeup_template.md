#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
* cloning.py & cloning_gen.py which are similar files to model.py but different versions.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
sh
python drive.py model.h5
```
OR watch the video
```
python video.py run1 --fps 48
```

####3. Submission code is usable and readable

The model.py file contains all the code used in this project.
The first part of the project, all dependencies are imported and installed. After that, the empty array that will be used to append the iamges from the .csv file; seperation of the data and the creation of the gerenator function that does most of the augmentantion (flipping the images, shuffling and corrections).

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based in the NVIDIA model that was provided in the classroom. Some changes were made to provide a better accuracy:
* data is normalized using lambda and setting the values between -0.5 & 0.5 by dividing the data by 255
* addition of input_shape with rows, columns, and chanel (values were assigned before hand).
* cropping of the images since the bottom and top portions of the images did not help with the training, which help lowering the training time.
* convolutions are as followed:
    ** 24
    ** 36
    ** 48
    ** 64
    ** 64
* changed the border mode to valid.
* activations in each convolutions is RELU.
* added dropout of 0.5
* flatten the output as followed:
    ** 100
    ** 50
    ** 10
    ** 1

####2. Attempts to reduce overfitting in the model

Dropout of 0.5 was added before the convolution changes to 64, before flatten and after each iteration of flatten. The same value was kept throghout the entire model.

####3. Model parameter tuning

Adam Optimizer was applied; therefore, the learning rate was not applied.

####4. Appropriate training data

At the beginning of this project I was using my own data rather than the data provided by Udacity. My personal data consisted of me driving normally (staying in the middle) in the first course, driving in reverse in the first course, started recording when I was off the road on both left and right side and recording how I was getting back to the road in the first course; and finally me driving normally in the second course which has lanes, less biased to the left and has more sharp turns.
After several tries and fails, I decided to give Udacity's data a try and on the first run the car managed to complete the course with no crazy behaviour.
In the end, after training the model; my loss was: 0.0119 & val_loss was: 0.0120


