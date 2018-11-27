## Behavioral Cloning Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<a href="https://youtu.be/tZ3duAM8d-E" target="_blank"><img src="http://img.youtube.com/vi/tZ3duAM8d-E/0.jpg" 
alt="This My test video" width="960" height="540" border="10" /></a>

### This My test video

## In this project, I used my deep neural network and convolutional neural network to clone driving behavior. I use Keras to train, validate and test the model. This model will output a steering angle to an autonomous vehicle.


[//]: # (Image References)

[image1]: ./graph_run.png "graph_run"
[image2]: ./md_output/Original_center_2018_11_27_18_36_20_501.jpg "Original_center_2018_11_27_18_36_20_501"
[image3]: ./md_output/Random_Bright_Cutcenter_2018_11_27_18_36_20_501.jpg "Random_Bright_Cutcenter_2018_11_27_18_36_20_501"
[image4]: ./md_output/Random_Bright_Cut_flippedcenter_2018_11_27_18_36_20_501.jpg "Random_Bright_Cut_flippedcenter_2018_11_27_18_36_20_501"

[image5]: ./Tensorboard.png "Tensorboard"

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/ChenKuanSun/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/ChenKuanSun/CarND-Behavioral-Cloning-P3/blob/master/drive.py)  for driving the car in autonomous mode
* [model.h5](https://github.com/ChenKuanSun/CarND-Behavioral-Cloning-P3/blob/master/model.h5)  containing a trained convolution neural network 
* [writeup_report.md](https://github.com/ChenKuanSun/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md) summarizing the results
* [video.mp4](https://github.com/ChenKuanSun/CarND-Behavioral-Cloning-P3/blob/master/video.mp4) Simulated driving view movie
* (Option) [Dataset](https://drive.google.com/open?id=1kQB2dtGGOKop65lwvEiWxxfHTUBg3CeV) Data set used to train the model
* (Option) [Environment](https://github.com/ChenKuanSun/CarND-Behavioral-Cloning-P3/blob/master/environment.yml) If you want to train in the same environment, here is my Conda environment profile.
#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model consists of a convolutional neural network (using [Nvidia's Deep Learning Self Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)) with filter sizes of 5x5 and 3x3, which includes a nonlinear RELU layer and uses the Keras lambda layer to map the data in the model. One.

The following is a visual model diagram using TensorBoard:

![alt text][image1]



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

```python
model.add(Dropout(rate=0.5))
```

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
```python
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=10, callbacks=model_callback)
```


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
```python
model.compile(loss='mse', optimizer='adam')
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving the model architecture is to use the model developed by Nvidia and apply it to my project. Because the model uses an image with a size of 66X200 as input, I try to use a rectangular image for input when cropping the image to meet the best model requirements.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

At first, my data set was too small, which caused many situation models to be unpredictable, the error became very large, and then I went back to the simulator.

How the steering should change when simulating various conditions.After collecting data many times, I found that the car would be excessively shaken.

So I also collected a number of smooth driving, and I also practiced this many times in simulated driving.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700    
_________________________________________________________________
activation_1 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
```
Here is a visualization of the architecture
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I recorded that the vehicle recovered from the left and right sides of the road to the center so that the vehicle learned how to distinguish the off-center lane.
I cropped the various input images and randomly adjusted the light (I switched to the HSV channel and adjusted the V Channel), then made a large amount of analog data by flipping it according to the adjusted pattern.

I have done random brightness adjustment and cropping when I entered the data.
```python
def process_image(image):
    #Convert to HSV to adjust brightness
    HSV_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    #Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    #Apply the brightness reduction to the V channel
    HSV_image[:,:,2] = HSV_image[:,:,2]*random_bright

    #Convert to RBG again
    HSV_image = cv2.cvtColor(HSV_image,cv2.COLOR_HSV2RGB)

    #Cut image
    Cut_image = HSV_image[55:135, :, :]

    #Convert to np.array
    processed_image = Cut_image.astype(np.float32)
    return processed_image
```
But in drive.py I only do the cropping to avoid random light affecting the model's prediction.

![alt text][image3]
![alt text][image4]

If you want to know more, I also randomly sample the image to ./md_output/ check.

There are many times when there is a 0-value steering problem during the data acquisition process.I randomly dropped him to avoid over-influencing the results.

```python
#Drop some 0 steering data
if float(line[3])== 0.0:
    if np.random.choice([True, False]):
        continue
```

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is the visualization data during the training process:

![alt text][image5]

You can open the Tensorboard to view the log that is normalized by entering the following command:
```sh
tensorboard --logdir=/log
```