### Eduardo Ribeiro de Campos - Udacity / SDCND - May./2019.


# Self-Driving Car Engineer Nanodegree

## **Project4: Behavioral Cloning** 


[//]: # (Image References)

[image1]: ./images/udacity_simulator.jpg
[image2]: ./images/dataset_images.jpg
[image3]: ./images/training_process.jpg
[image4]: ./images/cnn-architecture.jpg
[image5]: ./images/model_summary.jpg
[image6]: ./images/model_mean_squared_error_loss.jpg
[image7]: ./images/video.jpg

 

The goals / steps of this project are the following, more details see the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) 

1. **Use the simulator to collect data of good driving behavior**<br/> 
2. **Build, a convolution neural network in Keras that predicts steering angles from images**<br/> 
3. **Train and validate the model with a training and validation set**<br/> 
4. **Test that the model successfully drives around track one without leaving the road**<br/> 
 

**Here is a link to [code file](./model.py)**

## 1 - Use the simulator to collect data of good driving behavior.

The simulator used was provided by Udacity and it is available in project workspace. 

![alt text][image1]

To accomplish the goals of the project it is necessary train and simulate the car driving autonomously only in the track 1 (the left one).

The strategy to getting data was offered in the project session by Udacity as a proposal. Below it is possible see in the schematic picture the 4 laps performed and the objective for each lap.

![alt text][image3]

The output data was zipped and imported to my personal google drive account. Here is the link to download the [file](https://drive.google.com/file/d/16wo6T1mjTcNS6QnCiLscyo8yJAqEPnOU/view?usp=sharing).

To import the data in the project workspace it was used a Script available on [`stackoverflow`](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive), the answer was provided by
[Aatif Khan](https://stackoverflow.com/users/3292493/aatif-khan) on May 28 '18 at 21:11. The script file is available on the workspace, for more details see the file [download_google_drive_data.sh](./download_google_drive_data.sh).

Before unzip the file is necessary delete the current folder named `data`.

The output data is a folder with images and a csv file containing in each rows the name of the images (3 images - center , left and right cameras ) and the steering angle associated with the images.
Below 3 examples:

![alt text][image2]


## 2 - Build, a convolution neural network in Keras that predicts steering angles from images.


The reference used here was the convolutional neural network architeture provided by [Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).


![alt text][image4]


To construct the convolutional neral network was used [keras](https://www.tensorflow.org/guide/keras) interface from `tensorflow`<br/> 
Some parameter of the architeture were modified and the picture below show de results obtained from the function `model.summary()` (see line xxx).


![alt text][image5]



## 3 - Train and validate the model with a training and validation set.

The data set were created using a `generator` function following the guide provided by udacity in the project session. Using the python  [scikit-learn](https://scikit-learn.org/stable/)library. the function `train_test_split` was applied to split the train and validation samples.

#### Model parameter tuning:
No of epochs= 5<br/> 
Optimizer Used- Adam<br/> 
Learning Rate- Default 0.001<br/> 
Validation Data split- 0.2<br/> 
Generator batch size= 32<br/> 

Loss Function Used- MSE(Mean Squared Error as it is efficient for regression problem.

![alt text][image6]


## 4 - Test that the model successfully drives around track one without leaving the road .

To simulate the algorithm result and see the car driving autonomously. it was used the script files available on the workspace and provided by Udacity. In the linux bash was inserted the command `python drive.py model.h5 run1` as first step and to converted the images generated in the run1 folder, was applied a second command  `python video.py run1`. <br/> see below the result !!!

<br/> 

<video controls src="./video.mp4"/>


https://youtu.be/-1MkRQP87yQ
