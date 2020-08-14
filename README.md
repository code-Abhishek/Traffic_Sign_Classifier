# **Traffic Sign Recognition** 


### Developing a free tool for all the 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[imagesign_cat]: ./signs_by_category.png "Visualization"
[imagesign_cat_after]: ./signs_by_category_after.png "Visualization_after"
[imageaugment]: ./augmented.png "Original & Augmented Image"
[image1]: ./data/test_images/image1.jpeg "Traffic Sign 1"
[image2]: ./data/test_images/image2.jpeg "Traffic Sign 2"
[image3]: ./data/test_images/image3.jpeg "Traffic Sign 3"
[image4]: ./data/test_images/image4.jpeg "Traffic Sign 4"
[image5]: ./data/test_images/image5.jpeg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/code-Abhishek/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
The [dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is images of German Traffic Signs that have been annotated and witht their class. The file [signnames.csv](./signnames.csv) provides the different classes to classify with.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 + (Added augmented images 13525)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of traffic signs we have under the 43 categories present in the dataset.
![alt text][imagesign_cat]

We can see there's quite an imbalance in the amount of images for each category. This later proved as a challenge, solving which helped improve accuracy of the model by adding more data to the the data set

Here's the distribution of training images after the adding new data.
![alt text][imagesign_cat_after]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the images because that will help to smooth the processing of the image

Here is an example of a traffic sign image before and after normalizing.

![alt text][image2]

The second step was getting the new data through augmentation, I applied the following techniques :
- Image augmentation technique of rotating the images with a randomized preset angle in degrees
- Saved images for easier access and lower GPU-time, by avoiding regeneration of the images again and again. 

We have 13525 additional (augmented) images from the training data itself.

Here is an example of an original image and an augmented image:
![alt text][imageaugment]


The next step was to setup the model.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was developed from the base of the LeNet model.

It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB Preprocessed image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| Flatten 				| outputs 800x1 nodes single layer				|
| Fully connected		| outputs 256x1 nodes single layer				|
| RELU					|												|
| Dropout				| drop rate=0.8 								|
| Fully connected		| outputs 128x1 nodes single layer				|
| RELU					|												|
| Dropout				| drop rate=0.8 								|
| Output				| Fully Connected with 43 output classes		|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, and provide enough iterations for convergence hyperparameters set are :
learn_rate = 0.005
epochs = 15
batch_size = 128

I used average mean of cross_entropy with the outputs from model and one-hot encoded y labels for error calculation.
I utilized Adam optimizer with the learn_rate to minimize the error/loss calculated above.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

<!---If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?--->

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

- The 5x5 filter was used to achieve a little more smoothing of the image in the convolutional layers.I added dropout layer to provide redundant representations, and preventing overfit on data.

* Which parameters were tuned? How were they adjusted and why?
- I tuned the size of the inputs to the layers of the model to retain more features from a patch of the image.
- I also tuned the batch_size, epochs, and learn rate.


Epochs:          
- 10 
- 15 
- 20 

Learn-rate:
- 0.1
- 0.01
- 0.005 
- 0.001



<!-- * Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well-->
The model has x.xx accuracy on the test set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

Some of the text images might be difficult to classify given that they have a natural background to them, which is slightly different from the training images, and would require the transitional invariance understanding in the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h       		| Priority Road									| 
| U-turn     			| Yield											|
| Yield					| Yield											|
| Stop sign	      		| Traffic Signals				 				|
| Slippery Road			| Stop               							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares unfavorably to the accuracy on the test set images. Maybe the model is not as good at prediction for unseen images

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the second last cell of the Ipython notebook.

My model was not really showing predictions but rather was showing the logits, and the values.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


