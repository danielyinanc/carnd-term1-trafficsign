# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./graphs/training_histogram.png "Training Labels Distribution"
[image5]: ./graphs/validation_histogram.png "Validation Labels Distribution"
[image6]: ./graphs/testing_histogram.png "Testing Labels Distribution"
[image7]: ./graphs/non_normalized_image.png "Non normalized Traffic Sign"
[image8]: ./graphs/normalized_image.png "Normalized Traffic Sign"
[image9]: ./images/60kmspeedlimit.png "60 km speed limit"
[image10]: ./images/nopassing.png "No passing Sign"
[image11]: ./images/roadworksign.jpg "Road Work Sign"
[image12]: ./images/slipperyroad.png "Slippery Road Sign"
[image13]: ./images/noentry.jpg "No Entry Sign"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

## Submission Files

You're reading it! and here is a link to my [project code](https://github.com/danielyinanc/carnd-term1-trafficsign/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Dataset Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 in three channels
* The number of unique classes/labels in the data set is 42

#### Exploratory Visualization

Here is an exploratory visualization of the data set. Series of histogram charts show how training, validation and testing data labels are distributed. 
Collectively they paint a well distributed data set for training, validation and testing purposes.

![alt text][image4]
![alt text][image5]
![alt text][image6]

### Design and Test a Model Architecture

#### Preprocessing

As a first step, I looked into min-max scaling/normalizing training, validation and testing images in order to eliminate possible sources of invariant factors creeping in (light, color, contrast etc). 


Here is an example of a traffic sign images before and after normalization.
![alt text][image7]
![alt text][image8]

The difference between the original data set and the augmented data set is due to scaling, a lot of small variations in tone and contrast is successfully eliminated.


#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   outputs  28x28x6			| 
| RELU					|												|
| Max pooling       	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 3x3	    | 2x2 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling       	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten				|  output = 400                            		|
| Fully Connected		|  input = 400 output= 120				        |
| RELU					|												|
| Fully Connected		|  input = 120 output= 84				        |
| RELU					|												|
| Fully Connected		|  input = 84 output= 43				        |


This is an unmodified LENET architecture except input and output dimensions.

#### Model Training

To train the model, I used an 
* Optimizer: Adams optimizer
* Batch Size: 100
* Epochs: 50
* Learning Rate: 0.001

#### Solution Approach

My final model results were:
* validation set accuracy of 0.938
* test set accuracy of 0.934




If a well known architecture was chosen:
I chose LENET because it has been used to very successfully recognize hand written numbers which are part of the feature of a number of traffic signs. Final models validation and training
results are both over 93 percent which indicate neither overfitting, nor underfitting is taking place.
 

### Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs that I found on the web:

![alt text][image10] 

![alt text][image12] 

![alt text][image11] 

![alt text][image13]

![alt text][image9] 



Road work and No entry signs have a lot of additional elements lying about and have some image degradation. Looking at them alone, I would have thought model would have the greatest
difficulty with them. However it turns out that only one that is missed is 60 km as explained below... and that is mistaken to be 50 km speed limit sign.

#### Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Passing    		| No Passing									| 
| Slippery Road 		| Slippery Road									|
| Road Work				| Road Work										|
| No entry	      		| No entry  					 				|
| 60 km/h speed limit	| 50 km/h speed limit  							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.4 percent

#### Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is very sure that this is a no passing sign (probability of 1.0), and the image does contain no passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| No passing   									| 
| negligible    		| End of no passing								|
| negligible			| Dangerous curve to the right					|
| negligible   			| Dangerous curve to the left	 				|
| negligible    	    | Slippery road     							|


For the second image, the model is very sure that this is a slippery road sign (probability of 0.999), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9         			| Slippery road 								| 
| 0.075          		| Right-of-way at the next intersection			|
| negligible			| Dangerous curve to the left					|
| negligible   			| Children crossing         	 				|
| negligible    	    | Dangerous curve to the right					|

For the third image, the model is very sure that this is a road work sign (probability of 1.0), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| Road work      								| 
| negligible       		| Beware of ice/snow                    		|
| negligible			| Bicycles crossing         					|
| negligible   			| Stop                       	 				|
| negligible    	    | Priority road             					|

For the fourth image, the model is very sure that this is a road work sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| Road work      								| 
| negligible       		| Beware of ice/snow                    		|
| negligible			| Bicycles crossing         					|
| negligible   			| Stop                       	 				|
| negligible    	    | Priority road             					|

For the fifth image, the model is very sure that this is a 50 km speed limit sign (probability of 99.7), and the image however contains a 60 km speed limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| Speed limit (50km/h)   						| 
| negligible       		| Speed limit (30km/h)                   		|
| negligible			| Speed limit (60km/h)        					|
| negligible   			| Wild animals crossing        	 				|
| negligible    	    | Speed limit (80km/h)         					|

Considering that 60 km is a slightly curved but contains most characters/symbols of 50, we can consider this a near miss and speaks quite well about the neural network's robustness as to predicting captured images.