# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Before the the data set summary and exploration the first task is to load the dataset to the working juyter notebook. There data sets are loaded, the frist is set is training set and the second set is the validation set, the last dataset is the testing set. 

After the loading the data sets into the jupyter notebook a summary of was made for each of the data set which conists of different traffic sign images. The numpy libaray is used for this part. The folllowing is the summayr statistic for about the data sets. 


* The size of training set is = 34799
* The size of the validation set is = 4410
* The size of test set is = 12630
* The shape of a traffic sign image is = (32, 32, 3)
* The number of unique classes/labels in the data set is = 43

#### 2. Include an exploratory visualization of the dataset.
##### Visualization
Fisrt let us have an understanding of  some of the traffic signs that the training data set consists of. The following figure show the plot of randomly selected images for this data set and with thier assigned label number. 

![alt text](https://github.com/kulu80/Traffic_Sign_Classifier-DeepLearning-/blob/master/traffic_sign_image1.png)

I mostly used the pandas liberary and seaborn for data manipulation and data isualization of the traffic data set. The fist thing that I do is to explore the number of each (frequence/occurence) of each classes in the triaining data set. This important to know the nature of distribution the data set, if skwed to one class it greatly affects the performance of a model. The following bargraph dipicts the distribution of each classes of the traffic sign images for the training data set. 

![alt text](https://github.com/kulu80/Traffic_Sign_Classifier-DeepLearning-/blob/master/total_count.png)

As it can be percieved from the bar graph above, the distribution of classes of images in not  uniform or Guassian. The distribution could show the natural occurence or frequency of traffic signs on the road or it might not. The most frequeenct traffic sign (image) in the training data set is the 50km/hr speed limit sign (which is labeled as 2) and the least frquent traffic sign is the 20 km/hr speed limit (which is labeled as 0). About 27 of the traffic signs have frquency of less than 1000 and for traffic sign frquency less than 1000 I increased the frquecy of each to 1400 to get a uniform distribution for most of the classes. The figure below shows the distribution of the traffic sign classes after the modification performed. 

![alt text]( https://github.com/kulu80/Traffic_Sign_Classifier-DeepLearning-/blob/master/total_count_after.png)
![alt text][image1]




### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Pre-processing

It is recommended to convert images into gray scale to impove model performance, first I converted all images to gray-scale and also apply normalization and tried to train the model,however both prepocessing techniques  didn't imporve model performance. Though I included the codes in the project I didn't use them in the model development process due to the stated reason. I used another preprocessing technique, the histogram-equalization which improves model performance. The histogram equalizer imporves image contrast by allowing for areas of lower local contrast to gain a higher contrast. Histogram equalization accomplishes this by effectively spreading out the most frequent intensity values [https://en.wikipedia.org/wiki/Histogram_equalization](https://en.wikipedia.org/wiki/Histogram_equalization). The first figure below shows the orignal images and the second is after histogram equalizer applied. 

##### Orignal images
![alt text](https://github.com/kulu80/Traffic_Sign_Classifier-DeepLearning-/blob/master/traffic_sign_image1.png)

##### After histogram-equalization
![alt text](https://github.com/kulu80/Traffic_Sign_Classifier-DeepLearning-/blob/master/hist_equalizer.png)


##### Image Agumentation

The challenge of application of  convolution neural network or deep learning in general  for immage classiffication is that it demands huge data to be availabe so that to have a fine working model. One of the ways to solve luck of data is to use image agumentation technique, which require to transform the availables image data by applying different image processsing techniques. There are many libraray that works with python to achieve image agumentation. Here,I used the OpenCV library (cv2) to transform images.OpenCV provides two transformation functions, ** cv2.warpAffine ** and **cv2.warpPerspective ** , with which you can have all kinds of transformations [http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html](http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html). 

For this project I only have used image rotation transformation. I have rotated each image with angles which ranges between -10 and 10 with 2 degree intervals which increase the number of training images from about 34,000 to  more than 600,000 by many folds. 



 
As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

In this project the LeNet model architeture show below is used with out many modification. I have tried to change convolution layer size and stride size and the resuling architucture didn't improve the model performance. Then stick to  the original LeNet Architucture, the difference here was I used the dropout regularization at the second convolutional and fullly connected layer to reduce model overfitting. 

![alt text](https://github.com/kulu80/Traffic_Sign_Classifier-DeepLearning-/blob/master/lenet.png)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I first tried to train the model with out making any image modification and transformation at which the validation accuracy never jumps above 0.75. Then next I have applied normalization  and converting to gray-scale procedures to all images: training and validation sets but sill the validation accuracy never approachs to 90% . After applied a rotation transformation which rotates the all images in the range of angles between -10 and 10 with a step of 2 degree interval using OpenCV liberary(cv2). This transformation not only increases the number of training sets and validation sets conmibed to more than 650,000 images but also jumps the validation accuracy to cloth to 99% with only 20 iterations made. I also have applied normalization and gray-scale converion after rotaion transformation on images was applied, however din't imporve model performance while in the contrary. I also have tried to modifiy the model architecture given above however never was I lucky to get model accurcy to above 50%. Then decided to stick the model architecture provided in the above section.  

My final model results were:
* training set accuracy of = 99.9
* validation set accuracy of = 99.9
* test set accuracy of =94.6

![alt text]( https://github.com/kulu80/Traffic_Sign_Classifier-DeepLearning-/blob/master/validation_graph.png)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the third image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the fourth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

for the fifth image





### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


