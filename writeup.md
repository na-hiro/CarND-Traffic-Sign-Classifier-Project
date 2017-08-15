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

[image1]: ./reports/class_sample.jpg "Class_sample"
[image2]: ./reports/train_class.jpg "traning_class"
[image3]: ./reports/test_class.jpg "test_class"
[image4]: ./reports/augment.jpg "augment"
[image5]: ./reports/original.jpg "original"
[image6]: ./reports/proced.jpg "proced"
[image7]: ./reports/acc_fig.jpg "acc_fig"
[image8]: ./reports/loss_fig.jpg "loss_fig"
[image9]: ./reports/web_org_img1.jpg "web_org_img1"
[image10]: ./reports/web_org_img2.jpg "web_org_img2"
[image11]: ./reports/web_result.png "web_result"
[image12]: ./reports/out_map.jpg "out_map"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/na-hiro/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (H , V ,color_ch)=(32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
![alt text][image1]

![alt text][image2]

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Step 1: <br>
The above figure shows that nonuniform image data exists in the learning data depending on the class. By processing the original data, I enhanced the data for learning. Processing for augmentation is a combination of rotation with rotation, size change, movement, and size change. Through the above processing, we believe that learning accuracy for class units and image units can be improved.
The distribution of image data after preprocessing is shown in the figure below.

![alt text][image4]

#### Step 2: <br>
Convert image data to grayscale.
#### Step 3: <br>
As shown in the sample, the image data includes image data set under the under-exposure and over-exposure environment. The grayscale images are normalized by using the mean value and standard deviation of each image. Through the above preprocessing, image data is standardized to image data with an average of 0 and a variance of 1. Through this processing, it becomes possible to accurately compare each image. it seem that learning accuracy for class units and image units can be improved by processing.

![alt text][image5]![alt text][image6]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

#### Layer
| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|						-						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|						-						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten     |       outputs = 400              |
| Full Connection     |       outputs = 200              |
| Dropout     |       dropout ratio is 50%              |
| Full Connection     |       outputs = 100              |
| Dropout     |       dropout ratio is 50%              |
| Full Connection     |       outputs = 43              |
|  SoftMax         | -           |
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The following figure shows the history of accuracy and loss at the time of learning. As shown in the figure, accuracy and loss have improved for each learning, and it is shown that the convergence finally occurred. Since the learning data was enhanced, the this batch size setting was adopted because the numerical value was not stable when the batch size was small. Also, the above setting was adopted because accuracy and loss are good, and fluctuation after convergence is small. In addition, For the Optimization method, the following method was adopted.
Loss function: cross entropy
Optimization method: Adaptive Moment Estimation

![alt text][image7]
![alt text][image8]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.989.
* validation set accuracy of 0.968.
* test set accuracy of 0.946.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  - I selected the LeNet model as the base model. This model realized good results in the classification task using the MNIST data set. It is also a model that achieves both performance and cost.

* What were some problems with the initial architecture?
  - Initially, in the image data acquired from Web data, classification with fewer training data could not classify with misclassification and high probability in some cases. In the original data set, it is considered that there was a big difference in the number of image data for each class, and a proper feature amount could not be obtained. For example, images with similar feature quantities may be misclassified into classes with many images. Therefore, training data for each class was equalized. Furthermore, because there is no dropout processing, over learning may occur.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  - Initially, learning was done without strengthening image data. Since the amount of learning data is small, the batch size was about 128. However, for the purpose of further improving generalization, we enhanced image data and increased training data. At that time, the batch size was adjusted, but the values ​​around 1024 were less fluctuating after convergence and also gave good results in the classification of Web data. Perhaps it is thought that learning is carried out in a state covering many class data rather than repeating learning using a small number of images, which will improve generalization.

* Which parameters were tuned? How were they adjusted and why?
  - Learning rate, batch size, number of epochs, augmented training data size, were adjusted. As mentioned above, we believe that the batch size greatly affected classification accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  - The convolution process is a process of efficiently acquiring features of a pixel distribution of a two-dimensional image. Details are as follows. By the convolution processing, the inner product calculation is performed with the weight filter and the input image, or the weight filter and the feature map, and a feature map is acquired. The weight filter is learned by the gradient descent optimization method (Adam) by error back propagation method.
Maxpooling is a process of outputting a value from a small area of an input feature map and converting it into a new feature map. The purpose of the pooling process is as follows.First, it reduces the number of units by pooling and reduces the parameters to adjust.　Secondly, by outputting the response value from a certain small region, it is the object to acquire immutability to geometric change and the like.

If a well known architecture was chosen:
* What architecture was chosen?
  - I chosed LeNet.

* Why did you believe it would be relevant to the traffic sign application?
  - LeNet is a compact configuration and can be optimized for traffic sign classification with a little
customization. Therefore, LeNet is suitable for traffic sign classification in terms of both accuracy and cost.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  - In the final model, the accuracy of the training approaches 99%, and shows accuracy of more than 0.97 in both the validation data set and the test data set.
It can be expected that good classification results are output not only in image data used for learning but also in unknown image data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Test images include images of classes with fewer original learning images and many original learning images. For example, ClassID; 32, 40, the original training image is small. As shown below, all images are correctly classified. However, the probability of the traffic signs 120 km / h of ClassID 8 is similar to 20 km / h, which is judged as Top 2. It is shown that misclassification may occur in similar signs or signs containing the same character.

Here are the results of the prediction:
![alt text][image11]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The acquired feature map will be described below.
From the figure below, it seems that it is acquiring complicated visual patterns with the upper layer filter. In the lower layer, it seems that more characteristic parts are extracted from the feature map of the upper layer. In the lower layer, it seems that more important and distinctive properties are extracted when classifying.

![alt text][image12]

#### End of file
