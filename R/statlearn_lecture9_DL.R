# Statistical Learning ----

# Lecture 9: Deep Learning ------------------------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - Deep Learning
# - Tensorflow
# - Keras



# Deep Learning -----------------------------------------------------------

# https://srdas.github.io/DLBook/

# Neural networks originated in the computer science field to answer questions 
# that normal statistical approaches were not designed to answer at the time. 
# The MNIST data is one of the most common examples you will find, where the 
# goal is to to analyze hand-written digits and predict the numbers written. 
# This problem was originally presented to AT&T Bell Lab’s to help build 
# automatic mail-sorting machines for the USPS (LeCun et al. 1990).

# This problem is quite unique because many different features of the data can 
# be represented. As humans, we look at these numbers and consider features 
# such as angles, edges, thickness, completeness of circles, etc. We interpret 
# these different representations of the features and combine them to recognize 
# the digit. In essence, neural networks perform the same task albeit in a far 
# simpler manner than our brains. At their most basic levels, neural networks 
# have three layers: an input layer, a hidden layer, and an output layer. The 
# input layer consists of all of the original input features. The majority of 
# the learning takes place in the hidden layer, and the output layer outputs 
# the final predictions.

# Although simple on the surface, the computations being performed inside a 
# network require lots of data to learn and are computationally intense rendering 
# them impractical to use in the earlier days. However, over the past several 
# decades, advancements in computer hardware (off the shelf CPUs became faster 
# and GPUs were created) made the computations more practical, the growth in data
# collection made them more relevant, and advancements in the underlying 
# algorithms made the depth (number of hidden layers) of neural nets less of
# a constraint. These advancements have resulted in the ability to run very deep 
# and highly parameterized neural networks (i.e., DNNs).

# Such DNNs allow for very complex representations of data to be modeled, which 
# has opened the door to analyzing high-dimensional data (e.g., images, videos, 
# and sound bytes). In some machine learning approaches, features of the data 
# need to be defined prior to modeling (e.g., ordinary linear regression). One 
# can only imagine trying to create the features for the digit recognition 
# problem above. However, with DNNs, the hidden layers provide the means to 
# auto-identify useful features. A simple way to think of this is to go back 
# to our digit recognition problem. The first hidden layer may learn about the 
# angles of the line, the next hidden layer may learn about the thickness of 
# the lines, the next may learn the location and completeness of the circles, 
# etc. Aggregating these different attributes together by linking the layers 
# allows the model to accurately predict what digit each image represents.

# This is the reason that DNNs are so popular for very complex problems where 
# feature engineering is important, but rather difficult to do by hand (e.g., 
# facial recognition). However, at their core, DNNs perform successive non-linear 
# transformations across each layer, allowing DNNs to model very complex and 
# non-linear relationships. This can make DNNs suitable machine learning 
# approaches for traditional regression and classification problems as well.
# But it is important to keep in mind that deep learning thrives when dimensions 
# of your data are sufficiently large (e.g., very large training sets). As the
# number of observations (n) and feature inputs (p) decrease, shallow machine 
# learning approaches tend to perform just as well, if not better, and are more 
# efficient.


# * Model's Components ---------------------------------------------------

# ** Layers & Nodes -------------------------------------------------------

# The layers and nodes are the building blocks of our DNN and they decide how 
# complex the network will be. Layers are considered dense (fully connected) 
# when all the nodes in each successive layer are connected. Consequently, the 
# more layers and nodes you add the more opportunities for new features to be 
# learned (commonly referred to as the model’s capacity).36 Beyond the input 
# layer, which is just our original predictor variables, there are two main 
# types of layers to consider: hidden layers and an output layer.

# Hidden layers
# There is no well-defined approach for selecting the number of hidden layers 
# and nodes; rather, these are the first of many hyperparameters to tune. With 
# regular tabular data, 2–5 hidden layers are often sufficient but your best 
# bet is to err on the side of more layers rather than fewer. The number of 
# nodes you incorporate in these hidden layers is largely determined by the 
# number of features in your data. Often, the number of nodes in each layer 
# is equal to or less than the number of features but this is not a hard 
# requirement. It is important to note that the number of hidden layers and 
# nodes in your network can affect its computational complexity (e.g., training 
# time). When dealing with many features and, therefore, many nodes, training 
# deep models with many hidden layers can be computationally more efficient than 
# training a single layer network with the same number of high volume nodes 
# (Goodfellow, Bengio, and Courville 2016). Consequently, the goal is to find 
# the simplest model with optimal performance.
 
# Output layers
# The choice of output layer is driven by the modeling task. For regression 
# problems, your output layer will contain one node that outputs the final 
# predicted value. Classification problems are different. If you are predicting 
# a binary output (e.g., True/False, Win/Loss), your output layer will still 
# contain only one node and that node will predict the probability of success 
# (however you define success). However, if you are predicting a multinomial 
# output, the output layer will contain the same number of nodes as the number 
# of classes being predicted. For example, in our MNIST data, we are predicting 
# 10 classes (0–9); therefore, the output layer will have 10 nodes and the output
# would provide the probability of each class.


# ** Activation Functions -------------------------------------------------

# A key component with neural networks is what’s called activation. In the human
# brain, the biologic neuron receives inputs from many adjacent neurons. When 
# these inputs accumulate beyond a certain threshold the neuron is activated
# suggesting there is a signal. DNNs work in a similar fashion.

# As stated previously, each node is connected to all the nodes in the previous 
# layer. Each connection gets a weight and then that node adds all the incoming 
# inputs multiplied by its corresponding connection weight plus an extra bias
# parameter (w0). The summed total of these inputs become an input to an 
# activation function.
 
# The activation function is simply a mathematical function that determines 
# whether or not there is enough informative input at a node to fire a signal 
# to the next layer. There are multiple activation functions to choose from but 
# the most common ones include: Linear (identity), Rectified linear unit (ReLU), 
# Sigmoid or Softmax.
 
# When using rectangular data, the most common approach is to use ReLU activation 
# functions in the hidden layers. The ReLU activation function is simply taking 
# the summed weighted inputs and transforming them to a 0	(not fire) or >	0	
# (fire) if there is enough signal. For the output layers we use the linear 
# activation function for regression problems, the sigmoid activation function 
# for binary classification problems, and softmax for multinomial classification.


# ** Backpropagation ------------------------------------------------------

# On the first run (or forward pass), the DNN will select a batch of observations,
# randomly assign weights across all the node connections, and predict the output.
# The engine of neural networks is how it assesses its own accuracy and 
# automatically adjusts the weights across all the node connections to improve 
# that accuracy. This process is called backpropagation. To perform 
# backpropagation we need two things: an objective function and an optimizer.
 
# First, you need to establish an objective (loss) function to measure 
# performance. For regression problems this might be mean squared error (MSE) 
# and for classification problems it is commonly binary and multi-categorical 
# cross entropy. DNNs can have multiple loss functions.

# On each forward pass the DNN will measure its performance based on the loss 
# function chosen. The DNN will then work backwards through the layers, compute 
# the gradient of the loss with regards to the network weights, adjust the 
# weights a little in the opposite direction of the gradient, grab another batch 
# of observations to run through the model, …rinse and repeat until the loss 
# function is minimized. This process is known as mini-batch stochastic gradient 
# descent (mini-batch SGD). There are several variants of mini-batch SGD 
# algorithms; they primarily differ in how fast they descend the gradient 
# (controlled by the learning rate). These different variations make up the 
# different optimizers that can be used.


# * Models & Algorithms ---------------------------------------------------

# ** Feed Forward Neural Network ------------------------------------------
# ** Convolutional Neural Network -----------------------------------------
# ** Recurrent Neural Network ---------------------------------------------
# ** Autoencoders ---------------------------------------------------------
# ** Reinforcement Learning -----------------------------------------------


# * Deep Learning Issues --------------------------------------------------

# If a practitioner where to attempt to build a model, there are still a number 
# of questions that have to be answered, such as:
 	
# 	- What is the best model for the problem to be solved?
# 	- Given the plethora of training algorithms, such as Momentum, Adam, Batch 
#     Normalization, Dropout etc., which are the ones that are most appropriate
#     for the model?
# 	- How much data is sufficient to train the model?
# 	- How should we select the various parameters associated with the model? 
#     These parameters are collectively known as hyper-parameters.
    

# ** Choosing the model ----------------------------------------------------

# There are a number of options with regards to model choice:
 	
# 	- If the input data is such that the various classes are approximately 
#     linearly separated (which can verified by plotting a subset of input data
#     points in two dimensions using projection techniques), then a linear model 
#     will work.
#   - Deep Learning Networks are needed for more complex datasets with non-linear
#     boundaries between classes. If the input data has a 1-D structure, then a 
#     Deep Feed Forward Network will suffice.
#   - If the input data has a 2-D structure (such as black and white images), 
#     or a 3-D structure (such color images), then a Convolutional Neural Network
#     or ConvNet is called for. In some cases 1-D or 2-D data can be aggregated 
#     into a higher level structure which then becomes amenable to processing 
#     using ConvNets. ConvNets excel at object detection and recognition in 
#     images and they have also been applied to tasks such as DLN based image generation.
#   - If the input data forms a sequence with dependencies between the elements 
#     of the sequence, then a Recurrent Neural Network or RNN is required. 
#     Typical examples of this kind of data include: Speech waveforms, natural 
#     language sentences, stock prices etc. RNNs are ideal for tasks such as 
#     speech recognition, machine translation, captioning etc.
     
# The basic structures described above can be combined in various ways to generate 
# more complex models. For example if the input sequence consists of correlated 
# 2-D or 3-D data structures (such as individual frames in a video segment), 
# then the appropriate structure is a combination of RNNs with ConvNets. Another 
# well know hybrid structure is called Network-in-a-Network and consists of a
# combination of ConvNets with Deep Feed Forward Networks. In recent years the 
# number of models has exploded with new structures that advance the state of the 
# art in various directions, such as enabling better gradient propagation during Backprop.


# ** Choosing the Algorithms ----------------------------------------------

# We have a choice of algorithm in the following areas:
 	
# 	- Choice of Stochastic Gradient Descent Parameter Update Equation. Of these 
#     Momentum and Nesterov Momentum are designed to speed up the speed of 
#     convergence and Adagrad and RMProp are designed to automatically adapt 
#     the effective Learning Rate. Adam combines both these benefits and is 
#     usually the default choice. If you choose to use a purely momentum based 
#     technique, then it is advisable to combine it with Learning Rate Annealing.
 
#   - Choice of Learning Rate Annealing Technique. Two of the most popular 
#     techniques are:
#     Keep track of the validation error, and reduce the Learning Rate by a factor
#     of 2 when the error appears to plateau. 
#     Automatically reduce the Learning Rate using a predetermined schedule. 
#     Popular schedules are: (a) Exponential decrease: η = ηo*10^−(t/r), so that the 
#     Learning Rate drops by a factor od 10 every r steps, (b) η = ηo*(1+t/r)^−c, 
#     this leads to a smaller rate of decrease compared to exponential.
     
#   - Choice of Activation Functions: The general rules in this area are: Avoid 
#     Sigmoid and Tanh functions, use ReLu as a default and Leaky ReLU to improve 
#     performance, try out MaxOut and ELU on an experimental basis.
 
#   - Weight Initialization Rules: Use the Xavier-He initializations as a default.
 
#   - Data Pre-Processing and use of Batch Normalization: Data Normalization as 
#     is always advisable and in case of image data, the Centering operation is 
#     sufficient. Batch Normalization is a very powerful technique which helps in 
#     a vareity of ways, and if you are encountering problems in the training 
#     process, then its use is advised.


# ** How much Data is Needed? ---------------------------------------------

# Data is the life blood of Deep Learning models. and sometimes it makes more 
# sense to add to the training dataset rather than use a more sophisticated 
# model. The effect of training dataset size and complexity on model performance
# and its interplay with model capacity, and we summarize the main conclusions:

# 	- Underfitting: If the model exhibits symptoms of Underfitting, i.e., 
#     performance on the training data is poor, then it means that the capacity 
#     of the model is not sufficient to correctly classify the training data with
#     a high enough probability of success. In this situation it does not make 
#     sense to add more training data, instead the capacity of the model should 
#     be increased by adding Hidden Layers or by improving the learning process. 
#     If none of these changes result in any improvement, then it points to a 
#     problem with the quality of the training data.
#   - Overfitting: If the model exhibits symptoms of Overfitting, i.e., 
#     classification accuracy on the validation data is poor even with low 
#     training data error, then it points to lack of sufficient training data. 
#     In this situation either more data should be procured or if this is not 
#     feasible, then either Regularization should be used to make up for the 
#     lack of data (which is the preferred option) or the the capacity of the 
#     model should be reduced. It is also possible to expand the training data 
#     artificially by using the Data Augmentation techniques.

# A common strategy that is used is to start with a model whose capacity is 
# higher than what may be warranted given the amount of training data, and then 
# add Regularization to avoid Overfitting.


# ** Tuning ---------------------------------------------------------------

# Find an optimal model by tuning different hyperparameters. There are many ways
# to tune a DNN. Typically, the tuning process follows these general steps:
	
#   - Adjust model capacity (layers & nodes);
#   - Add batch normalization;
#   - Add regularization;
#   - Adjust learning rate.



# * Final Considerations --------------------------------------------------

# Training DNNs often requires more time and attention than other ML algorithms.
# With many other algorithms, the search space for finding an optimal model is 
# small enough that Cartesian grid searches can be executed rather quickly. 
# With DNNs, more thought, time, and experimentation is often required up front 
# to establish a basic network architecture to build a grid search around. 
# However, even with prior experimentation to reduce the scope of a grid search, 
# the large number of hyperparameters still results in an exploding search space 
# that can usually only be efficiently searched at random.

# Historically, training neural networks was quite slow since runtime requires 
# O(NpML) operations where N = # observations, p = # features, M = # hidden 
# nodes, and L = # epchos. Fortunately, software has advanced tremendously 
# over the past decade to make execution fast and efficient. With open source 
# software such as TensorFlow and Keras available via R APIs, performing state 
# of the art deep learning methods is much more efficient, plus you get all the
# added benefits these open source tools provide (e.g., distributed computations
# across CPUs and GPUs, more advanced DNN architectures such as convolutional 
# and recurrent neural nets, autoencoders, reinforcement learning, and more!).



# Tensorflow --------------------------------------------------------------

# https://www.tensorflow.org/
# https://tensorflow.rstudio.com/

# TensorFlow is an end-to-end open source platform for machine learning. It has 
# a comprehensive, flexible ecosystem of tools, libraries and community 
# resources that lets researchers push the state-of-the-art in ML and developers 
# easily build and deploy ML powered applications.

# Easy model building:
# TensorFlow offers multiple levels of abstraction so you can choose the right 
# one for your needs. Build and train models by using the high-level Keras API, 
# which makes getting started with TensorFlow and machine learning easy.

# If you need more flexibility, eager execution allows for immediate iteration 
# and intuitive debugging. For large ML training tasks, use the Distribution 
# Strategy API for distributed training on different hardware configurations 
# without changing the model definition.

# Robust ML production anywhere:
# TensorFlow has always provided a direct path to production. Whether it's on 
# servers, edge devices, or the web, TensorFlow lets you train and deploy your
# model easily, no matter what language or platform you use.

# Use TensorFlow Extended (TFX) if you need a full production ML pipeline. For 
# running inference on mobile and edge devices, use TensorFlow Lite. Train and 
# deploy models in JavaScript environments using TensorFlow.js.

# Powerful experimentation for research:
# Build and train state-of-the-art models without sacrificing speed or 
# performance. TensorFlow gives you the flexibility and control with features 
# like the Keras Functional API and Model Subclassing API for creation of 
# complex topologies. For easy prototyping and fast debugging, use eager 
# execution.

# TensorFlow also supports an ecosystem of powerful add-on libraries and models 
# to experiment with, including Ragged Tensors, TensorFlow Probability, 
# Tensor2Tensor and BERT.



# Keras -------------------------------------------------------------------

# https://keras.io/
# https://keras.rstudio.com/
# https://blogs.rstudio.com/ai/
# https://tensorflow.rstudio.com/tutorials/

# Keras is an API designed for human beings, not machines. Keras follows best 
# practices for reducing cognitive load: it offers consistent & simple APIs, 
# it minimizes the number of user actions required for common use cases, and 
# it provides clear & actionable error messages. It also has extensive 
# documentation and developer guides.

# Iterate at the speed of thought:
# Keras is the most used deep learning framework among top-5 winning teams on 
# Kaggle. Because Keras makes it easier to run new experiments, it empowers 
# you to try more ideas than your competition, faster. And this is how you win.

# Exascale machine learning:
# Built on top of TensorFlow 2, Keras is an industry-strength framework that 
# can scale to large clusters of GPUs or an entire TPU pod. It's not only 
# possible; it's easy.

# Deploy anywhere:
# Take advantage of the full deployment capabilities of the TensorFlow platform. 
# You can export Keras models to JavaScript to run directly in the browser, 
# to TF Lite to run on iOS, Android, and embedded devices. It's also easy to 
# serve Keras models as via a web API.

# A vast ecosystem:
# Keras is a central part of the tightly-connected TensorFlow 2 ecosystem, 
# covering every step of the machine learning workflow, from data management 
# to hyperparameter training to deployment solutions.

# State-of-the-art research:
# Keras is used by CERN, NASA, NIH, and many more scientific organizations 
# around the world (and yes, Keras is used at the LHC). Keras has the low-level 
# flexibility to implement arbitrary research ideas while offering optional 
# high-level convenience features to speed up experimentation cycles.

# An accessible superpower:
# Because of its ease-of-use and focus on user experience, Keras is the deep 
# learning solution of choice for many university courses. It is widely
# recommended as one of the best ways to learn deep learning.


# * State of the Ecosystem ------------------------------------------------

# Let us start with a characterization of the ecosystem, and a few words on its 
# history. When we say Keras, we mean R – as opposed to Python – Keras. This 
# immediately translates to the R package keras. But keras alone wouldn’t get 
# you far. While keras provides the high-level functionality – neural network 
# layers, optimizers, workflow management, and more – the basic data structure 
# operated upon, tensors, lives in tensorflow. Thirdly, as soon as you’ll need 
# to perform less-then-trivial pre-processing, or can no longer keep the whole 
# training set in memory because of its size, you’ll want to look into tfdatasets.

# So it is these three packages – tensorflow, tfdatasets, and keras – that should
# be understood by “Keras” in the current context 2 . (The R-Keras ecosystem, 
# on the other hand, is quite a bit bigger. But other packages, such as tfruns 
# or cloudml, are more decoupled from the core.)
 
# Matching their tight integration, the aforementioned packages tend to follow a
# common release cycle, itself dependent on the underlying Python library, 
# TensorFlow. For each of tensorflow, tfdatasets, and keras , the current CRAN 
# version is 2.7.0, reflecting the corresponding Python version. The synchrony 
# of versioning between the two Kerases, R and Python, seems to indicate that 
# their fates had developed in similar ways. Nothing could be less true, and 
# knowing this can be helpful.
 
# In R, between present-from-the-outset packages tensorflow and keras, 
# responsibilities have always been distributed the way they are now: tensorflow 
# providing indispensable basics, but often, remaining completely transparent to
# the user; keras being the thing you use in your code. In fact, it is possible 
# to train a Keras model without ever consciously using tensorflow.
 
# On the Python side, things have been undergoing significant changes, ones 
# where, in some sense, the latter development has been inverting the first.
# In the beginning, TensorFlow and Keras were separate libraries, with TensorFlow 
# providing a backend – one among several – for Keras to make use of. At some 
# point, Keras code got incorporated into the TensorFlow codebase. Finally 
# (as of today), following an extended period of slight confusion, Keras got 
# moved out again, and has started to – again – considerably grow in features.


# * Installation ----------------------------------------------------------

# https://tensorflow.rstudio.com/installation/
# https://rstudio.github.io/reticulate/articles/python_packages.html


# First, install the reticulate package

install.packages("reticulate")

# Then, install the tensorflow R package from GitHub as follows

install.packages("tensorflow")

# Then, use the install_tensorflow() function to install TensorFlow. Note that 
# on Windows you need a working installation of Anaconda.

library(tensorflow)
install_tensorflow()

# or a specific version via install_tensorflow(version = "2.6.2").
# install_tensorflow is a wraper around reticulate::py_install().
# You can confirm that the installation succeeded with
	
library(tensorflow)
tensorflow::tf_config()
tensorflow::tf_version()

tf$constant("Hellow Tensorflow")
## tf.Tensor(b'Hellow Tensorflow', shape=(), dtype=string)

# This will provide you with a default installation of TensorFlow suitable for 
# use with the tensorflow R package. Read on if you want to learn about 
# additional installation options, including installing a version of TensorFlow 
# that takes advantage of Nvidia GPUs if you have the correct CUDA libraries 
# installed.


# Alternatively, you may directly install keras with
	
install.packages("keras")

# The Keras R interface uses the TensorFlow backend engine by default.

library(keras)
keras::install_keras()

# This will provide you with default CPU-based installations of Keras and 
# TensorFlow. If you want a more customized installation, e.g. if you want to 
# take advantage of NVIDIA GPUs, see the documentation for install_keras() and 
# the installation section.

# Finally, you should also install some useful helpers Tensorflow packages

install.packages("tfruns")
install.packages("tfestimators")


# * Loading ---------------------------------------------------------------

library(dplyr)

library(keras) # for fitting DNNs
library(tfruns) # for additional grid search & model training functions
library(tfestimators) # provides grid search & model training interface


# * Data ------------------------------------------------------------------

# We’ll use the MNIST data to illustrate various DNN concepts. With DNNs, it is 
# important to note a few items:
# 	- Feedforward DNNs require all feature inputs to be numeric. Consequently, 
#     if your data contains categorical features they will need to be numerically 
#     encoded (e.g., one-hot encoded, integer label encoded, etc.).
#   - Due to the data transformation process that DNNs perform, they are highly 
#     sensitive to the individual scale of the feature values. Consequently, we
#     should standardize our features first. Although the MNIST features are
#     measured on the same scale (0–255), they are not standardized (i.e., have 
#     mean zero and unit variance); the code chunk below standardizes the MNIST 
#     data to resolve this.
#   - Since we are working with a multinomial response (0–9), keras requires our 
#     response to be a one-hot encoded matrix, which can be accomplished with the
#     keras function to_categorical().
    
# Import MNIST training data
mnist <- dslabs::read_mnist()
mnist_train_x <- mnist$train$images
mnist_train_y <- mnist$train$labels
mnist_test_x <- mnist$test$images
mnist_test_y <- mnist$test$labels

# Rename columns and standardize feature values
colnames(mnist_train_x) <- paste0("V", 1:ncol(mnist_train_x))
colnames(mnist_test_x) <- paste0("V", 1:ncol(mnist_test_x))
mnist_train_x <- mnist_train_x / 255
mnist_test_x <- mnist_test_x / 255

# One-hot encode response
mnist_train_y <- to_categorical(mnist_train_y, 10)
mnist_test_y <- to_categorical(mnist_test_y, 10)


# * Create Network Structure ----------------------------------------------

# The keras package allows us to develop our network with a layering approach. 
# First, we initiate our sequential feedforward DNN architecture with 
# keras_model_sequential() and then add some dense layers. This example creates 
# two hidden layers, the first with 128 nodes and the second with 64, followed 
# by an output layer with 10 nodes. One thing to point out is that the first 
# layer needs the input_shape argument to equal the number of features in your 
# data; however, the successive layers are able to dynamically interpret the 
# number of expected inputs based on the previous layer.

model <- keras_model_sequential() %>%
	layer_dense(units = 128, input_shape = ncol(mnist_x)) %>%
	layer_dense(units = 64) %>%
	layer_dense(units = 10)
summary(model)


# * Control Activations ---------------------------------------------------

# To control the activation functions used in our layers we specify the 
# activation argument. For the two hidden layers we add the ReLU activation 
# function and for the output layer we specify activation = softmax (since 
# MNIST is a multinomial classification problem).

model <- keras_model_sequential() %>%
	layer_dense(units = 128, activation = "relu", input_shape = p) %>%
	layer_dense(units = 64, activation = "relu") %>%
	layer_dense(units = 10, activation = "softmax")
summary(model)


# * Incorporate Backpropagation -------------------------------------------

# To incorporate the backpropagation piece of our DNN we include compile() in our
# code sequence. In addition to the optimizer and loss function arguments, we 
# can also identify one or more metrics in addition to our loss function to track 
# and report.

compile(
	loss = 'categorical_crossentropy',
	optimizer = optimizer_rmsprop(),
	metrics = c('accuracy')
)

model <- keras_model_sequential() %>%
	# Network architecture
	layer_dense(units = 128, activation = "relu", input_shape = ncol(mnist_x)) %>%
	layer_dense(units = 64, activation = "relu") %>%
	layer_dense(units = 10, activation = "softmax") %>%
	# Backpropagation
	compile(
		loss = 'categorical_crossentropy',
		optimizer = optimizer_rmsprop(),
		metrics = c('accuracy')
	)
summary(model)


# * Training --------------------------------------------------------------

# We’ve created a base model, now we just need to train it with some data. To do
# so we feed our model into a fit() function along with our training data. We 
# also provide a few other arguments that are worth mentioning:
 	
# 	- batch_size: As we mentioned in the last section, the DNN will take a batch
#     of data to run through the mini-batch SGD process. Batch sizes can be 
#     between one and several hundred. Small values will be more computationally 
#     burdensome while large values provide less feedback signal. Values are 
#     typically provided as a power of two that fit nicely into the memory 
#     requirements of the GPU or CPU hardware like 32, 64, 128, 256, and so on.
#   - epochs: An epoch describes the number of times the algorithm sees the 
#     entire data set. So, each time the algorithm has seen all samples in the 
#     data set, an epoch has completed. In our training set, we have 60,000 
#     observations so running batches of 128 will require 469 passes for one 
#     epoch. The more complex the features and relationships in your data, the 
#     more epochs you’ll require for your model to learn, adjust the weights, 
#     and minimize the loss function.
#   - validation_split: The model will hold out XX% of the data so that we can 
#     compute a more accurate estimate of an out-of-sample error rate.
#   - verbose: We set this to FALSE for brevity; however, when TRUE you will see
#     a live update of the loss function in your RStudio IDE.
    
# Plotting the output shows how our loss function (and specified metrics) improve 
# for each epoch. We see that our model’s performance is optimized at 5–10 epochs 
# and then proceeds to overfit, which results in a flatlined accuracy rate.

# Train the model
net_fit <- model %>%
	fit(
		x = mnist_train_x,
		y = mnist_train_y,
		epochs = 25,
		batch_size = 128,
		validation_split = 0.2,
		verbose = FALSE
	)
net_fit
## Trained on 48,000 samples, validated on 12,000 samples (batch_size=128, epochs=25)
## Final epoch (plot to see history):
## val_loss: 0.1512
##  val_acc: 0.9773
##     loss: 0.002308
##      acc: 0.9994
plot(net_fit)


# * Evaluating ------------------------------------------------------------

# We can now make predictions with our model using the predict function

preds <- predict(net_fit, mnist_test_x)
head(preds, 2)

# By default predict will return the output of the last Keras layer. In our case
# this is the probability for each class. You can also use predict_classes and 
# predict_proba to generate class and probability - these functions are slighly 
# different then predict since they will be run in batches.

# You can access the model performance on a different dataset using the evaluate 
# function, for example
	
model %>%	evaluate(mnist_test_x, mnist_test_y, verbose = 0)


# * Tuning ----------------------------------------------------------------

# Now that we have an understanding of producing and running a DNN model, the 
# next task is to find an optimal model by tuning different hyperparameters. 
# There are many ways to tune a DNN. Typically, the tuning process follows z
# these general steps:

#   - Adjust model capacity (layers & nodes);
#   - Add batch normalization;
#   - Add regularization;
#   - Adjust learning rate.










