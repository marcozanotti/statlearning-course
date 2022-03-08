# Statistical Learning ----

# Lecture 4: Hyperparameter Tuning ----------------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - 
# - 
# - 



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")



# Data --------------------------------------------------------------------

# Let’s use again data from Hill, LaPan, Li, and Haney (2007), available 
# in the modeldata package, to predict cell image segmentation quality 
# with resampling. To start, we load this data into R.

data(cells, package = "modeldata")
cells %>% glimpse()	


# * Explorative Data Analysis ---------------------------------------------

cells %>% glimpse()

cells %>% 
	count(class) %>% 
	mutate(prop = n/sum(n))


cells %>% 
	skimr::skim() 

cells %>% 
	DataExplorer::create_report()


# * Data Splitting --------------------------------------------------------

set.seed(123)
cell_split <- initial_split(cells %>% select(-case), strata = class)

cell_train <- training(cell_split)
cell_test <- testing(cell_split)



# Hyperparameter Tuning ---------------------------------------------------

# Random forest models are a tree-based ensemble method, and typically 
# perform well with default hyperparameters. However, the accuracy of 
# some other tree-based models, such as boosted tree models or decision 
# tree models, can be sensitive to the values of hyperparameters. 
# In this lecture, we will train a decision tree model. There are several
# hyperparameters for decision tree models that can be tuned for better 
# performance. Let’s explore:
# 	
# 	 * the complexity parameter (which we call cost_complexity in tidymodels)
#      for the tree, and
#    * the maximum tree_depth.

# Tuning these hyperparameters can improve model performance because 
# decision tree models are prone to overfitting. This happens because 
# single tree models tend to fit the training data too well — so well, 
# in fact, that they over-learn patterns present in the training data 
# that end up being detrimental when predicting new data.
 
# We will tune the model hyperparameters to avoid overfitting. Tuning 
# the value of cost_complexity helps by pruning back our tree. It adds 
# a cost, or penalty, to error rates of more complex trees; a cost closer
# to zero decreases the number tree nodes pruned and is more likely to 
# result in an overfit tree. However, a high cost increases the number 
# of tree nodes pruned and can result in the opposite problem—an 
# underfit tree. Tuning tree_depth, on the other hand, helps by stopping 
# our tree from growing after it reaches a certain depth. We want to tune 
# these hyperparameters to find what those two values should be for our
# model to do the best job predicting image segmentation.

# We use the training data for tuning the model.












