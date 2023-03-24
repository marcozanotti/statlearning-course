# Statistical Learning ----

# Lecture 6: Hyperparameter Tuning ----------------------------------------
# Marco Zanotti

# Goals:
# - Tuning
# - Grids
# - Cross-Validation



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")



# Data --------------------------------------------------------------------

# In lecture 3, we introduced a data set of images of cells that were 
# labeled by experts as well-segmented (WS) or poorly segmented (PS). 
# We trained a random forest model to predict which images are segmented 
# well vs. poorly, so that a biologist could filter out poorly segmented 
# cell images in their analysis. We used resampling to estimate the 
# performance of our model on this data.

data(cells, package = "modeldata")


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


# * Cross-Validation ------------------------------------------------------

# Let’s create cross-validation folds for tuning

set.seed(123)
cell_folds <- vfold_cv(cell_train)



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


# * Engine ----------------------------------------------------------------

# Let’s start with the parsnip package, using a decision_tree() model 
# with the rpart engine. To tune the decision tree hyperparameters 
# cost_complexity and tree_depth, we create a model specification that 
# identifies which hyperparameters we plan to tune.

tree_spec <- decision_tree(
	mode = "classification",
	cost_complexity = tune(),
	tree_depth = tune()
) %>% 
	set_engine("rpart")
tree_spec


# * Grid Search -----------------------------------------------------------

# Think of tune() here as a placeholder. After the tuning process, 
# we will select a single numeric value for each of these hyperparameters. 
# For now, we specify our parsnip model object and identify the 
# hyperparameters we will tune().

# We can’t train this specification on a single data set (such as the 
# entire training set) and learn what the hyperparameter values should be, 
# but we can train many models using resampled data and see which models 
# turn out best. We can create a regular grid of values to try using 
# some convenience functions for each hyperparameter.

# There are different grid functions from the dials package that one
# can use, based on different approaches to chose parameters' values.

?grid_regular()
?grid_random()
?grid_max_entropy()
?grid_latin_hypercube()

# One can explore the different hyperparameter to tune via the args() 
# function, this allows to see which parsnip object arguments are available.
	
args(decision_tree)

# The default parameters' ranges of values may be extracted 
# with the corresponding functions.

cost_complexity()
tree_depth()

# Let's create a grid through with grid_regular()

set.seed(123)
tree_grid <- grid_regular(
	cost_complexity(),
	tree_depth(),
	levels = 5
)
tree_grid

# The function grid_regular() is from the dials package. It chooses 
# sensible values to try for each hyperparameter; here, we asked for 
# 5 of each. Since we have two to tune, grid_regular() returns 
# 5 × 5 = 25 different possible tuning combinations to try in a tidy 
# tibble format.


# * Tuning ----------------------------------------------------------------

# We are ready to tune! Let’s use tune_grid() to fit models at all the 
# different values we chose for each tuned hyperparameter. There are 
# several options for building the object for tuning:
# 	
#   * Tune a model specification along with a recipe or model, or 
#   * Tune a workflow() that bundles together a model specification and
#     a recipe or m/odel preprocessor.

# Here we use a workflow() with a straightforward formula; if this model 
# required more involved data preprocessing, we could use add_recipe() 
# instead of add_formula().

set.seed(123)
tree_wf <- workflow() %>%
	add_model(tree_spec) %>%
	add_formula(class ~ .)

tree_res <- tree_wf %>% 
	tune_grid(
		resamples = cell_folds,
		grid = tree_grid
	)
tree_res


# * Evaluating ------------------------------------------------------------

# Once we have our tuning results, we can both explore them through 
# visualization and then select the best result. The function 
# collect_metrics() gives us a tidy tibble with all the results. 
# We had 25 candidate models and two metrics, accuracy and roc_auc, 
# and we get a row for each .metric and model.

tree_res %>% collect_metrics()

# We might get more out of plotting these results

tree_res %>%
	collect_metrics() %>%
	mutate(tree_depth = factor(tree_depth)) %>%
	ggplot(aes(cost_complexity, mean, color = tree_depth)) +
	geom_line(size = 1.5, alpha = 0.6) +
	geom_point(size = 2) +
	facet_wrap(~ .metric, scales = "free", nrow = 2) +
	scale_x_log10(labels = scales::label_number()) +
	scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

# We can see that our “stubbiest” tree, with a depth of 1, is the 
# worst model according to both metrics and across all candidate 
# values of cost_complexity. Our deepest tree, with a depth of 15, 
# did better. However, the best tree seems to be between these 
# values with a tree depth of 4. The show_best() function shows us 
# the top 5 candidate models by default.
	
tree_res %>% show_best("accuracy", n = 3)

# We can also use the select_best() function to pull out the single 
# set of hyperparameter values for our best decision tree model.
	
best_tree <- tree_res %>%	select_best("accuracy")
best_tree

# These are the values for tree_depth and cost_complexity that
# maximize accuracy in this data set of cell images.


# * Re-fitting ------------------------------------------------------------

# We can update (or “finalize”) our workflow object tree_wf with the
# values from select_best().

final_wf <-	tree_wf %>% 
	finalize_workflow(best_tree)
final_wf

# Finally, let’s fit this final model to the training data and use our 
# test data to estimate the model performance we expect to see with new 
# data. We can use the function last_fit() with our finalized model; 
# this function fits the finalized model on the full training data set 
# and evaluates the finalized model on the testing data.

final_fit <- final_wf %>%
	last_fit(cell_split) 

final_fit %>%	collect_metrics()

final_fit %>%
	collect_predictions() %>% 
	roc_curve(class, .pred_PS) %>% 
	autoplot()

# The performance metrics from the test set indicate that we did not 
# overfit during our tuning procedure.

# The final_fit object contains a finalized, fitted workflow that you 
# can use for predicting on new data or further understanding the results. 
# You may want to extract this object, using one of the extract_ 
# helper functions.

extract_workflow(final_fit)

final_fit %>% 
	extract_workflow() %>% 
	extract_fit_engine() %>%
	rpart.plot::rpart.plot(roundint = FALSE)

# Perhaps we would also like to understand what variables are important 
# in this final model. We can use the vip package to estimate variable 
# importance based on the model’s structure.

final_fit %>% 
	extract_workflow() %>%
	extract_fit_parsnip() %>% 
	vip::vip()

# These are the automated image analysis measurements that are the most 
# important in driving segmentation quality predictions.



# Regression - The House Prices Dataset -----------------------------------

source("R/utils.R")
source("R/packages.R")

# * Data ------------------------------------------------------------------

# * Load Data
artifacts_list <- read_rds("artifacts/artifacts_list.rds")
data <- artifacts_list$reg_data

# * Train / Test Sets 
set.seed(123)
splits <- initial_split(data, prop = .8)

# * Recipes 
rcp_spec <- recipe(SalePrice ~ ., data = training(splits)) %>% 
	step_dummy(all_nominal(), -all_outcomes())
rcp_spec %>% prep() %>% juice() %>% glimpse()


# * Cross-Validation ------------------------------------------------------

set.seed(123)
folds <- vfold_cv(training(splits))


# * Hyperparameter Tuning -------------------------------------------------

# * Engine
model_spec <- linear_reg(
	mode = "regression",
	penalty = tune(),
	mixture = tune()
) %>%
	set_engine("glmnet")
model_spec

# * Workflow
wrkfl <- workflow() %>%
	add_model(model_spec) %>%
	add_recipe(rcp_spec)

# * Grid Search
args(linear_reg)
penalty()
mixture()

set.seed(123)
model_grid <- grid_regular(
	penalty(),
	mixture(),
	levels = 10
)
model_grid
model_grid %>% map(unique)

# * Tuning
set.seed(123)
model_res <- wrkfl %>% 
	tune_grid(
		resamples = folds,
		grid = model_grid
	)
model_res


# * Evaluation ------------------------------------------------------------

model_res %>% collect_metrics()
model_res %>% show_best("rmse", n = 3)

best_model <- model_res %>%	select_best("rmse")
best_model


# * Re-fitting ------------------------------------------------------------

wrkfl_fit_final <- wrkfl %>%	
	finalize_workflow(best_model) %>% 
  last_fit(splits) 

wrkfl_fit_final %>%	collect_metrics()
wrkfl_fit_final %>% collect_predictions()



# Classification - The House Prices Dataset -------------------------------

source("R/utils.R")
source("R/packages.R")

# * Data ------------------------------------------------------------------

# * Load Data
artifacts_list <- read_rds("artifacts/artifacts_list.rds")
data <- artifacts_list$class_data

# * Train / Test Sets
set.seed(123)
splits <- initial_split(data, prop = .8)

# * Recipes
rcp_spec <- recipe(Value ~ ., data = training(splits)) %>% 
	step_dummy(all_nominal(), -all_outcomes())
rcp_spec %>% prep() %>% juice() %>% glimpse()


# * Cross-Validation ------------------------------------------------------

set.seed(123)
folds <- vfold_cv(training(splits))


# * Hyperparameter Tuning -------------------------------------------------

# * Engine
model_spec <- boost_tree(
	mode = "classification",
	mtry = tune(),
	trees = 1000,
	min_n = tune(),
	tree_depth = tune(),
	learn_rate = 0.0005
) %>%
	set_engine("xgboost")
model_spec

# * Workflow
wrkfl <- workflow() %>%
	add_model(model_spec) %>%
	add_recipe(rcp_spec)

# * Grid Search
args(boost_tree)
mtry() # must specify the range
min_n()
tree_depth()

set.seed(123)
model_grid <- grid_regular(
	mtry(range = c(1, 40)),
	min_n(),
	tree_depth(),
	levels = 10
)
model_grid
model_grid %>% map(unique)

# * Tuning
set.seed(123)
model_res <- wrkfl %>% 
	tune_grid(
		resamples = folds,
		grid = model_grid
	)
model_res


# * Evaluation ------------------------------------------------------------

model_res %>% collect_metrics()
model_res %>% show_best("roc_auc", n = 3)

best_model <- model_res %>%	select_best("roc_auc")
best_model


# * Re-fitting ------------------------------------------------------------

wrkfl_fit_final <-	wrkfl %>%	
	finalize_workflow(best_model) %>% 
	last_fit(splits) 

wrkfl_fit_final %>%	collect_metrics()
wrkfl_fit_final %>% collect_predictions()

