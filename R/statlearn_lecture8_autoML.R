# Statistical Learning ----

# Lecture 8: Automatic ML -------------------------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - H2O
# - AutoML
# - Tidymodels H2O integration



# H2O - Framework ---------------------------------------------------------

# H2O AI
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html
# Algorithms
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html

# H2O is an “in-memory platform”. Saying that it’s in-memory means that the 
# data being used is loaded into main memory (RAM). Reading from main memory, 
# (also known as primary memory) is typically much faster than secondary memory 
# (such as a hard drive).

# H2O is a “platform.” A platform is software which can be used to build 
# something – in this case, machine learning models.
# Putting this togther we now know that H2O is an in-memory environment for 
# building machine learning models.

# H2O is Distributed and Scalable. H2O can be run on a cluster. Hadoop is an
# example of a cluster which can run H2O. H2O is said to be distributed because 
# your object can be spread amongst several nodes in your cluster. H2O does 
# this by using a Distributed Key Value (DKV). You can read more about it here, 
# but essentially what this means, is that any object you create in H2O can be 
# distributed amongst several nodes in the cluster.
# The key-value part of DKV means that when you load data into H2O, you get back 
# a key into a hashmap containing your (potentially distributed) object.

# How H2O Runs Under the Hood. We spoke earlier about H2O being a platform. 
# It’s important to distinguish between the R interface for H2O, and H2O itself.
# H2O can exist perfectly fine without R. H2O is just a .jar which can be run 
# on its own. If you don’t know (or particularly care) what a .jar is – just 
# think of it as Java code packaged with all the stuff you need in order to run
# it.

# When you start H2O, you actually create a server which can respond to REST 
# calls. Again, you don’t really need to know how REST works in order to use 
# H2O. But if you do care, just know that you can use any HTTP client to speak 
# with an H2O instance.
# R is just a client interfact for H2O. All the R functions you call when working
# with H2O are actually calling H2O using a REST API (a JSON POST request) under 
# the hood. The Python H2O library, as well as the Flow UI, interface with H2O 
# in a similar way. If this is all very confusing just think about it like this:
# 	you use R to send commands to H2O. You could equally well just use Flow or 
# Python to send commands.


# * Initialize H2O --------------------------------------------------------

# Dependency on JAVA
# Possible problems related to initialization of H2O from R / Python API:
# - Old JAVA version
# - root privileges
# - JAVA 32bit installed and not 64bit
# - JAVA_HOME env variable not set, Sys.getenv('JAVA_HOME')
# Solutions:
# - https://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/index.html
# - https://docs.h2o.ai/h2o/latest-stable/h2o-docs/faq/java.html
# - https://stackoverflow.com/questions/3892510/set-environment-variables-for-system-in-r
# - Sys.setenv('JAVA_HOME')

# Common steps:
# 1) Uninstall H2O
# if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
# if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# 2) Install the latest version of JAVA
# https://www.oracle.com/technetwork/java/javase/downloads/index.html
# 3) Install H2O again
# install.packages("h2o", type = "source", repos = (c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
# library(h2o)
# 4) Set the JAVA_HOME env variable
# Sys.getenv('JAVA_HOME')
# Sys.setenv(JAVA_HOME="/usr/lib/jvm/jdk-17/")
# Sys.getenv('JAVA_HOME')

library(h2o)

Sys.setenv(JAVA_HOME = "/usr/lib/jvm/jdk-17/")
h2o.init()
h2o.clusterInfo()


# * Data Conversion -------------------------------------------------------

source("R/utils.R")
source("R/packages.R")

# We use again the Canadian Wind Turbine Database from Natural Resources Canada.
wind_raw <- read_csv("data/wind.csv")
wind <-	wind_raw %>%
	dplyr::select(
		province_territory, 
		total_project_capacity_mw,
		turbine_rated_capacity_kw = turbine_rated_capacity_k_w,
		rotor_diameter_m,
		hub_height_m,
		year = commissioning_date
	) %>%
	group_by(province_territory) %>%
	mutate(
		year = as.numeric(year),
		province_territory = case_when(
			n() < 50 ~ "Other",
			TRUE ~ province_territory
		) %>% as.factor()
	) %>%
	dplyr::filter(!is.na(year)) %>%
	ungroup() %>% 
	drop_na()
wind

# Convert data into H2O frame
wind_h2o <- as.h2o(wind)
wind_h2o

h2o.ls()
h2o.describe(wind_h2o)


# * Data Splitting --------------------------------------------------------

# First off, we’ll split our data into a training set and a test set.
# h2o.splitFrame() uses approximate splitting. That is, it won’t split the 
# data into an exact 80%-20% split. Setting the seed allows us to create 
# reproducible results.

splits <- h2o.splitFrame(data = wind_h2o, ratios = c(0.8), seed = 123) # partition data into 80% and 20% chunks

train <- splits[[1]]
test <- splits[[2]]

# We can use h2o.nrow() to check the number of rows in our train and test sets.
h2o.nrow(train)
h2o.nrow(test)


# * H2O Modelling ---------------------------------------------------------

# We’ve got our H2O instance up and running, with some data in it. Let’s go 
# ahead and do some machine learning.

# Y and X variable names
y <- "turbine_rated_capacity_kw"
x <- setdiff(names(wind_h2o), y)


# ** GLM ------------------------------------------------------------------

h2o_glm <- h2o.glm(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	model_id = "glm",
	seed = 123,
	keep_cross_validation_predictions = TRUE
)

h2o.performance(h2o_glm, test) 
h2o.predict(h2o_glm, test)


# ** Random Forest --------------------------------------------------------

h2o_rf <- h2o.randomForest(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	model_id = "rf",
	seed = 123,
	keep_cross_validation_predictions = TRUE
)

h2o.performance(h2o_rf, test) 
h2o.predict(h2o_rf, test)


# ** GBM ------------------------------------------------------------------

h2o_gbm <- h2o.gbm(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	model_id = "gbm",
	seed = 123,
	keep_cross_validation_predictions = TRUE
)

h2o.performance(h2o_gbm, test) 
h2o.predict(h2o_gbm, test)


# ** XGBoost --------------------------------------------------------------

h2o_xgb <- h2o.xgboost(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	model_id = "xgb",
	seed = 123, 
	keep_cross_validation_predictions = TRUE
)

h2o.performance(h2o_xgb, test) 
h2o.predict(h2o_xgb, test)


# ** Neural Networks ------------------------------------------------------

h2o_nnet <- h2o.deeplearning(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	model_id = "nnet",
	seed = 123, 
	keep_cross_validation_predictions = TRUE
)

h2o.performance(h2o_nnet, test) 
h2o.predict(h2o_nnet, test)


# ** Stacking Ensemble ----------------------------------------------------

h2o_stack <- h2o.stackedEnsemble(
	x = x, y = y,
	training_frame = train,
	model_id = "stack",
	base_models = c(h2o_rf, h2o_xgb, h2o_nnet),
	metalearner_algorithm = "AUTO",
	metalearner_nfolds = 5,
	seed = 123
)

h2o.performance(h2o_stack, test) 
h2o.predict(h2o_stack, test)


# * Evaluation ------------------------------------------------------------

h2o_models <- list(
	"glm" = h2o_glm,
	"rf" = h2o_rf,
	"gbm" = h2o_gbm,
	"xgb" = h2o_xgb,
	"nnet" = h2o_nnet,
	"stack" = h2o_stack
)

map_df(h2o_models, evaluate_h2o, metric = "RMSE") %>% 
	pivot_longer(cols = everything()) %>% 
	set_names(c("Method", "RMSE"))


# * Shout-down H2O --------------------------------------------------------

h2o.shutdown(prompt = FALSE)



# AutoML with H2O ---------------------------------------------------------

# Although H2O has made it easy for non-experts to experiment with machine 
# learning, there is still a fair bit of knowledge and background in data 
# science that is required to produce high-performing machine learning models. 
# Deep Neural Networks in particular are notoriously difficult for a non-expert 
# to tune properly. In order for machine learning software to truly be accessible 
# to non-experts, we have designed an easy-to-use interface which automates the
# process of training a large selection of candidate models. H2O’s AutoML can
# also be a helpful tool for the advanced user, by providing a simple wrapper 
# function that performs a large number of modeling-related tasks that would 
# typically require many lines of code, and by freeing up their time to focus 
# on other aspects of the data science pipeline tasks such as data-preprocessing,
# feature engineering and model deployment.

# H2O’s AutoML can be used for automating the machine learning workflow, which 
# includes automatic training and tuning of many models within a user-specified 
# time-limit.

# The H2O AutoML interface is designed to have as few parameters as possible so 
# that all the user needs to do is point to their dataset, identify the response 
# column and optionally specify a time constraint or limit on the number of total
# models trained.
 
# In both the R and Python API, AutoML uses the same data-related arguments, x,
# y, training_frame, validation_frame, as the other H2O algorithms. Most of the
# time, all you’ll need to do is specify the data arguments. You can then 
# configure values for max_runtime_secs and/or max_models to set explicit time 
# or number-of-model limits on your run.

h2o.init()
wind_h2o <- as.h2o(wind)
splits <- h2o.splitFrame(data = wind_h2o, ratios = c(0.8), seed = 123) # partition data into 80% and 20% chunks
train <- splits[[1]]
test <- splits[[2]]
y <- "turbine_rated_capacity_kw"
x <- setdiff(names(wind_h2o), y)


# * AutoML Estimation -----------------------------------------------------

# - DRF (This includes both the Distributed Random Forest (DRF) and Extremely 
#   Randomized Trees (XRT) models. Refer to the Extremely Randomized Trees 
#   section in the DRF chapter and the histogram_type parameter description 
#   for more information.)
# - GLM (Generalized Linear Model with regularization)
# - XGBoost (XGBoost GBM)
# - GBM (H2O GBM)
# - DeepLearning (Fully-connected multi-layer artificial neural network)
#   StackedEnsemble (Stacked Ensembles, includes an ensemble of all the base 
#   models and ensembles using subsets of the base models)

# Run AutoML for 20 base models
h2o_auto <- h2o.automl(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	max_models = 5,
	# max_runtime_secs = 30,
	# max_runtime_secs_per_model = 30,
	# exclude_algos = c("DeepLearnig"),
	# include_algos = c("DRF")
	seed = 123
)
h2o_auto


# * Extract Models --------------------------------------------------------

# AutoML leaderboard
lb <- h2o.get_leaderboard(object = h2o_auto, extra_columns = "ALL")
print(lb, n = nrow(lb)) # 10 models + 11 stacks

# AutoML Best Model
h2o_best <- h2o.get_best_model(h2o_auto)
h2o_best

h2o.get_best_model(h2o_auto, criterion = "mae") # get the best model using logloss sort metric
h2o.get_best_model(h2o_auto, algorithm = "xgboost") # get the best XGBoost model using default sort metric
h2o.get_best_model(h2o_auto, algorithm = "xgboost", criterion = "mae") # get the best XGBoost model, ranked by logloss


# * Evaluate Results ------------------------------------------------------

h2o.performance(h2o_best, test)
evaluate_h2o(h2o_best, "RMSE")

# shout-down H2O 
h2o.shutdown(prompt = FALSE)


# * Tuning with H2O -------------------------------------------------------




# Tidymodels Interface to H20 ---------------------------------------------

# https://github.com/stevenpawley/h2oparsnip

# h2oparsnip provides a set of wrappers to bind h2o algorthms with the 
# 'parsnip' package.

# This package is early in development. Currently the following h2o algorithms 
# are implemented:
 	
# 	- h2o.naiveBayes engine added to naive_Bayes specification
# 	- h2o.glm engine added to multinom_reg, logistic_reg and linear_reg model specifications
#   - h2o.randomForest engine added to parsnip::rand_forest model specification
#   - h2o.gbm engine added to parsnip::boost_tree model specification
#   - h2o.deeplearning engine added to parsnip::mlp model specification
#   - a new model, automl
#   - h2o.rulefit engine added to parsnip::rule_fit

# The package currently is based on the concept of using h2o as a disposable 
# backend, using h2o as a drop-in replacement for the traditionally used 
# 'engines' within the parsnip package. However, performing tasks such as 
# hyperparameter tuning via the 'tune' packge will be less efficient if 
# working on a remote cluster than using h2o directly because data is being 
# sent back and forth.

# h2oparsnip also does not provide any management of the h2o cluster. If lots 
# of models are being run then available memory within the cluster may be 
# exhausted. Currently this has to be managed using the commands in the h2o 
# package.

# The package is not yet on CRAN and can be installed with
devtools::install_github("stevenpawley/h2oparsnip")

h2o.init()
library(h2oparsnip)


# * Data ------------------------------------------------------------------

# split into training and testing sets
set.seed(123)
splits <- initial_split(wind)

# use a 5-fold cross-validation
set.seed(123)
folds <- rsample::vfold_cv(training(splits), v = 5)

# set up a basic recipe
rcp_spec <- recipe(turbine_rated_capacity_kw ~ ., data = training(splits)) %>%
	step_dummy(all_nominal()) %>%
	step_zv(all_predictors())

# for simplicity we use just rmse
metric <- metric_set(rmse)


# ** GLM ------------------------------------------------------------------

# Engine
model_spec_glm <- linear_reg(
	mode = "regression",
	penalty = tune(),
	mixture = tune()
) %>%
	set_engine("h2o")

# Workflow
wrkfl_glm <- workflow() %>% 
	add_recipe(rcp_spec) %>% 
	add_model(model_spec_glm)

# Grid
set.seed(123)
grid_glm <- grid_regular(
	penalty(),
	mixture(),
	levels = 3
)

# Fit
set.seed(123)
wrkfl_fit_glm <- wrkfl_glm %>%  
	tune_grid(
		resamples = folds,
		grid = grid_glm,
		metrics = metric
	)
wrkfl_fit_glm

# A problem with using tune::tune_grid is that performance is reduced because 
# the data for every tuning hyperparameter iteration and resampling is moved 
# from R to the h2o cluster. To minimize this, the tune_grid_h2o function can
# be used to tune model arguments, as a near drop-in replacement:
	
wrkfl_fit_glm_h2otune <- wrkfl_glm %>% 
	tune_grid_h2o(
	  resamples = folds,
	  grid = grid_glm,
	  metrics = metric
	)
wrkfl_fit_glm_h2otune

# Currently, tune_grid_h2o can only tune model parameters and does not handle 
# recipes with tunable parameters. tune_grid_h2o moves the data to the h2o 
# cluster only once, i.e. the complete dataset specified by the resamples 
# argument is moved to the cluster, and then the equivalent h2o.frame is split 
# based on the row indices in the resampling object, and the h2o::h2o.grid 
# function is used for tuning on the h2o frames. To avoid repeatedly moving 
# predictions back from h2o to R, all metrics are also calculated on the cluster. 
# This restricts the range of metrics to what is available in h2o (tune_grid_h2o
# maps yardstick metrics to their h2o equivalents). The available metrics are 
# listed in the tune_grid_h2o help documentation. However, hyperparameter tuning 
# using tune_grid_h2o should be similarly performant as when using h2o directly.


# ** Random Forest --------------------------------------------------------

# Engine
model_spec_rf <- rand_forest(
	mode = "regression",
	mtry = tune(),
	min_n = tune(),
	trees = 1000
) %>%
	set_engine("h2o")

# Workflow
wrkfl_rf <- workflow() %>% 
	add_recipe(rcp_spec) %>% 
	add_model(model_spec_rf)

# Grid
set.seed(123)
grid_rf <- grid_regular(
	mtry(range = c(1, 20)),
	min_n(),
	levels = 3
)

# Fit
set.seed(123)
wrkfl_fit_rf_h2otune <- wrkfl_rf %>% 
	tune_grid_h2o(
		resamples = folds,
		grid = grid_rf,
		metrics = metric
	)
wrkfl_fit_rf_h2otune


# ** GBM ------------------------------------------------------------------

# Engine
model_spec_gbm <- boost_tree(
	mode = "regression",
	mtry = tune(),
	min_n = tune(),
	trees = 1000
) %>%
	set_engine("h2o")

# Workflow
wrkfl_gbm <- workflow() %>% 
	add_recipe(rcp_spec) %>% 
	add_model(model_spec_gbm)

# Grid
set.seed(123)
grid_gbm <- grid_regular(
	mtry(range = c(1, 20)),
	min_n(),
	levels = 3
)

# Fit
set.seed(123)
wrkfl_fit_gbm_h2otune <- wrkfl_gbm %>% 
	tune_grid_h2o(
		resamples = folds,
		grid = grid_gbm,
		metrics = metric
	)
wrkfl_fit_gbm_h2otune


# ** Neural Networks ------------------------------------------------------

# Engine
model_spec_nnet <- mlp(
	mode = "regression",
	epochs = tune(),
	hidden_units = tune()
) %>%
	set_engine("h2o")

# Workflow
wrkfl_nnet <- workflow() %>% 
	add_recipe(rcp_spec) %>% 
	add_model(model_spec_nnet)

# Grid
set.seed(123)
grid_nnet <- grid_regular(
	epochs(),
	hidden_units(),
	levels = 3
)

# Fit
set.seed(123)
wrkfl_fit_nnet_h2otune <- wrkfl_nnet %>% 
	tune_grid_h2o(
		resamples = folds,
		grid = grid_nnet,
		metrics = metric
	)
wrkfl_fit_nnet_h2otune


# ** AutoML ---------------------------------------------------------------

# Engine
model_spec_automl <- automl(mode = "regression") %>%
	set_engine("h2o")

# Workflow
wrkfl_automl <- workflow() %>% 
	add_recipe(rcp_spec) %>% 
	add_model(model_spec_automl)

# Fit
# NOT TO BE RUN!!!!!!!!!!!!!!! REALLY SLOW
# set.seed(123)
# wrkfl_fit_automl_h2otune <- wrkfl_automl %>% 
# 	fit_resamples(
# 		resamples = folds,
# 		metrics = metric
# 	)
# wrkfl_fit_automl_h2otune

# NOT TO BE RUN!!!!!!!!!!!!!!! REALLY SLOW
# set.seed(123)
# fit_automl <- h2o_automl_train(turbine_rated_capacity_kw ~ ., data = wind)
# fit_automl


# * Evaluate Results ------------------------------------------------------

wrkfls <- list(
	"glm" = wrkfl_glm,
	"rf" = wrkfl_rf,
	"gbm" = wrkfl_gbm,
	"nnet" = wrkfl_nnet
)

model_results <- list(
	"glm" = wrkfl_fit_glm_h2otune,
	"rf" = wrkfl_fit_rf_h2otune,
	"gbm" = wrkfl_fit_gbm_h2otune,
	"nnet" = wrkfl_fit_nnet_h2otune
)

model_results %>% map(collect_metrics) # validation set metrics
best_models <- model_results %>% map(select_best, metric = "rmse")

wrkfls_fit_final <- map2(wrkfls, best_models, finalizing_and_fitting) 

wrkfl_fit_final %>%	
	map(collect_metrics) %>% 
	map2(
		names(wrkfls), 
		~ mutate(.x, Member = .y) %>% 
			dplyr::filter(.metric == "rmse") %>% 
			dplyr::select(Member, .metric, .estimate) %>% 
			set_names(c("Member", "Metric", "Estimate"))
	) %>%
	bind_rows()



# Regression - The House Prices Dataset -----------------------------------

source("R/utils.R")
source("R/packages.R")
library(h2o)


# * Data ------------------------------------------------------------------

# * Load Data
artifacts_list <- read_rds("artifacts/artifacts_list.rds")
data <- artifacts_list$reg_data

# * Feature Engineering via recipe 
data_feats <- recipe(SalePrice ~ ., data = data) %>% 
	step_dummy(all_nominal(), -all_outcomes()) %>% 
	prep() %>% 
	juice()
data_feats %>% glimpse()


# * AutoML H2O ------------------------------------------------------------

# Initialize H2O
Sys.setenv(JAVA_HOME = "/usr/lib/jvm/jdk-17/")
h2o.init()

# Data Conversion
data_h2o <- as.h2o(data_feats)
splits <- h2o.splitFrame(data = data_h2o, ratios = c(0.8), seed = 123) # partition data into 80% and 20% chunks
train <- splits[[1]]
test <- splits[[2]]
y <- "SalePrice"
x <- setdiff(names(data_h2o), y)

# Run AutoML for 20 base models
h2o_auto <- h2o.automl(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	max_models = 20,
	# max_runtime_secs = 30,
	max_runtime_secs_per_model = 20,
	# exclude_algos = c("DeepLearnig"),
	# include_algos = c("DRF")
	seed = 123
)
h2o_auto

# AutoML leaderboard
lb <- h2o.get_leaderboard(object = h2o_auto)
print(lb, n = nrow(lb)) 

# AutoML Best Model
h2o_best <- h2o.get_best_model(h2o_auto)
h2o_best

# Evaluate Results 
h2o.performance(h2o_best, test)
evaluate_h2o(h2o_best, "RMSE")

# shout-down H2O 
h2o.shutdown(prompt = FALSE)



# Classification - The House Prices Dataset -------------------------------

source("R/utils.R")
source("R/packages.R")
library(h2o)


# * Data ------------------------------------------------------------------

# * Load Data
artifacts_list <- read_rds("artifacts/artifacts_list.rds")
data <- artifacts_list$class_data

# * Feature Engineering via recipe 
data_feats <- recipe(Value ~ ., data = data) %>% 
	step_dummy(all_nominal(), -all_outcomes()) %>% 
	step_mutate(Value = as.factor(as.character(Value))) %>% 
	prep() %>% 
	juice()
data_feats %>% glimpse()


# * AutoML H2O ------------------------------------------------------------

# Initialize H2O
Sys.setenv(JAVA_HOME = "/usr/lib/jvm/jdk-17/")
h2o.init()

# Data Conversion
data_h2o <- as.h2o(data_feats)
splits <- h2o.splitFrame(data = data_h2o, ratios = c(0.8), seed = 123) # partition data into 80% and 20% chunks
train <- splits[[1]]
test <- splits[[2]]
y <- "Value"
x <- setdiff(names(data_h2o), y)

# for classification, response should be a factor
# train[, y] <- as.factor(train[, y])
# test[, y] <- as.factor(test[, y])

# Run AutoML for 20 base models
h2o_auto <- h2o.automl(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	# max_models = 20,
	max_runtime_secs = 30,
	# max_runtime_secs_per_model = 20,
	# exclude_algos = c("DeepLearnig"),
	# include_algos = c("DRF")
	seed = 123
)
h2o_auto

# AutoML leaderboard
lb <- h2o.get_leaderboard(object = h2o_auto)
print(lb, n = nrow(lb)) 

# AutoML Best Model
h2o_best <- h2o.get_best_model(h2o_auto)
h2o_best

# Evaluate Results 
h2o.performance(h2o_best, test)
evaluate_h2o(h2o_best, "AUC")

# shout-down H2O 
h2o.shutdown(prompt = FALSE)
