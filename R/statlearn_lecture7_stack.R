# Statistical Learning ----

# Lecture 7: Ensemble Learning --------------------------------------------
# Marco Zanotti

# Goals:
# - Ensembling
# - Simple Ensembles (Averages)
# - Stacking



# Stacks ------------------------------------------------------------------

# https://stacks.tidymodels.org/index.html

# stacks is an R package for model stacking that aligns with the tidymodels. 
# Model stacking is an ensembling method that takes the outputs of many models 
# and combines them to generate a new model—referred to as an ensemble in this 
# package—that generates predictions informed by each of its members.
 
# The process goes something like this:
#  1. Define candidate ensemble members using functionality from rsample, 
#     parsnip, workflows, recipes, and tune
#  2. Initialize a data_stack object with stacks()
#  3. Iteratively add candidate ensemble members to the data_stack with add_candidates()
#  4. Evaluate how to combine their predictions with blend_predictions()
#  5. Fit candidate ensemble members with non-zero stacking coefficients with fit_members()
#  6. Predict on new data with predict()

# stacks is generalized with respect to:
#  * Model type: Any model type implemented in parsnip or extension packages 
#    is fair game to add to a stacks model stack
#  * Cross-validation scheme: Any resampling algorithm implemented in rsample 
#    or extension packages is fair game for resampling data for use in training
#    a model stack.
#  * Error metric: Any metric function implemented in yardstick or extension 
#    packages is fair game for evaluating model stacks and their members. 

# stacks uses a regularized linear model to combine predictions from ensemble 
# members, though this model type is only one of many possible learning 
# algorithms that could be used to fit a stacked ensemble model. For 
# implementations of additional ensemble learning algorithms, check out h2o and 
# SuperLearner.



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")



# Data --------------------------------------------------------------------

# Natural Resources Canada has published the Canadian Wind Turbine Database, 
# which contains the precise latitude and longitude of every turbine, along 
# with details like its dimensions, its power output, its manufacturer and 
# the date it was commissioned.
 
# Making use of the stacks package, we’ll build a stacked ensemble model to 
# predict turbine capacity in kilowatts based on turbine characteristics.

wind_raw <- read_csv("data/wind.csv")

# First thing, we’ll subset down to variables that we’ll use in the stacked 
# ensemble model. For the most part, I’m just getting rid of ID variables 
# and qualitative variables with a lot of levels.

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
		)
	) %>%
	dplyr::filter(!is.na(year)) %>%
	ungroup()
wind


# * Data Splitting --------------------------------------------------------

# split into training and testing sets
set.seed(123)
splits <- initial_split(wind)


# * Cross-Validation ------------------------------------------------------

# use a 5-fold cross-validation (5 just for computing time)
set.seed(123)
folds <- rsample::vfold_cv(training(splits), v = 5)


# * Recipes ---------------------------------------------------------------

# set up a basic recipe
rcp_spec <- recipe(turbine_rated_capacity_kw ~ ., data = training(splits)) %>%
	step_dummy(all_nominal()) %>%
	step_zv(all_predictors())


# * Metric ----------------------------------------------------------------

# For simplicity we use just rmse
metric <- metric_set(rmse)



# Base Models -------------------------------------------------------------

# At the highest level, ensembles are formed from model definitions. In the 
# stacks package, model definitions are an instance of a minimal workflow, 
# containing a model specification (as defined in the parsnip package) and, 
# optionally, a preprocessor (as defined in the recipes package). 
# Model definitions specify the form of candidate ensemble members.

# To be used in the same ensemble, each of these model definitions must share 
# the same resample. This rsample set object, when paired with the model 
# definitions, can be used to generate the tuning/fitting results objects 
# for the candidate ensemble members with tune.

# We’ll define three different model definitions to try to predict turbine 
# capacity—a linear model, a spline model (with hyperparameters to tune), and 
# a support vector machine model (again, with hyperparameters to tune).

# Tuning and fitting results for use in ensembles need to be fitted with the 
# control arguments save_pred = TRUE and save_workflow = TRUE—these settings
# ensure that the assessment set predictions, as well as the workflow used to 
# fit the resamples, are stored in the resulting object. For convenience, 
# stacks supplies some control_stack_*() functions to generate the appropriate
# objects for you.


# * Linear Regression -----------------------------------------------------

# Engine
model_spec_lm <- linear_reg() %>%
	set_engine("lm")

# Workflow
wrkfl_lm <- workflow() %>% 
	add_recipe(rcp_spec) %>% 
	add_model(model_spec_lm)

# Fit 
set.seed(123)
wrkfl_fit_lm <- wrkfl_lm %>% 
	fit_resamples(
		resamples = folds,
		metrics = metric,
		control = control_stack_resamples() # control options to save pred and workflows to be used by the meta-learner
	)


# * Spline Regression -----------------------------------------------------

# Spline Recipe
rcp_spec_spline <- rcp_spec %>%
	step_ns(rotor_diameter_m, deg_free = tune::tune("length")) # tune a recipe parameter!!!!

# Workflow
wrkfl_spline <- workflow() %>% 
	add_recipe(rcp_spec_spline) %>% 
	add_model(model_spec_lm)

# Fit
# tune deg_free and fit to the 5-fold cv
set.seed(123)
wrkfl_fit_spline <- wrkfl_spline %>%  
	tune_grid(
		resamples = folds,
		metrics = metric,
		control = control_stack_grid()
	)


# * SVM -------------------------------------------------------------------

# Engine
model_spec_svm <- svm_rbf(
	mode = "regression",
	cost = tune(), 
	rbf_sigma = tune()
) %>%
	set_engine("kernlab")

# Workflow
wrkfl_svm <- workflow() %>% 
	add_recipe(rcp_spec) %>% 
	add_model(model_spec_svm)

# Fit
# tune cost and rbf_sigma and fit to the 5-fold cv
set.seed(123)
wrkfl_fit_svm <- wrkfl_svm %>% 
	tune_grid(
		resamples = folds, 
		grid = 5,
		control = control_stack_grid()
	)



# Stacking Ensemble -------------------------------------------------------

# * Data Stack ------------------------------------------------------------

# Candidate members first come together in a data_stack object through the 
# add_candidates() function. Principally, these objects are just tibbles, 
# where the first column gives the true outcome in the assessment set, and 
# the remaining columns give the predictions from each candidate ensemble 
# member. (When the outcome is numeric, there’s only one column per candidate 
# ensemble member. Classification requires as many columns per candidate as 
# there are levels in the outcome variable.) 

# The first step to building a data stack is the initialization step. The 
# stacks() function creates a basic structure that the object will be built on 
# top of using %>%.

data_stack <- stacks() %>%
	add_candidates(wrkfl_fit_lm) %>%
	add_candidates(wrkfl_fit_spline) %>%
	add_candidates(wrkfl_fit_svm)
data_stack


# * Model Stack -----------------------------------------------------------

# Then, the data stack can be evaluated using blend_predictions() to determine 
# to how best to combine the outputs from each of the candidate members.

# The outputs of each member are likely highly correlated. Thus, depending on 
# the degree of regularization you choose, the coefficients for the inputs of 
# (possibly) many of the members will zero out—their predictions will have no 
# influence on the final output, and those terms will thus be thrown out.

model_stack <- data_stack %>%	blend_predictions()
model_stack


# * Ensembling ------------------------------------------------------------

# The blend_predictions function determines how member model output will 
# ultimately be combined in the final prediction, and is how we’ll calculate 
# our stacking coefficients. Now that we know how to combine our model output, 
# we can fit the models that we now know we need on the full training set. 
# Any candidate ensemble member that has a stacking coefficient of zero 
# doesn’t need to be refitted!

fit_model_stack <- model_stack %>% fit_members()
fit_model_stack

# Now that we’ve fitted the needed ensemble members, our model stack is ready 
# to go! For the most part, a model stack is just a list that contains a 
# bunch of ensemble members and instructions on how to combine their predictions.



# Evaluation --------------------------------------------------------------

# To make sure that we have the right trade-off between minimizing the number 
# of members and optimizing performance, we can use the autoplot() method.
autoplot(fit_model_stack)
autoplot(fit_model_stack, type = "weights")
# autoplot(fit_model_stack, type = "members")

# Let’s check out how well the model stack performs! Predicting on new data:
test_pred <- bind_cols(
	testing(splits) %>%	select(turbine_rated_capacity_kw),
	predict(fit_model_stack, testing(splits))
)
test_pred

rmse(test_pred, truth = turbine_rated_capacity_kw, estimate = .pred) %>% 
	mutate(member = "stack")

# We can use the type = "members" argument to generate predictions from each 
# of the ensemble members.

test_pred_member <- bind_cols(
	testing(splits) %>%	select(turbine_rated_capacity_kw),
	predict(fit_model_stack, testing(splits), members = TRUE)
)
test_pred_member

colnames(test_pred_member) %>%
	map_dfr(rmse, truth = turbine_rated_capacity_kw, data = test_pred_member) %>%
	mutate(
		member = colnames(test_pred_member),
		member = ifelse(member == ".pred", "stack", member)
	)



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

# * Cross-Validation
set.seed(123)
folds <- vfold_cv(training(splits), v = 5)


# * Base Models -----------------------------------------------------------

# * ELASTIC NET
model_spec_elanet <- linear_reg(
	mode = "regression",
	penalty = tune(),
	mixture = tune()
) %>%
	set_engine("glmnet")

wrkfl_elanet <- workflow() %>%
	add_model(model_spec_elanet) %>%
	add_recipe(rcp_spec)

set.seed(123)
model_res_elanet <- wrkfl_elanet %>% 
	tune_grid(
		resamples = folds,
		grid = 5,
		control = control_stack_grid()
	)


# * XGBOOST
model_spec_xgb <- boost_tree(
	mode = "regression",
	mtry = tune(),
	min_n = tune(),
	trees = 1000
) %>%
	set_engine("xgboost")

wrkfl_xgb <- workflow() %>%
	add_model(model_spec_xgb) %>%
	add_recipe(rcp_spec) 

set.seed(123)
model_res_xgb <- wrkfl_xgb %>% 
	tune_grid(
		resamples = folds,
		grid = 5,
		control = control_stack_grid()
	)


# * NEURAL NETWORK
model_spec_nnet <- mlp(
	mode = "regression",
	hidden_units = 10,
	penalty = tune(),
	epochs = tune()
) %>%
	set_engine("nnet")

wrkfl_nnet <- workflow() %>%
	add_model(model_spec_nnet) %>%
	add_recipe(rcp_spec)

set.seed(123)
model_res_nnet <- wrkfl_nnet %>% 
	tune_grid(
		resamples = folds,
		grid = 5,
		control = control_stack_grid()
	)


# * Stacking Ensembles ----------------------------------------------------

# * Data Stack
data_stack <- stacks() %>%
	add_candidates(model_res_elanet) %>%
	add_candidates(model_res_xgb) %>%
	add_candidates(model_res_nnet)
data_stack

# * Model Stack
model_stack <- data_stack %>%	blend_predictions()
model_stack

# * Ensembling
fit_model_stack <- model_stack %>% fit_members()
fit_model_stack


# * Evaluation ------------------------------------------------------------

autoplot(fit_model_stack)
autoplot(fit_model_stack, type = "weights")

fit_model_stack %>% calibrate_evaluate_stacks(y = "SalePrice", mode = "regression")


# * Comparison with simple Ensembles --------------------------------------

wrkfls <- list(
	"elanet" = wrkfl_elanet, 
	"xgb" = wrkfl_xgb,
	"nnet" = wrkfl_nnet
)

model_results <- list(
	"elanet" = model_res_elanet, 
	"xgb" = model_res_xgb,
	"nnet" = model_res_nnet
)

# Average Ensemble
simple_ensemble(model_results, wrkfls, y = "SalePrice", mode = "regression", ensemble_fun = mean)

# Median Ensemble
simple_ensemble(model_results, wrkfls, y = "SalePrice", mode = "regression", ensemble_fun = median)



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

# * Cross-Validation
set.seed(123)
folds <- vfold_cv(training(splits), v = 5)


# * Base Models -----------------------------------------------------------

# * NAIVE BAYES
model_spec_nb <- naive_Bayes() %>%
	set_engine("klaR")

wrkfl_nb <- workflow() %>%
	add_model(model_spec_nb) %>%
	add_recipe(rcp_spec)

set.seed(123)
model_res_nb <- wrkfl_nb %>% 
	fit_resamples(
		resamples = folds,
		control = control_stack_resamples()
	)


# * KNN
model_spec_knn <- nearest_neighbor(
	mode = "classification",
	neighbors = tune(),
	dist_power = tune(),
	weight_func = "optimal"
) %>%
	set_engine("kknn")

wrkfl_knn <- workflow() %>%
	add_model(model_spec_knn) %>%
	add_recipe(rcp_spec) 

set.seed(123)
model_res_knn <- wrkfl_knn %>% 
	tune_grid(
		resamples = folds,
		grid = 5,
		control = control_stack_grid()
	)


# * RANDOM FOREST
model_spec_rf <- rand_forest(
	mode = "classification",
	mtry = tune(),
	min_n = tune(),
	trees = 1000
) %>%
	set_engine("ranger")

wrkfl_rf <- workflow() %>%
	add_model(model_spec_rf) %>%
	add_recipe(rcp_spec)

set.seed(123)
model_res_rf <- wrkfl_rf %>% 
	tune_grid(
		resamples = folds,
		grid = 5,
		control = control_stack_grid()
	)


# * Stacking Ensembles ----------------------------------------------------

# * Data Stack
data_stack <- stacks() %>%
	add_candidates(model_res_nb) %>%
	add_candidates(model_res_knn) %>%
	add_candidates(model_res_rf)
data_stack

# * Model Stack
model_stack <- data_stack %>%	blend_predictions()
model_stack

# * Ensembling
fit_model_stack <- model_stack %>% fit_members()
fit_model_stack


# * Evaluation ------------------------------------------------------------

autoplot(fit_model_stack)
autoplot(fit_model_stack, type = "weights")

fit_model_stack %>% calibrate_evaluate_stacks(y = "Value", mode = "classification")


# * Comparison with Simple Ensembles --------------------------------------

wrkfls <- list(
	"nb" = wrkfl_nb, 
	"knn" = wrkfl_knn,
	"rf" = wrkfl_rf
)

model_results <- list(
	"nb" = model_res_nb, 
	"knn" = model_res_knn,
	"rf" = model_res_rf
)

# Mode Ensemble
simple_ensemble(model_results, wrkfls, y = "Value", mode = "classification", ensemble_fun = stat_mode)

