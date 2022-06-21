# Statistical Learning ----

# Lecture 5: Classification with ML Models --------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - Naive Bayes
# - Logistic Regression
# - Elastic Net
# - SVM
# - KNN
# - Decision Tree
# - Bagging
# - Random Forest
# - XGBoost, Light GBM
# - Neural Networks



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")

# packages treesnip and catboost have to be installed from dev versions
# remotes::install_github("curso-r/treesnip")



# Data --------------------------------------------------------------------

artifacts_list <- read_rds("artifacts/artifacts_list.rds")
data <- artifacts_list$class_data

# * Train / Test Sets -----------------------------------------------------

set.seed(123)
splits <- initial_split(data, prop = .8)

# * Recipes ---------------------------------------------------------------

rcp_spec <- recipe(Value ~ ., data = training(splits)) %>% 
	step_dummy(all_nominal(), -all_outcomes())
rcp_spec %>% prep() %>% juice() %>% glimpse()



# NAIVE BAYES -------------------------------------------------------------

?naive_Bayes()
# - Strengths: simple and efficient algorithm
# - Weaknesses: usually too simple

# * Engines ---------------------------------------------------------------

model_spec_nb <- naive_Bayes() %>%
	set_engine("klaR")

# * Workflows -------------------------------------------------------------

wrkfl_fit_nb <- workflow() %>%
	add_model(model_spec_nb) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_nb %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# LOGISTIC REGRESSION -------------------------------------------------------

?logistic_reg()
# - Strengths: easy to use and to interpret
# - Weaknesses: not so good with non-linearities

# * Engines ---------------------------------------------------------------

model_spec_logit <- logistic_reg() %>%
	set_engine("glm")

# * Workflows -------------------------------------------------------------

wrkfl_fit_logit <- workflow() %>%
	add_model(model_spec_logit) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_logit %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# ELASTIC NET -------------------------------------------------------------

?logistic_reg()
# Strengths: Very good for feature selection & collinearity
# Weaknesses: Not as good for complex patterns (i.e. non-linearities)

# * Engines ---------------------------------------------------------------

# RIDGE
model_spec_ridge <- logistic_reg(
	mode = "classification",
	penalty = 0.01,
	mixture = 0
) %>%
	set_engine("glmnet")

# LASSO
model_spec_lasso <- logistic_reg(
	mode = "classification",
	penalty = 0.01,
	mixture = 1
) %>%
	set_engine("glmnet")

# MIXED
model_spec_elanet <- logistic_reg(
	mode = "classification",
	penalty = 0.01,
	mixture = 0.5
) %>%
	set_engine("glmnet")

# * Workflows -------------------------------------------------------------

# RIDGE 
wrkfl_fit_ridge <- workflow() %>%
	add_model(model_spec_ridge) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# LASSO 
wrkfl_fit_lasso <- workflow() %>%
	add_model(model_spec_lasso) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# MIXED 
wrkfl_fit_elanet <- workflow() %>%
	add_model(model_spec_elanet) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_ridge %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")

wrkfl_fit_lasso %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")

wrkfl_fit_elanet %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# SVM ---------------------------------------------------------------------

# Support Vector Machines
?svm_linear
?svm_poly
?svm_rbf
# Strengths: Well-rounded algorithm
# Weaknesses: Needs tuned or can overfit and can be computationally inefficient

# * Engines ---------------------------------------------------------------

# SVM Linear
model_spec_svm_linear <- svm_linear(
	mode = "classification",
	cost = 10,
	margin = 0.1
) %>%
	set_engine("kernlab")

# SVM Polynomial
model_spec_svm_poly <- svm_poly(
	mode = "classification",
	cost = 10,
	degree = 2,
	scale_factor = 1,
	margin = 0.1
) %>%
	set_engine("kernlab")

# SVM Radial
model_spec_svm_rbf <- svm_rbf(
	mode = "classification",
	cost = 1,
	rbf_sigma = 0.01,
	margin = 0.1
) %>%
	set_engine("kernlab")

# * Workflows -------------------------------------------------------------

# SVM Linear
set.seed(123)
wrkfl_fit_svm_linear <- workflow() %>%
	add_model(model_spec_svm_linear) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# SVM Poly
set.seed(123)
wrkfl_fit_svm_poly <- workflow() %>%
	add_model(model_spec_svm_poly) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# SVM Radial
set.seed(123)
wrkfl_fit_svm_rbf <- workflow() %>%
	add_model(model_spec_svm_rbf) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_svm_linear %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")

wrkfl_fit_svm_poly %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")

wrkfl_fit_svm_rbf %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# KNN ---------------------------------------------------------------------

# K Neighrest Neighbors
?nearest_neighbor()
# Strengths: Uses neighboring points to estimate
# Weaknesses: Cannot predict beyond the maximum/minimum target

# * Engines ---------------------------------------------------------------

model_spec_knn <- nearest_neighbor(
	mode = "classification",
	neighbors = 50,
	dist_power = 10,
	weight_func = "optimal"
) %>%
	set_engine("kknn")

# * Workflows -------------------------------------------------------------

set.seed(123)
wrkfl_fit_knn <- workflow() %>%
	add_model(model_spec_knn) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_knn %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# DECISION TREE -----------------------------------------------------------

?decision_tree()
# - Strengths: Easy and interpretable
# - Weaknesses: Usually poor performance

# * Engines ---------------------------------------------------------------

model_spec_tree <- decision_tree(
	mode = "classification",
	cost_complexity = 0,
	tree_depth = 10,
	min_n = 2
) %>%
	set_engine("rpart") 

# * Workflows -------------------------------------------------------------

set.seed(123)
wrkfl_fit_tree <- workflow() %>%
	add_model(model_spec_tree) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

wrkfl_fit_tree %>%
	extract_fit_engine() %>%
	rpart.plot::rpart.plot(roundint = FALSE)

wrkfl_fit_tree %>% 
	extract_fit_parsnip() %>% 
	vip::vip()

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_tree %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# BAGGING -----------------------------------------------------------------

?bag_tree()
# Strengths: Reduces Variance
# Weaknesses: Tend to overfit hence must be tunes

# * Engines ---------------------------------------------------------------

model_spec_bag <- bag_tree(
	mode = "classification",
	cost_complexity = 0,
	tree_depth = 10,
	min_n = 2
) %>%
	set_engine("rpart",	times = 50) # 50 ensemble members

# * Workflows -------------------------------------------------------------

set.seed(123)
wrkfl_fit_bag <- workflow() %>%
	add_model(model_spec_bag) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_bag %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# RANDOM FOREST -----------------------------------------------------------

?rand_forest()
# - Strengths: Can model complex patterns very well
# - Weaknesses: Cannot predict beyond the maximum/minimum target

# * Engines ---------------------------------------------------------------

model_spec_rf <- rand_forest(
	mode = "classification",
	mtry = 25,
	trees = 1000,
	min_n = 25
) %>%
	# set_engine("randomForest")
	set_engine("ranger") # faster implementation

# * Workflows -------------------------------------------------------------

set.seed(123)
wrkfl_fit_rf <- workflow() %>%
	add_model(model_spec_rf) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_rf %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# BOOSTING ----------------------------------------------------------------

?boost_tree()

# LIGHT GBM
# https://lightgbm.readthedocs.io/en/latest/
# https://github.com/microsoft/LightGBM

# CAT BOOST
# https://catboost.ai/en/docs/
# https://github.com/catboost/catboost

# Strengths: Best for complex patterns
# Weaknesses: Cannot predict beyond the maximum/minimum target

# * Engines ---------------------------------------------------------------

# XGBOOST
model_spec_xgb <- boost_tree(
	mode = "classification",
	mtry = 25,
	trees = 1000,
	min_n = 2,
	tree_depth = 12,
	learn_rate = 0.3,
	loss_reduction = 0
) %>%
	set_engine("xgboost")

# LIGHT GBM
model_spec_lightgbm <- boost_tree(mode = "classification") %>%
	set_engine("lightgbm")
# objective = "reg:tweedie"

# * Workflows -------------------------------------------------------------

# XGBOOST
set.seed(123)
wrkfl_fit_xgb <- workflow() %>%
	add_model(model_spec_xgb) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

wrkfl_fit_xgb %>%
	extract_fit_parsnip() %>%
	pluck("fit") %>%
	xgboost::xgb.importance(model = .) %>%
	xgboost::xgb.plot.importance()

# LIGHT GBM
set.seed(123)
wrkfl_fit_lightgbm <- workflow() %>%
	add_model(model_spec_lightgbm) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

wrkfl_fit_lightgbm %>%
	parsnip::extract_fit_engine() %>%
	lightgbm::lgb.importance() %>%
	lightgbm::lgb.plot.importance()

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_xgb %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")

wrkfl_fit_lightgbm %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# NEURAL NETWORKS ---------------------------------------------------------

?mlp()
# - Single Layer / Multi Layer Perceptron Network
# - Simple network 
# - Like linear classification with non-linear functions
# - Can improve learning by adding more hidden units, epochs, etc

# * Engines ---------------------------------------------------------------

model_spec_nnet <- mlp(
	mode = "classification",
	hidden_units = 10,
	penalty = 1,
	epochs = 100
) %>%
	set_engine("nnet")

# * Workflows -------------------------------------------------------------

set.seed(123)
wrkfl_fit_nnet <- workflow() %>%
	add_model(model_spec_nnet) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_nnet %>% 
	calibrate_evaluate_plot(y = "Value", mode = "classification", type = "testing")



# ML Models' Performance --------------------------------------------------

# * Comparison ------------------------------------------------------------

wrkfl_fit_list <- list(
	# NAIVE BAYES
	wrkfl_fit_nb,
	# LOGISTIC REGRESSION 
	wrkfl_fit_logit,
	# ELASTIC NET
	wrkfl_fit_ridge,
	wrkfl_fit_lasso,
	wrkfl_fit_elanet,
	# SVM
	wrkfl_fit_svm_linear,
	wrkfl_fit_svm_poly,
	wrkfl_fit_svm_rbf,
	# KNN
	wrkfl_fit_knn,
	# DECISION TREE
	wrkfl_fit_tree,
	# BAGGING
	wrkfl_fit_bag,
	# RANDOM FOREST
	wrkfl_fit_rf,
	# BOOSTING
	wrkfl_fit_xgb,
	wrkfl_fit_lightgbm,
	# NEURAL NETWORK
	wrkfl_fit_nnet
)

methods_list <- list(
	"NAIVE",
	"LOGIT",
	"RIDGE", "LASSO", "ELASTIC NET",
	"SVM LINEAR", "SVM POLY", "SVM RADIAL", 
	"KNN", 
	"TREE", "BAGGING", "RANDOM FOREST", 
	"XGBOOST", "LIGHT GBM", 
	"MLP"
)

results <- map2(
	wrkfl_fit_list, 
	methods_list, 
	~ collect_results(.x, y = "Value", mode = "classification", method = .y)
)

metrics <- results %>% map("pred_metrics") %>% bind_rows()
metrics %>% 
	pivot_wider(names_from = Metric, values_from = Estimate) %>% 
	mutate(across(where(is.numeric), ~ round(., 3))) %>% 
	filter(Type == "testing") %>% 
	DT::datatable(options = list(pageLength = 15))

rocs <- results %>% map("pred_roc") %>% bind_rows()
p_rocs <- rocs %>% 
	ggplot(aes(x = 1 - specificity, y = sensitivity, col = Type)) +
	geom_path() +
	geom_abline(lty = 3) +
	coord_equal() + 
	scale_color_viridis_d(option = "plasma", end = .6) +
	facet_wrap(~ Method, ncol = 3) +
	theme_minimal() 
plotly::ggplotly(p_rocs)

p_rocs_methods <- rocs %>% 
	ggplot(aes(x = 1 - specificity, y = sensitivity, col = Method)) +
	geom_path() +
	geom_abline(lty = 3) +
	coord_equal() + 
	facet_wrap(~ Type, ncol = 3) +
	theme_minimal() 
plotly::ggplotly(p_rocs_methods)

preds <- results %>% map("pred_results") %>% bind_rows()

