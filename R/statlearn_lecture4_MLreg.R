# Statistical Learning ----

# Lecture 4: Regression with ML Models ------------------------------------
# Marco Zanotti

# Goals:
# - Linear Regression
# - Elastic Net
# - MARS
# - SVM
# - KNN
# - Bagging
# - Random Forest
# - XGBoost, Light GBM
# - Cubist
# - Neural Networks



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")

# packages treesnip and catboost have to be installed from dev versions
# remotes::install_github("curso-r/treesnip")
# devtools::install_url(
# 	"https://github.com/catboost/catboost/releases/download/v1.0.0/catboost-R-Linux-1.0.0.tgz",
# 	INSTALL_opts = c("--no-multiarch", "--no-test-load")
# )



# Data --------------------------------------------------------------------

artifacts_list <- read_rds("artifacts/artifacts_list.rds")
data <- artifacts_list$reg_data

# * Train / Test Sets -----------------------------------------------------

set.seed(123)
splits <- initial_split(data, prop = .8)

# * Recipes ---------------------------------------------------------------

rcp_spec <- recipe(SalePrice ~ ., data = training(splits)) %>% 
	step_dummy(all_nominal(), -all_outcomes())
rcp_spec %>% prep() %>% juice() %>% glimpse()



# LINEAR REGRESSION -------------------------------------------------------

?linear_reg()
# Baseline model for regression ML

# * Engines ---------------------------------------------------------------

model_spec_lm <- linear_reg() %>%
	set_engine("lm")

# * Workflows -------------------------------------------------------------

wrkfl_fit_lm <- workflow() %>%
	add_model(model_spec_lm) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_lm %>% extract_fit_parsnip() %>% tidy()

wrkfl_fit_lm %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "training")

wrkfl_fit_lm %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# ELASTIC NET -------------------------------------------------------------

?linear_reg()
# Strengths: Very good for feature selection & collinearity
# Weaknesses: Not as good for complex patterns (i.e. non-linearities)

# * Engines ---------------------------------------------------------------

# RIDGE
model_spec_ridge <- linear_reg(
	mode = "regression",
	penalty = 0.01,
	mixture = 0
) %>%
	set_engine("glmnet")

# LASSO
model_spec_lasso <- linear_reg(
	mode = "regression",
	penalty = 0.01,
	mixture = 1
) %>%
	set_engine("glmnet")

# MIXED
model_spec_elanet <- linear_reg(
	mode = "regression",
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
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")

wrkfl_fit_lasso %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")

wrkfl_fit_elanet %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# MARS --------------------------------------------------------------------

# Multiple Adaptive Regression Splines
?mars()
# Strengths: Good algorithm for modeling linearities
# Weaknesses:
#  - Not good for complex patterns (i.e. non-linearities)
#  - Don't combine with splines! MARS makes splines. 

# * Engines ---------------------------------------------------------------

model_spec_mars <- mars(
	mode = "regression",
	num_terms = 10
) %>%
	set_engine("earth", endspan = 100)

# * Workflows -------------------------------------------------------------

wrkfl_fit_mars <- workflow() %>%
	add_model(model_spec_mars) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_mars %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



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
	mode = "regression",
	cost = 10,
	margin = 0.1
) %>%
	set_engine("kernlab")

# SVM Polynomial
model_spec_svm_poly <- svm_poly(
	mode = "regression",
	cost = 10,
	degree = 2,
	scale_factor = 1,
	margin = 0.1
) %>%
	set_engine("kernlab")

# SVM Radial
model_spec_svm_rbf <- svm_rbf(
	mode = "regression",
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
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")

wrkfl_fit_svm_poly %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")

wrkfl_fit_svm_rbf %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# KNN ---------------------------------------------------------------------

# K Neighrest Neighbors
?nearest_neighbor()
# Strengths: Uses neighboring points to estimate
# Weaknesses: Cannot predict beyond the maximum/minimum target

# Trend Issue Example
# sample_data_tbl <- tibble(
#   date = timetk::tk_make_timeseries("2021", by = "quarter", length_out = 20),
#   value = 1:20
# )
# sample_data_tbl %>% timetk::plot_time_series(date, value, .smooth = FALSE)
# model_fit_knn <- nearest_neighbor(mode = "regression") %>%
#   set_engine("kknn") %>%
#   fit(value ~ as.numeric(date), sample_data_tbl)
# modeltime::modeltime_table(model_fit_knn) %>%
#   modeltime::modeltime_forecast(
#     new_data = bind_rows(
#       sample_data_tbl,
#       timetk::future_frame(sample_data_tbl, .length_out = "2 years")
#     ),
#     actual_data = sample_data_tbl
#   ) %>%
#   modeltime::plot_modeltime_forecast()

# * Engines ---------------------------------------------------------------

model_spec_knn <- nearest_neighbor(
	mode = "regression",
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
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# BAGGING -----------------------------------------------------------------

?bag_tree()
# Strengths: Reduces Variance
# Weaknesses: Tend to overfit hence must be tunes

# * Engines ---------------------------------------------------------------

model_spec_bag <- bag_tree(
	mode = "regression",
	cost_complexity = 0,
	tree_depth = 10,
	min_n = 2
) %>%
	set_engine("rpart",	times = 50) # 25 ensemble members

# * Workflows -------------------------------------------------------------

set.seed(123)
wrkfl_fit_bag <- workflow() %>%
	add_model(model_spec_bag) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_bag %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# RANDOM FOREST -----------------------------------------------------------

?rand_forest()
# - Strengths: Can model complex patterns very well
# - Weaknesses: Cannot predict beyond the maximum/minimum target

# * Engines ---------------------------------------------------------------

model_spec_rf <- rand_forest(
	mode = "regression",
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
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



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
	mode = "regression",
	mtry = 25,
	trees = 1000,
	min_n = 2,
	tree_depth = 12,
	learn_rate = 0.3,
	loss_reduction = 0
) %>%
	set_engine("xgboost")

# LIGHT GBM
model_spec_lightgbm <- boost_tree(mode = "regression") %>%
	set_engine("lightgbm")
# objective = "reg:tweedie"

# CAT BOOST
# model_spec_catboost <- boost_tree(mode = "regression") %>%
# 	set_engine("catboost")
# loss_function = "Tweedie:variance_power=1.5"

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

# CAT BOOST
# set.seed(123)
# wrkfl_fit_catboost <- workflow() %>%
# 	add_model(model_spec_catboost) %>%
# 	add_recipe(rcp_spec) %>%
# 	fit(training(splits))
# 
# wrkfl_fit_catboost %>%
# 	parsnip::extract_fit_engine() %>%
# 	catboost::catboost.get_feature_importance() %>%
# 	as_tibble(rownames = "feature") %>%
# 	rename(value = V1) %>%
# 	arrange(-value) %>%
# 	mutate(feature = as_factor(feature) %>% fct_rev()) %>%
# 	dplyr::slice(1:10) %>%
# 	ggplot(aes(value, feature)) +
# 	geom_col()

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_xgb %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")

wrkfl_fit_lightgbm %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")

# wrkfl_fit_catboost %>% 
# 	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# CUBIST ------------------------------------------------------------------

?cubist_rules()
# Like XGBoost, but the terminal (final) nodes are fit using linear regression
# Strengths: can predict beyond maximum

# * Engines ---------------------------------------------------------------

model_spec_cubist <- cubist_rules(
	committees = 50,
	neighbors = 7,
	max_rules = 100
) %>%
	set_engine("Cubist")

# * Workflows -------------------------------------------------------------

set.seed(123)
wrkfl_fit_cubist <- workflow() %>%
	add_model(model_spec_cubist) %>%
	add_recipe(rcp_spec) %>%
	fit(training(splits))

# * Calibration, Evaluation & Plotting ------------------------------------

wrkfl_fit_cubist %>% 
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# NEURAL NETWORKS ---------------------------------------------------------

?mlp()
# - Single Layer / Multi Layer Perceptron Network
# - Simple network 
# - Like linear regression with non-linear functions
# - Can improve learning by adding more hidden units, epochs, etc

# * Engines ---------------------------------------------------------------

model_spec_nnet <- mlp(
	mode = "regression",
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
	calibrate_evaluate_plot(y = "SalePrice", mode = "regression", type = "testing")



# ML Models' Performance --------------------------------------------------

# * Comparison ------------------------------------------------------------

wrkfl_fit_list <- list(
	# LM
	wrkfl_fit_lm,
	# ELASTIC NET
	wrkfl_fit_ridge,
	wrkfl_fit_lasso,
	wrkfl_fit_elanet,
	# MARS
	wrkfl_fit_mars,
	# SVM
	wrkfl_fit_svm_linear,
	wrkfl_fit_svm_poly,
	wrkfl_fit_svm_rbf,
	# KNN
	wrkfl_fit_knn,
	# BAGGING
	wrkfl_fit_bag,
	# RANDOM FOREST
	wrkfl_fit_rf,
	# BOOSTING
	wrkfl_fit_xgb,
	wrkfl_fit_lightgbm,
	#wrkfl_fit_catboost,
	# CUBIST
	wrkfl_fit_cubist,
	# NEURAL NETWORK
	wrkfl_fit_nnet
)

methods_list <- list(
	"LM", 
	"RIDGE", "LASSO", "ELASTIC NET",
	"MARS",
	"SVM LINEAR", "SVM POLY", "SVM RADIAL", 
	"KNN", 
	"BAGGING", "RANDOM FOREST", 
	"XGBOOST", "LIGHT GBM", #"CATBOOST", 
	"CUBIST", 
	"MLP"
)

results <- map2(
	wrkfl_fit_list, 
	methods_list, 
	~ collect_results(.x, y = "SalePrice", mode = "regression", method = .y)
)

metrics <- results %>% map("pred_metrics") %>% bind_rows()
metrics %>% 
	pivot_wider(names_from = Metric, values_from = Estimate) %>% 
	mutate(across(where(is.numeric), ~ round(., 3))) %>% 
	filter(Type == "testing") %>% 
	DT::datatable(options = list(pageLength = 15))

preds <- results %>% map("pred_results") %>% bind_rows()
p <- preds %>% 
	select(-Type) %>% 
	group_by(Method) %>% 
	mutate(id = 1:n()) %>% 
	ungroup() %>% 
	ggplot(aes(x = id)) +
	geom_point(aes(y = Actual, col = "Actual")) +
	geom_point(aes(y = Pred, col = "Pred")) +
	scale_color_manual(values = c("black", "red")) +
	labs(x = "", y = "") +
	facet_wrap(~ Method, ncol = 3) +
	theme_minimal() 
plotly::ggplotly(p)

