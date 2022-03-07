# Statistical Learning ----

# Lecture 1: Model Building -----------------------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - Tidymodels
# - Engines
# - Model Building



# Tidymodels --------------------------------------------------------------

# https://www.tidymodels.org/

# Modeling with the tidyverse uses the collection of tidymodels packages
# The tidymodels framework is a collection of packages for modeling and 
# machine learning using tidyverse principles.


# * Parsnip ---------------------------------------------------------------

# https://parsnip.tidymodels.org/
# https://www.tidymodels.org/find/

# The goal of parsnip is to provide a tidy, unified interface to models 
# that can be used to try a range of models without getting bogged down 
# in the syntactical minutiae of the underlying packages.

# One challenge with different modeling functions available in R that do the 
# same thing is that they can have different interfaces and arguments.
# The model syntax can be very different and that the argument names (and 
# formats) are also different. This is a pain if you switch between 
# implementations.

# The goals of parsnip are to:  
# * Separate the definition of a model from its evaluation.  
# * Decouple the model specification from the implementation 
#   (whether the implementation is in R, spark, or something else).  
# * Harmonize argument names (e.g. n.trees, ntrees, trees) so that users 
#   only need to remember a single name. This will help across model types 
#   too so that trees will be the same argument across random forest as 
#   well as boosting or bagging.

# In particular, parsnip dose this by defining:  
# * the type of model is "random forest"  
# * the mode of the model is "regression" or "classification"  
# * the computational engine is the name of the R package.  

# A list of all parsnip models across different CRAN packages can be found 
# at tidymodels.org.


# * Recipes ---------------------------------------------------------------

# https://recipes.tidymodels.org/index.html

# You may consider recipes as an alternative method for creating and 
# pre-processing design matrices (also known as model matrices) that can 
# be used for modeling or visualization.
# With recipes, you can use dplyr-like pipeable sequences of feature 
# engineering steps to get your data ready for modeling.


# * Rsample ---------------------------------------------------------------

# https://rsample.tidymodels.org/
  
# The rsample package provides functions to create different types of 
# resamples and corresponding classes for their analysis. The goal is 
# to have a modular set of methods that can be used for:
# * resampling for estimating the sampling distribution of a statistic
# * estimating model performance using a holdout set  
# 
# The scope of rsample is to provide the basic building blocks for creating 
# and analyzing resamples of a data set, but this package does not include 
# code for modeling or calculating statistics.


# * Tune ------------------------------------------------------------------

# https://tune.tidymodels.org/

# The goal of tune is to facilitate hyperparameter tuning for the tidymodels 
# packages. It relies heavily on recipes, parsnip, and dials.


# * Yardstick -------------------------------------------------------------

# https://yardstick.tidymodels.org/

# yardstick is a package to estimate how well models are working using tidy 
# data principles.



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")



# Data --------------------------------------------------------------------

# Let’s use the data from Constable (1993) to explore how three different 
# feeding regimes affect the size of sea urchins over time. The initial 
# size of the sea urchins at the beginning of the experiment probably 
# affects how big they grow as they are fed.

urchins <- read_csv("data/urchins.csv")
urchins %>% glimpse()

urchins <- urchins %>% 
	setNames(c("food_regime", "initial_volume", "width")) %>% 
	mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))
urchins

# For each of the 72 urchins, we know their:
# 	* experimental feeding regime group (food_regime: either Initial, Low, or High),
#   * size in milliliters at the start of the experiment (initial_volume), and
#   * suture width at the end of the experiment (width).

urchins %>% 
	ggplot(aes(x = initial_volume, y = width,	group = food_regime, col = food_regime)) + 
	geom_point() + 
	geom_smooth(method = lm, se = FALSE) +
	scale_color_viridis_d(option = "plasma", end = .7)



# Linear Modelling --------------------------------------------------------

# A standard two-way analysis of variance (ANOVA) model makes sense for this 
# dataset because we have both a continuous predictor and a categorical 
# predictor. Since the slopes appear to be different for at least two of the 
# feeding regimes, let’s build a model that allows for two-way interactions. 
# Specifying an R formula with our variables in this way:

# `width ~ initial_volume * food_regime`

# allows our regression model depending on initial volume to have separate 
# slopes and intercepts for each food regime.
# For this kind of model, ordinary least squares is a good initial approach.


# * Engine ----------------------------------------------------------------

# With tidymodels, we start by specifying the functional form of the model 
# that we want using the parsnip package. Since there is a numeric outcome 
# and the model should be linear with slopes and intercepts, the model 
# type is “linear regression”.  

linear_reg()

# That is pretty underwhelming since, on its own, it doesn’t really do much. 
# However, now that the type of model has been specified, a method for 
# fitting or training the model can be stated using the engine. The engine 
# value is often a mash-up of the software that can be used to fit or train 
# the model as well as the estimation method. For example, to use ordinary 
# least squares, we can set the engine to be lm. 
# The documentation page for linear_reg() lists the possible engines.

lm_mod <-	linear_reg() %>% 
	set_engine("lm")
lm_mod


# * Fitting ---------------------------------------------------------------

# The model can be estimated or trained using the fit() function, 
# specifying the desired formula.

lm_fit <- lm_mod %>% 
	fit(width ~ initial_volume * food_regime, data = urchins)
lm_fit

# Perhaps our analysis requires a description of the model parameter 
# estimates and their statistical properties. Although the summary() 
# function for lm objects can provide that, it gives the results back 
# in an unwieldy format. Many models have a tidy() method that provides 
# the summary results in a more predictable and useful format 
# (e.g. a data frame with standard column names).

summary(lm_fit$fit)
tidy(lm_fit)

lm_fit %>% 
	tidy() %>% 
	dwplot(
		dot_args = list(size = 2, color = "black"),
		whisker_args = list(color = "black"),
		vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2)
	)


# * Predicting ------------------------------------------------------------

# This fitted object lm_fit has the lm model output built-in, which you can
# access with lm_fit$fit, but there are some benefits to using the fitted 
# parsnip model object when it comes to predicting.
# Suppose that it would be particularly interesting to predict the mean
# body size for urchins that started the experiment with an initial volume 
# of 20ml. We start with some new example data that we will make predictions for.

new_points <- expand.grid(initial_volume = 20, food_regime = c("Initial", "Low", "High"))
new_points

# To get our predicted results, we can use the predict() function to find 
# the mean values at 20ml.
# It is also important to communicate the variability, so we also need to 
# find the predicted confidence intervals. If we had used lm() to fit the 
# model directly, a few minutes of reading the documentation page for 
# predict.lm() would explain how to do this. However, if we decide to use 
# a different model to estimate urchin size (spoiler: we will!), it is 
# likely that a completely different syntax would be required.
# Instead, with tidymodels, the types of predicted values are standardized 
# so that we can use the same syntax to get these values.

# When making predictions, the tidymodels convention is to always produce 
# a tibble of results with standardized column names. This makes it easy to 
# combine the original data and the predictions in a usable format.

# First, let’s generate the mean body width values
mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred

# Second, let's compute the confidence intervals
conf_int_pred <- predict(lm_fit, new_data = new_points, type = "conf_int")
conf_int_pred

# Third, combine 
pred_data <- new_points %>% 
	bind_cols(mean_pred) %>% 
	bind_cols(conf_int_pred)
pred_data

pred_data %>% 
	ggplot(aes(x = food_regime)) + 
	geom_point(aes(y = .pred)) + 
	geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
	labs(y = "urchin size")



# Bayesian Model ----------------------------------------------------------

# W are interested in knowing if the results would be different if the model 
# were estimated using a Bayesian approach. In such an analysis, a prior 
# distribution needs to be declared for each model parameter that represents 
# the possible values of the parameters (before being exposed to the observed 
# data). We may suggest that the priors should be bell-shaped but, since we 
# don't have any idea what the range of values should be, we decided to take 
# a conservative approach and make the priors wide using a Cauchy distribution 
# (which is the same as a t-distribution with a single degree of freedom).

# The documentation on the rstanarm package shows us that the stan_glm() 
# function can be used to estimate this model, and that the function 
# arguments that need to be specified are called prior and prior_intercept. 
# It turns out that linear_reg() has a stan engine. Since these prior 
# distribution arguments are specific to the Stan software, they are passed 
# as arguments to parsnip::set_engine().


# * Engine ----------------------------------------------------------------

# set the prior distribution
prior_dist <- rstanarm::student_t(df = 1)

set.seed(123)
bayes_mod <- linear_reg() %>% 
	set_engine(
		"stan", 
		prior_intercept = prior_dist, 
		prior = prior_dist
	) 


# * Fitting ---------------------------------------------------------------

# train the model
bayes_fit <- bayes_mod %>% 
	fit(width ~ initial_volume * food_regime, data = urchins)
bayes_fit
print(bayes_fit, digits = 5)

summary(bayes_fit$fit)
tidy(bayes_fit, conf.int = TRUE)


# * Predicting ------------------------------------------------------------

# A goal of the tidymodels packages is that the interfaces to common 
# tasks are standardized (as seen in the tidy() results above). The same 
# is true for getting predictions; we can use the same code even though 
# the underlying packages use very different syntax.

mean_bayes_pred <- predict(bayes_fit, new_data = new_points)
mean_bayes_pred

conf_int_bayes_pred <- predict(bayes_fit, new_data = new_points, type = "conf_int")
conf_int_bayes_pred

bayes_pred_data <- new_points %>% 
	bind_cols(mean_bayes_pred) %>% 
	bind_cols(conf_int_bayes_pred)

bayes_pred_data %>% 
	ggplot(aes(x = food_regime)) + 
	geom_point(aes(y = .pred)) + 
	geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
	labs(y = "urchin size") + 
	ggtitle("Bayesian model with t(1) prior distribution")


# The extra step of defining the model using a function like linear_reg() 
# might seem superfluous since a call to lm() is much more succinct. 
# However, the problem with standard modeling functions is that they don’t 
# separate what you want to do from the execution. The benefit of separating
# the two processes will be clear when dealing with model workflow and 
# hyperparameter tuning. Indeed, model tuning with tidymodels uses the 
# specification of the model to declare what parts of the model should be tuned. 
# That would be very difficult to do if linear_reg() immediately fit the model.



# Comparing Different Models ----------------------------------------------

# Different models comparisons can be easily done since the model outputs
# have been standardized working with tidymodels.

compare_pred_data <- bind_rows(
	pred_data %>% add_column("model" = "OLS"),
	bayes_pred_data %>% add_column("model" = "Bayes")
)

compare_pred_data %>% 
	ggplot(aes(x = food_regime, col = model)) + 
	geom_point(aes(y = .pred)) + 
	geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
	# facet_wrap(~ model) + 
	labs(y = "urchin size") + 
	ggtitle("Model Comparison")

