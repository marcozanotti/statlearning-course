# Statistical Learning ----

# Lecture 2: Features Engineering -----------------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - Feature Engineering
# - Recipes
# - Workflows



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")



# Data --------------------------------------------------------------------

# Let’s use the nycflights13 data to predict whether a plane arrives more 
# than 30 minutes late. This data set contains information on 325,819 
# flights departing near New York City in 2013. Let’s start by loading the 
# data and making a few changes to the variables.

nycflights13::
	
flights %>% glimpse()
weather %>% glimpse()


flight_data <- flights %>% 
	inner_join(weather, by = c("origin", "time_hour")) %>% # filtering join
	mutate(
		# Convert the arrival delay to a factor
		arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
		arr_delay = factor(arr_delay),
		# We will use the date (not date-time) in the recipe below
		date = lubridate::as_date(time_hour)
	) %>% 
	mutate(across(where(is.character), as.factor)) %>% 
	drop_na() %>% 
	select(
		dep_time, flight, origin, dest, air_time, 
		distance, carrier, date, arr_delay, time_hour
	)

flight_data %>% 
	count(arr_delay) %>% 
	mutate(prop = n/sum(n))

# We can see that about 16% of the flights in this data set arrived 
# more than 30 minutes late.


# * Explorative Data Analysis ---------------------------------------------

# Before we start building up our recipe, let’s take a quick look at a 
# few specific variables that will be important for both preprocessing 
# and modeling.

flight_data %>% glimpse()

flight_data %>% 
	skimr::skim(dest, carrier) 

# First, notice that the variable we created called arr_delay is a 
# factor variable; it is important that our outcome variable for 
# training a logistic regression model is a factor.

# Second, there are two variables that we don’t want to use as predictors 
# in our model, but that we would like to retain as identification variables 
# that can be used to troubleshoot poorly predicted data points. These are 
# flight, a numeric value, and time_hour, a date-time value.

# Third, there are 104 flight destinations contained in dest and 16 distinct 
# carriers.

flight_data %>% 
	DataExplorer::create_report()

# Because we’ll be using a simple logistic regression model, the variables 
# dest and carrier will be converted to dummy variables. However, some 
# of these values do not occur very frequently and this could complicate 
# our analysis. We’ll discuss specific steps later in this article that 
# we can add to our recipe to address this issue before modeling.


# * Data Splitting --------------------------------------------------------

# Let’s split this single dataset into two: a training 
# set and a testing set. We’ll keep most of the rows in the original 
# dataset (subset chosen randomly) in the training set. The training 
# data will be used to fit the model, and the testing set will be used 
# to measure model performance.

# To do this, we can use the rsample package to create an object that 
# contains the information on how to split the data, and then two more 
# rsample functions to create data frames for the training and testing sets.
	

set.seed(123)
data_split <- initial_split(flight_data, prop = 0.75) # Put 3/4 of the data into the training set 
data_split

train_data <- training(data_split)
test_data  <- testing(data_split)



# Recipe ------------------------------------------------------------------

# In lecture 1, we learned how to specify and train models with different 
# engines using the parsnip package. 
# In this lecture, we’ll explore another tidymodels package, recipes, which 
# is designed to help you preprocess your data before training your 
# model. Recipes are built as a series of preprocessing steps, such as:
# 	
# 	* converting qualitative predictors to indicator variables 
#     (also known as dummy variables),
#   * transforming data to be on a different scale (e.g., taking the
#   	logarithm of a variable),
#   * transforming whole groups of predictors together,
#   * extracting key features from raw variables (e.g., getting the day
#   	of the week out of a date variable),
# 
# and so on. If you are familiar with R’s formula interface, a lot 
# of this might sound familiar and like what a formula already does. 
# Recipes can be used to do many of the same things, but they have 
# a much wider range of possibilities. 
# In this lecture we will understand how to use recipes for modeling.


# * Create Recipe and Update Roles ----------------------------------------

# Let’s create a recipe for a simple logistic regression model. 
# Before training the model, we can use a recipe to create a few new 
# predictors and conduct some preprocessing required by the model.
 
# Let’s initiate a new recipe
	
flights_rec <- recipe(arr_delay ~ ., data = train_data) 
flights_rec

# The recipe() function as we used it here has two arguments:
#	
# 	* A formula. Any variable on the left-hand side of the tilde (~) 
#     is considered the model outcome (here, arr_delay). 
#     On the right-hand side of the tilde are the predictors. 
#     Variables may be listed by name, or you can use the dot (.) to 
#     indicate all other variables as predictors.
#   * The data. A recipe is associated with the data set used to 
#     create the model. This will typically be the training set, so 
#     data = train_data here. Naming a data set doesn’t actually change 
#     the data itself; it is only used to catalog the names of the 
#     variables and their types, like factors, integers, dates, etc.

# Now we can add roles to this recipe. We can use the update_role() 
# function to let recipes know that flight and time_hour are variables 
# with a custom role that we called "ID" (a role can have any character 
# value). Whereas our formula included all variables in the training 
# set other than arr_delay as predictors, this tells the recipe to keep 
# these two variables but not use them as either outcomes or predictors.

flights_rec <- recipe(arr_delay ~ ., data = train_data) %>% 
	update_role(flight, time_hour, new_role = "ID") 
flights_rec

# This step of adding roles to a recipe is optional; the purpose of using 
# it here is that those two variables can be retained in the data but not 
# included in the model. This can be convenient when, after the model is 
# fit, we want to investigate some poorly predicted value. These ID 
# columns will be available and can be used to try to understand what went 
# wrong.

# To get the current set of variables and roles, use the summary() function
summary(flights_rec)


# * Features Engineering --------------------------------------------------

# Now we can start adding steps onto our recipe using the pipe operator. 
# Perhaps it is reasonable for the date of the flight to have an effect on 
# the likelihood of a late arrival. A little bit of feature engineering might 
# go a long way to improving our model. 

# How should the date be encoded into the model? 
# The date column has an R date object so including that column 
# “as is” will mean that the model will convert it to a numeric format 
# equal to the number of days after a reference date.

flight_data %>% 
	distinct(date) %>% 
	mutate(numeric_date = as.numeric(date)) 
 
# It’s possible that the numeric date variable is a good option for modeling; 
# perhaps the model would benefit from a linear trend between the log-odds 
# of a late arrival and the numeric date variable. However, it might be better 
# to add model terms derived from the date that have a better potential to be 
# important to the model. For example, we could derive meaningful
# features from the single date variable, such as the day of the week, the 
# month, and whether or not the date corresponds to a holiday.
# Let’s do all three of these by adding steps to our recipe.
	
flights_rec <- recipe(arr_delay ~ ., data = train_data) %>% 
	update_role(flight, time_hour, new_role = "ID") %>% 
	step_date(date, features = c("dow", "month")) %>%               
	step_holiday(date, holidays = timeDate::listHolidays("US"), keep_original_cols = FALSE)

# What do each of these steps do?
# 
# 	* With step_date(), we created two new factor columns with the 
#     appropriate day of the week and the month.
#   * With step_holiday(), we created a binary variable indicating 
#     whether the current date is a holiday or not. The argument 
#     value of timeDate::listHolidays("US") uses the timeDate package 
#     to list the 17 standard US holidays. 
#     With keep_original_cols = FALSE, we remove the original date 
#     variable since we no longer want it in the model. Many recipe 
#     steps that create new variables have this argument.
  
# Next, we’ll turn our attention to the variable types of our 
# predictors. Because we plan to train a logistic regression model, 
# we know that predictors will ultimately need to be numeric, as 
# opposed to nominal data like strings and factor variables. 
# In other words, there may be a difference in how we store our data 
# (in factors inside a data frame), and how the underlying equations 
# require them (a purely numeric matrix).
# For factors like dest and origin, standard practice is to convert them 
# into dummy or indicator variables to make them numeric. These are binary 
# values for each level of the factor. For example, our origin variable has 
# values of "EWR", "JFK", and "LGA". The standard dummy variable encoding, 
# shown below, will create two numeric columns of the data that are 1 when 
# the originating airport is "JFK" or "LGA" and zero otherwise, respectively.

# But, unlike the standard model formula methods in R, a recipe does not 
# automatically create these dummy variables for you; you’ll need to tell 
# your recipe to add this step. This is for two reasons. First, many models 
# do not require numeric predictors, so dummy variables may not always be 
# preferred. Second, recipes can also be used for purposes outside of 
# modeling, where non-dummy versions of the variables may work better. 
# For example, you may want to make a table or a plot with a variable as 
# a single factor. For those reasons, you need to explicitly tell recipes 
# to create dummy variables using step_dummy().
	
flights_rec <- recipe(arr_delay ~ ., data = train_data) %>% 
	update_role(flight, time_hour, new_role = "ID") %>% 
	step_date(date, features = c("dow", "month")) %>%               
	step_holiday(date, holidays = timeDate::listHolidays("US"), keep_original_cols = FALSE) %>% 
	step_dummy(all_nominal_predictors())

# Here, we did something different than before: instead of applying a 
# step to an individual variable, we used selectors to apply this 
# recipe step to several variables at once, all_nominal_predictors(). 
# The selector functions can be combined to select intersections of variables.

# At this stage in the recipe, this step selects the origin, dest, and 
# carrier variables. It also includes two new variables, date_dow and 
# date_month, that were created by the earlier step_date().
 
# More generally, the recipe selectors mean that you don’t always have to 
# apply steps to individual variables one at a time. Since a recipe knows 
# the variable type and role of each column, they can also be selected 
# (or dropped) using this information.

 
# We need one final step to add to our recipe. Since carrier and dest have 
# some infrequently occurring factor values, it is possible that dummy 
# variables might be created for values that don’t exist in the training 
# set. For example, there is one destination that is only in the test set.
	
test_data %>% 
	distinct(dest) %>% 
	anti_join(train_data)

# When the recipe is applied to the training set, a column is made for 
# LEX because the factor levels come from flight_data (not the training set), 
# but this column will contain all zeros. This is a “zero-variance predictor” 
# that has no information within the column. While some R functions will not 
# produce an error for such predictors, it usually causes warnings and other 
# issues. step_zv() will remove columns from the data when the training set 
# data have a single value, so it is added to the recipe after step_dummy().

flights_rec <- recipe(arr_delay ~ ., data = train_data) %>% 
	update_role(flight, time_hour, new_role = "ID") %>% 
	step_date(date, features = c("dow", "month")) %>%               
	step_holiday(date, holidays = timeDate::listHolidays("US"), keep_original_cols = FALSE) %>% 
	step_dummy(all_nominal_predictors()) %>% 
	step_zv(all_predictors())
flights_rec


# * Evaluate a Recipe -----------------------------------------------------

flights_rec %>% prep() %>% juice() %>% glimpse()

# Now we’ve created a specification of what should be done with the data. 
# How do we use the recipe we made?



# Engine ------------------------------------------------------------------

# In order to model our data, we need a model specification with a
# model engine. Let’s use logistic regression to model the flight data. 
# As we saw lecture 1, we start by building a model specification using 
# the parsnip package.

logit_mod <- logistic_reg() %>% 
	set_engine("glm")



# Workflow ----------------------------------------------------------------

# * Create a Workflow -----------------------------------------------------

# We will want to use our recipe across several steps as we train and test 
# our model. We will:
# 	
# 	* Process the recipe using the training set: this involves any estimation
#     or calculations based on the training set. For our recipe, the training 
#     set will be used to determine which predictors should be converted to 
#     dummy variables and which predictors will have zero-variance in the 
#     training set, and should be slated for removal.
#   * Apply the recipe to the training set: we create the final predictor 
#     set on the training set.
#   * Apply the recipe to the test set: we create the final predictor set on 
#     the test set. Nothing is recomputed and no information from the test 
#     set is used here; the dummy variable and zero-variance results from 
#     the training set are applied to the test set.

# To simplify this process, we can use a model workflow, which pairs a 
# model specification engine and recipe together. This is a straightforward 
# approach because different recipes are often needed for different models, 
# so when a model and recipe are bundled, it becomes easier to train and test
# workflows. We’ll use the workflows package from tidymodels to bundle our 
# parsnip model (lr_mod) with our recipe (flights_rec).

flights_wflow <- workflow() %>% 
	add_model(logit_mod) %>% 
	add_recipe(flights_rec)
flights_wflow


# * Fitting ---------------------------------------------------------------

# Now, there is a single function that can be used to prepare the recipe
# and train the model from the resulting predictors, the fit() function.
	
flights_fit <- flights_wflow %>% 
	fit(data = train_data)
flights_fit

# This object has the finalized recipe and fitted model objects inside. 
# You may want to extract the model or recipe objects from the workflow. 
# To do this, you can use the helper functions extract_fit_parsnip() and 
# extract_recipe(). For example, here we pull the fitted model object 
# then use the broom::tidy() function to get a tidy tibble of model 
# coefficients.
	
flights_fit %>%
	extract_fit_parsnip() %>% 
	tidy()


# * Predicting ------------------------------------------------------------

# Our goal was to predict whether a plane arrives more than 30 minutes late. 
# We have just:
# 	
# 	* Built the model (lr_mod),
#   * Created a preprocessing recipe (flights_rec),
#   * Bundled the model and recipe (flights_wflow), and
#   * Trained our workflow using a single call to fit().

# The next step is to use the trained workflow (flights_fit) to predict 
# with the unseen test data, which we will do with a single call to 
# predict(). The predict() method applies the recipe to the new data, 
# then passes them to the fitted model.

predict(flights_fit, test_data)

# Because our outcome variable here is a factor, the output from predict() 
# returns the predicted class: late versus on_time. But, let’s say we want 
# the predicted class probabilities for each flight instead. To return those, 
# we can specify type = "prob" when we use predict() or use augment() with 
# the model plus test data to save them together.
	
flights_aug <- augment(flights_fit, test_data)
flights_aug %>%	select(arr_delay, time_hour, flight, .pred_class, .pred_on_time)


# * Evaluating ------------------------------------------------------------

# Now that we have a tibble with our predicted class probabilities, 
# how will we evaluate the performance of our workflow? We can see 
# from these first few rows that our model predicted these 5 on time 
# flights correctly because the values of .pred_on_time are p > .50. 
# But we also know that we have 81,455 rows total to predict. 
# We would like to calculate a metric that tells how well our model 
# predicted late arrivals, compared to the true status of our outcome 
# variable, arr_delay.

# Let’s use the area under the ROC curve as our metric, computed 
# using roc_curve() and roc_auc() from the yardstick package.

# To generate a ROC curve, we need the predicted class probabilities 
# for late and on_time, which we just calculated in the code chunk 
# above. We can create the ROC curve with these values, using 
# roc_curve() and then piping to the autoplot() method.

flights_aug %>% 
	roc_curve(truth = arr_delay, .pred_late) %>% 
	autoplot()

flights_aug %>% 
	roc_auc(truth = arr_delay, .pred_late)



# Model Comparison --------------------------------------------------------

# * Logistic Regression without Recipe ------------------------------------

# We would like to test out this workflow without this recipe. 
# We can use workflows::add_formula(arr_delay ~ .) 
# instead of add_recipe() (remember to remove the identification 
# variables first!), and see whether our recipe improved our model’s 
# ability to predict late arrivals.

flights_norecipe_fit <- workflow() %>% 
	add_formula(arr_delay ~ .) %>% 
	add_model(logit_mod) %>% 
	fit(data = train_data %>% select(-flight, -time_hour))

flights_norecipe_aug <- augment(flights_norecipe_fit, test_data) 

flights_norecipe_aug %>% 
	roc_curve(truth = arr_delay, .pred_late) %>% 
	autoplot()

flights_norecipe_aug %>% 
	roc_auc(truth = arr_delay, .pred_late)


# * Decision Tree ---------------------------------------------------------

# We want also to see whether a tree based model performs better than our
# logistic regression model on the test set.

?decision_tree

tree_mod <- decision_tree(mode = "classification") %>%
	set_engine("rpart")

flights_tree_fit <- workflow() %>% 
	add_model(tree_mod) %>% 
	add_recipe(flights_rec) %>% 
	fit(data = train_data)
flights_tree_fit

flights_tree_aug <- augment(flights_tree_fit, test_data) 

flights_tree_aug %>% 
	roc_curve(truth = arr_delay, .pred_late) %>% 
	autoplot()

flights_tree_aug %>% 
	roc_auc(truth = arr_delay, .pred_late)

