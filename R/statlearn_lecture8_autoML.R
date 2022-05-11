# Statistical Learning ----

# Lecture 8: Automatic ML -------------------------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - H2O
# - AutoML



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
	ungroup()
wind

# Convert data into H2O frame
wind_h2o <- as.h2o(wind)

h2o.ls()
h2o.describe(wind_h2o)


# * Data Splitting --------------------------------------------------------

# First off, we’ll split our data into a training set and a test set.
# h2o.splitFrame() uses approximate splitting. That is, it won’t split the 
# data into an exact 80%-20% split. Setting the seed allows us to create 
# reproducible results.

splits <- h2o.splitFrame(
	data = wind_h2o,
	ratios = c(0.8),  # partition data into 80% and 20% chunks
	seed = 123
)

train <- splits[[1]]
test <- splits[[2]]

# We can use h2o.nrow() to check the number of rows in our train and test sets.
h2o.nrow(train)
h2o.nrow(test)


# * H2O Modelling ---------------------------------------------------------

# We’ve got our H2O instance up and running, with some data in it. Let’s go 
# ahead and do some machine learning.

y <- "turbine_rated_capacity_kw"
x <- setdiff(names(wind_h2o), y)

h2o_rf <- h2o.randomForest(
	x = x, y = y,
	training_frame = train,
	nfolds = 5,
	model_id = "rf",
	seed = 123
)
print(h2o_rf)


h2o.performance(model = h2o_rf, newdata = test)
h2o.predict(h2o_rf, test)




















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




train <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
test <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

# Identify predictors and response
y <- "response"
x <- setdiff(names(train), y)

# For binary classification, response should be a factor
train[, y] <- as.factor(train[, y])
test[, y] <- as.factor(test[, y])

# Run AutoML for 20 base models
aml <- h2o.automl(
	x = x, y = y,
	training_frame = train,
	max_models = 1,
	seed = 1
)

# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))

# To generate predictions on a test set, you can make predictions
# directly on the `H2OAutoML` object or on the leader model
# object directly
pred <- h2o.predict(aml, test)  # predict(aml, test) also works

# or:
pred <- h2o.predict(aml@leader, test)

# Get leaderboard with all possible columns
lb <- h2o.get_leaderboard(object = aml, extra_columns = "ALL")
lb


# Get the best model using the metric
m <- aml@leader
# this is equivalent to
m <- h2o.get_best_model(aml)

# Get the best model using a non-default metric
m <- h2o.get_best_model(aml, criterion = "logloss")

# Get the best XGBoost model using default sort metric
xgb <- h2o.get_best_model(aml, algorithm = "xgboost")

# Get the best XGBoost model, ranked by logloss
xgb <- h2o.get_best_model(aml, algorithm = "xgboost", criterion = "logloss")






# Tidymodels Interface to H20 ---------------------------------------------









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



