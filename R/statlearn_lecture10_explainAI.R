# Statistical Learning ----

# Lecture 10: Explainable AI -----------------------------------------------
# 2021/2022
# Marco Zanotti

# Goals:
# - How to explain ML model results?
# - DrWhy
# - DALEX, LIME & modelStudio



# DrWhy -------------------------------------------------------------------

# https://modeloriented.github.io/DrWhy/index.html
# https://github.com/ModelOriented/DrWhy
# https://www.mi2.ai/

# DrWhy is part of the DrWhy.AI developed by the MI^2 DataLab and it is the 
# collection of tools for eXplainable AI (XAI). It's based on shared
# principles and simple grammar for exploration, explanation and visualisation 
# of predictive models.

# The way how we do predictive modeling is very ineffective. We spend way too 
# much time on manual time consuming and easy to automate activities like data 
# cleaning and exploration, crisp modeling, model validation. Instead of focusing
# on model understanding, productisation and communication.

# Here we gather tools that can be use to make out work more efficient through 
# the whole model lifecycle. The unified grammar beyond DrWhy.AI universe is 
# described in the Predictive Models: Visual Exploration, Explanation and 
# Debugging book (https://ema.drwhy.ai/).


# * Lifecycle for Predictive Models ---------------------------------------

# https://github.com/ModelOriented/DrWhy/blob/master/images/ModelDevelopmentProcess.pdf

# The DrWhy is based on an unified Model Development Process inspired by RUP. 

# The process is divided into four phases of model life cycle. These
# phases are used to trace the progress in the model development. As
# in other methodologies MDP is based on series of iterations. Each
# phase may be composed out of one or more iterations.  
# Tasks that are to be performed in each iteration are listed in rows. As
# the knowledge about the problem increases every iteration, different
# tasks require more attention.


# ** Phases and iterations ------------------------------------------------

# Process of model development is divided into four phases of model life cycle 
# from the conception, assembly, tuning till the production.
 
#   1. Problem formulation: The goal of this phase is to precisely define needs 
#      for the predictive models, write down definition of done and define data 
#      set that will be used for training and validation. After this phase we 
#      know which performance measures will be used for the assessment of the 
#      final model.
 
#   2. Crisp modeling: The goal of this phase is to validate the definition of 
#      done. Here we create first versions of models in order to better understand 
#      how close we are to the desired solution.
      
#   3. Fine tuning: The goal of this phase is to tune models identified in the 
#      previous phase. Usually in this phase we create large collection of models 
#      in order to select the best one (according to some metrics).
      
#   4. Maintaining and decommissioning: Developed model go for production. The 
#      goal of this phase is to monitor the model and make sure that model 
#      performance have not degenerated. Every model will be outdated some day, 
#      prepare for the end of model life cycle.

# In more complex projects one phase may be divided into set of iterations. 
# The maintaining phase usually is composed out of series of periodic health-checks.


# ** Tasks ----------------------------------------------------------------

# Phases corresponds to the general progress in model development, while tasks 
# corresponds to programming and analytic activities that needs to be done in 
# each iteration. Importance of particular tasks is changing along model 
# life-cycle. In the diagram we showed some general patterns, but specific 
# problems may require more effort in some of these phases.
 
#   1. Data preparation: Activities needed for selection of the training, test 
#      and validation data.
#      - Data acquisition. Sometimes data needs to be read from file, from 
#        database or from some stream. Sometimes we need to scrap data from 
#        website. Sometimes one datset is not enough and we need to acquire more 
#        (maybe paid) datasets that will be combined.
#      - Data cleaning. Different data sources have different quality.
#        Sometimes some values needs recoding, errors in the data needs
#        to be spotted.
#      - Sample selection. Good model requires good and carefully
#        selected dataset. Outliers needs to be handled. If data is not
#        balanced or is heterogeneous this needs to be handled, typically
#        through oversampling, undersampling or segmentation.  
        
#   2. Data understanding: Activities needed for getting some lever of 
#      familiarity with the data, needed for further modeling.
#      - Data exploration. What are uni- and multi- variate distributions.
#        What are relation between dependent variable and explanatory variables. 
#        Do we have missing values. How strong is the correlation between 
#        different features.
#      - Feature selection. Which variables shall be included in the model. 
#        Assessment of their predictive power independently and in groups of 
#        other variables.
#      - Feature engineering. How variables should be encoded. Factors may need 
#        some recoding, continuous variables may need some transformations of 
#        discretisation. Groups of variables may need blending.
    	
#   3. Model assembly: Activities needed for the construction of the model.
#      - Model selection. There is an increasing number of different procedures 
#        for model construction. Further, new models can be created as a 
#        combination of other models.
#      - Parameters tuning. Most procedures for model constructions are 
#        parametrized. Different strategies may be employed to identify best set 
#        of parameters.
    
#   4. Model audit: Activities needed for monitoring model performance, fairness 
#      and stability.
#      - Data validation. Is there a change in the structure of the data,
#        distributions of variables or relation structure?
#      - Model validation. Is there a change in model performance between 
#        training, test and validation data. Is there a change in performance in 
#        the new batch of validation data? Is there any issue in model fairness?
#      - Model benchmarking. How good is a given model in comparison to other models?
    
#   5. Model delivery: Activities needed for model release.
#      - Model deployment. Model needs to be put in the production environment 
#        keeping same version of dependent libraries.
#      - Documentation. Decisions that lead to the final model needs to be saved. 
#        Model and data used for training should be clearly defined. 
#        Documentation shall be gathered and expanded through the full model lifetime.
#      - Communication. Reports, charts, tables, all artifact that are used to 
#        consult the model with the client in a easy to understand way.
#      - Model updates. With new batches of data one may plan model retraining to 
#        adjust for recent data. This phase is more common for time-series models.


# * The DrWhy.AI Family ---------------------------------------------------

# Packages in the DrWhy.AI family of models may be divided into four classes.
 
#  - Model adapters: Predictive models created with different tools have 
#    different structures, and different interfaces. Model adapters create 
#    uniform wrappers. This way other packages may operate on models in an 
#    unified way. DALEX is a lightweight package with generic interface. 
#    DALEXtra is a package with extensions for heavyweight interfaces like 
#    scikitlearn, h2o, mlr.
 
#  - Model agnostic explainers: These packages implement specific methods for 
#    model exploration. They can be applied to a single model or they can 
#    compare different models. ingredients implements variable specific 
#    techniques like Ceteris Paribus, Partial Dependency, Permutation based 
#    Feature Importance. iBreakDown implements techniques for variable 
#    attribution, like Break Down or SHAPley values. auditor implements 
#    techniques for model validation, residual diagnostic and performance 
#    diagnostic.
 
#  - Model specific explainers: These packages implement model specific 
#    techniques. randomForestExplainer implements techniques for exploration of 
#    randomForest models. EIX implements techniques for exploration of gbm and 
#    xgboost models. cr19 implements techniques for exploration of survival models.
 
#  - Automated exploration: These packages combine series of model exploration 
#    techniques and produce an automated report of website for model exploration. 
#    modelStudio implements a dashboard generator for local and global 
#    interactive model exploration. modelDown implements a HTML website generator 
#    for global model cross comparison.

  
# * Installs --------------------------------------------------------------

install.packages("DALEX")


# * Loading ---------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")



# Data --------------------------------------------------------------------

# We will illustrate the methods presented by using two datasets related to:
# 	- predicting probability of survival for passengers of the RMS Titanic
#   - predicting prices of apartments in Warsaw.

# The first dataset will be used to illustrate the application of the techniques 
# in the case of a predictive (classification) model for a binary dependent 
# variable. The second dataset will be used to illustrate the exploration of 
# prediction models for a continuous dependent variable. 


# * Titanic ---------------------------------------------------------------

# The stablelearner package in R includes a data frame with information about 
# passengers’ characteristics. The dataset, after some data cleaning and variable 
# transformations, is also available in the DALEX package for R and in the dalex 
# library for Python. In particular, the titanic data frame contains 2207 
# observations (for 1317 passengers and 890 crew members) and nine variables:
#  - gender, person’s (passenger’s or crew member’s) gender, a factor 
#    (categorical variable) with two levels (categories): “male” (78%) and 
#    “female” (22%)
#  - age, person’s age in years, a numerical variable; the age is given in 
#    (integer) years, in the range of 0–74 years
#  - class, the class in which the passenger travelled, or the duty class of a 
#    crew member; a factor with seven levels: “1st” (14.7%), “2nd” (12.9%), “3rd” 
#    (32.1%), “deck crew” (3%), “engineering crew” (14.7%), “restaurant staff” 
#    (3.1%), and “victualling crew” (19.5%)
#  - embarked, the harbor in which the person embarked on the ship, a factor with 
#    four levels: “Belfast” (8.9%), “Cherbourg” (12.3%), “Queenstown” (5.6%), and
#    “Southampton” (73.2%)
#  - country, person’s home country, a factor with 48 levels; the most common 
#    levels are “England” (51%), “United States” (12%), “Ireland” (6.2%), and 
#    “Sweden” (4.8%)
#  - fare, the price of the ticket (only available for passengers; 0 for crew 
#  	 members), a numerical variable in the range of 0–512
#  - sibsp, the number of siblings/spouses aboard the ship, a numerical variable 
#    in the range of 0–8
#  - parch, the number of parents/children aboard the ship, a numerical variable 
#    in the range of 0–9
#  - survived, a factor with two levels: “yes” (67.8%) and “no” (32.2%) 
#    indicating whether the person survived or not.

titanic <- DALEX::titanic %>% 
	as_tibble() %>% 
	drop_na() %>% 
	mutate(
		parch_cat = NA_character_,
		age_cat = NA_character_,
		sibsp_cat = NA_character_,
		fare_cat = NA_character_,
		country_cat = as.character(country)
	) %>% 
	mutate(
		age_cat = case_when(
			age >= 0 & age <= 5 ~ "0-5",
			age >= 6 & age <= 10 ~ "6-10",
			age >= 11 & age <= 20 ~ "11-20",
			age >= 21 & age <= 30 ~ "21-30",
			age >= 31 ~ "30+",
			TRUE ~ as.character(age)
		),
		parch_cat = case_when(
			parch >= 2 & parch <= 3 ~ "2-3",
			parch > 3 ~ "4+",
			TRUE ~ as.character(parch)
		),
		sibsp_cat = case_when(
			sibsp >= 2 & sibsp <= 3 ~ "2-3",
			sibsp > 3 ~ "4+", 
			TRUE ~ as.character(sibsp)
		),
		fare_cat = case_when(
			fare < 1 ~ "0",
			fare >= 1 & fare < 10 ~ "1-10",
			fare >= 10 & fare < 25 ~ "10-25",
			fare >= 25 & fare < 50 ~ "25-50",
			fare >= 50 ~ "50+",
			TRUE ~ as.character(fare)
		),
		country_cat = case_when(
			country_cat != "England" & country_cat != "United States" & country_cat != "Ireland" & country_cat != "Sweden" ~ "Others",
			TRUE ~ country_cat
		)
	) 
titanic

(
	ggplot(data = titanic) + geom_mosaic(aes(x = product(gender), fill = survived)) +
		ggplot(data = titanic) + geom_mosaic(aes(x = product(age_cat), fill = survived)) 
) / (
	ggplot(data = titanic) + geom_mosaic(aes(x = product(parch_cat), fill = survived)) + 
		ggplot(data = titanic) + geom_mosaic(aes(x = product(sibsp_cat), fill = survived))
) / (
	ggplot(data = titanic) + geom_mosaic(aes(x = product(class), fill = survived)) +
		ggplot(data = titanic) + geom_mosaic(aes(x = product(fare_cat), fill = survived))
) / (
	ggplot(data = titanic) + geom_mosaic(aes(x = product(embarked), fill = survived)) +
		ggplot(data = titanic) + geom_mosaic(aes(x = product(country_cat), fill = survived))
) +
plot_layout(guides = "collect") &
	ggplot2::theme_minimal() 


# * Apartment Prices ------------------------------------------------------

# Predicting house prices is a common exercise used in machine-learning courses. 
# Various datasets for house prices are available at websites like Kaggle or 
# UCI Machine Learning Repository.
# 
# We will work with an interesting variant of this problem. The apartments 
# dataset contains simulated data that match key characteristics of real 
# apartments in Warsaw, the capital of Poland. However, the dataset is created 
# in a way that two very different models, namely linear regression and random 
# forest, offer almost exactly the same overall accuracy of predictions. The 
# natural question is then: which model should we choose? We will show that 
# model-explanation tools provide important insight into the key model 
# characteristics and are helpful in model selection.
# 
# The dataset is available in the DALEX package in R and the dalex library in 
# Python. It contains 1000 observations (apartments) and six variables:
# 	
# 	- m2.price, apartment’s price per square meter (in EUR), a numerical variable 
#     in the range of 1607–6595
#   - construction.year, the year of construction of the block of flats in which 
#     the apartment is located, a numerical variable in the range of 1920–2010
#   - surface, apartment’s total surface in square meters, a numerical variable 
#     in the range of 20–150
#   - floor, the floor at which the apartment is located (ground floor taken to 
#   	be the first floor), a numerical integer variable with values ranging from 
#     1 to 10
#   - no.rooms, the total number of rooms, a numerical integer variable with 
#     values ranging from 1 to 6
#   - district, a factor with 10 levels indicating the district of Warsaw where 
#     the apartment is located.
# 
# Models considered for this dataset will use m2.price as the (continuous) 
# dependent variable. Models’ predictions will be validated on a set of 9000 
# apartments included in data frame apartments_test.
# 
# Note that, usually, the training dataset is larger than the testing one. In 
# this example, we deliberately use a small training set, so that model selection 
# may be more challenging.

apartments <- DALEX::apartments %>% 
	as_tibble() %>% 
	drop_na() 
apartments

(
	ggplot(data = apartments, aes(x = m2.price)) + geom_histogram()
) / (
	ggplot(data = apartments, aes(x = construction.year, y = m2.price)) + geom_jitter() + geom_smooth(se = FALSE) +
		ggplot(data = apartments, aes(x = surface, y = m2.price)) + geom_jitter() + geom_smooth(se = FALSE)  
) / (
	ggplot(data = apartments, aes(x = factor(floor), y = m2.price)) + geom_jitter() + geom_boxplot(alpha = 0.3) +
		ggplot(data = apartments, aes(x = factor(no.rooms), y = m2.price)) + geom_jitter() + geom_boxplot(alpha = 0.3)  
) / (
	ggplot(data = apartments, aes(x = district, y = m2.price)) + geom_jitter() + geom_boxplot(alpha = 0.3)
) +
	plot_layout(guides = "collect") &
	ggplot2::theme_minimal() 



# Modelling ---------------------------------------------------------------

# * Titanic Models --------------------------------------------------------

# ** Logistic Regression


# ** Random Forest


# ** XGBoost


# ** SVM





# * Apartment Prices Models -----------------------------------------------

# ** Linear Regression


# ** Random Forest


# ** XGBoost


# ** SVM



# Local Explainers --------------------------------------------------------



# Global Explainers -------------------------------------------------------



































