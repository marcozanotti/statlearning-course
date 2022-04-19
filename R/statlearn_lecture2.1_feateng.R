# Statistical Learning ----

# Lecture 2.1: Features Engineering - The House Prices Dataset ------------
# 2021/2022
# Marco Zanotti

# Goals:
# - House Prices Dataset
# - Feature Engineering



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")



# Data --------------------------------------------------------------------

# Throughout the ML & DL lectures, the Ames House Prices Dataset is used.
# Hence, this lecture is focused on the exploration and the preparation
# of the dataset for both regression & classification tasks. 

# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

house_prices <- read_csv("data/ames.csv") 
house_prices %>% glimpse()



# Features Engineering ----------------------------------------------------

# Let's explore the data and look at the description file.
house_prices %>% skimr::skim() 
# 1460 observations spread across 81 variables
# 38 numeric variables and 43 character variables

# character type is useless for analysis, hence convert to factor
house_prices %>% 
	mutate(across(where(is.character), as.factor)) %>% 
	skimr::skim()


house_prices %>%
	mutate( # missing values imputation
		LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage),
		Alley = ifelse(is.na(Alley), "No", "Yes"),
		MasVnrArea = ifelse(is.na(MasVnrArea), 0, MasVnrArea),
		Fence = ifelse(is.na(Fence), "No", "Yes")
	) %>% 
	mutate( # re-coding variables and create new features
		LotShape = ifelse(LotShape == "Reg", "Reg", "Irreg"),
		LandContour = ifelse(LandContour == "Lvl", "Lvl", "NoLvl"),
		LotConfig = ifelse(LotConfig == "Inside", "Ins", "NoIns"),
		RelevantNearZones = ifelse(Condition1 == "Norm", "No", "Yes"),
		BldgType = ifelse(BldgType == "1Fam", "1Fam", "Others"),
		HouseStyle = case_when(
			str_detect(HouseStyle, "1") ~ "Type1",
			str_detect(HouseStyle, "2") ~ "Type2",
			TRUE ~ "Others"
		),
		HouseQuality = (OverallQual + OverallCond) / 2,
		HouseAge = YrSold - YearBuilt,
		Remod = ifelse(YearBuilt == YearRemodAdd, "No", "Yes"),
		RoofStyle = ifelse(RoofStyle == "Gable", RoofStyle, "Others"),
		Masonry = ifelse(MasVnrArea > 0, "Yes", "No"),
		ExternalQuality = (
			as.numeric(factor(ExterQual, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE))  +
				as.numeric(factor(ExterCond, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)) 
		) / 2,
		Foundation = ifelse(Foundation %in% c("CBlock", "PConc"), Foundation, "Others"),
		BasementQuality = (
			as.numeric(factor(BsmtQual, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)) +
				as.numeric(factor(BsmtCond, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE))
		) / 2,
		BasementQuality = ifelse(is.na(BasementQuality), 0,  BasementQuality),
		HeatingQuality = as.numeric(factor(HeatingQC, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)),
		CentralAir = ifelse(CentralAir == "Y", "Yes", "No"),
		Electrical = ifelse(is.na(Electrical) | Electrical != "SBrkr", "Others", Electrical),
		BasementBaths = ifelse(BsmtFullBath > 0 | BsmtHalfBath > 0, "Yes", "No"),
		Baths = FullBath + (HalfBath / 2),
		Beds = BedroomAbvGr, 
		Kitchens = KitchenAbvGr, 
		KitchenQuality = as.numeric(factor(KitchenQual, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)),
		Rooms = TotRmsAbvGrd, 
		Functional = ifelse(Functional == "Typ", Functional, "Others"),
		Garage = ifelse(GarageArea > 0, "Yes", "No"),
		PavedDrive = ifelse(PavedDrive == "Y", "Yes", "No"),
		SpecialAreas = WoodDeckSF + OpenPorchSF + EnclosedPorch + `3SsnPorch` + ScreenPorch,
		YearSold = as.factor(YrSold),
		TotalIndoorSF = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF` + GarageArea
	) %>% 
	select( # remove some "useless" variables
		-c(
			Id, MSSubClass, MSZoning, Street, Utilities, LandSlope, 
			Neighborhood, Condition1, Condition2, OverallQual, OverallCond,
			YearBuilt, YearRemodAdd, RoofMatl, Exterior1st, Exterior2nd,
			MasVnrType, MasVnrArea, ExterQual, ExterCond, 
			BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, BsmtExposure, BsmtFinType1, 
			BsmtFinType2, BsmtQual, BsmtCond, Heating, HeatingQC, BsmtFullBath, BsmtHalfBath,
			FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual, 
			TotRmsAbvGrd, FireplaceQu, GarageType, GarageYrBlt, GarageFinish, 
			GarageCars, GarageQual, GarageCond, WoodDeckSF, OpenPorchSF, 
			EnclosedPorch, `3SsnPorch`, ScreenPorch, PoolQC, MiscFeature,
			MoSold, YrSold, SaleType, SaleCondition,
			TotalBsmtSF, `1stFlrSF`, `2ndFlrSF`, GarageArea
		)
	) %>% 
	mutate(across(where(is.character), as.factor)) %>% # convert character to factor
	mutate(across( # re-scale some numeric to log(x + 1) 
		c(LotFrontage, LotArea, LowQualFinSF, GrLivArea, PoolArea, MiscVal, SpecialAreas, TotalIndoorSF),
		log1p
	))


# create features with the function prepare_house_prices()
house_prices %>% 
	prepare_house_prices("SalePrice") %>% 
	skimr::skim()

house_prices %>% 
	prepare_house_prices("SalePrice") %>% 
	DataExplorer::create_report(y = "SalePrice")



# Regression --------------------------------------------------------------

# These new features may be used to predict the house's sale price.
house_prices_reg <- house_prices %>% prepare_house_prices("SalePrice")

# For this type of problem we will have to create a simple recipe
# with just 1 steps:
recipe(SalePrice ~ ., data = house_prices_reg) %>% 
	step_dummy(all_nominal(), -all_outcomes()) # convert to dummy all categorical variables



# Classification ----------------------------------------------------------

# We may want to use the new features to classify the house's value
# into high and low. To do this we create a new response variable
# based on the sale price. 
# Value = Low if SalePrice <= mean(SalePrice), else Value = High 
house_prices_class <- house_prices %>% 
	prepare_house_prices("SalePrice") %>%  
	mutate( # create Value response
		Value = factor(
			ifelse(SalePrice <= mean(SalePrice), "Low", "High"), 
			levels = c("Low", "High"),  
			ordered = TRUE
		), 
		.before = everything()
	) %>% 
	select(-SalePrice) # remove SalePrice
house_prices_class$Value %>% table() %>% proportions() %>% round(3) * 100

# Finally, also in this case the recipe is based just on 1 step:
recipe(Value ~ ., data = house_prices_class) %>% 
	step_dummy(all_nominal(), -all_outcomes())



# * Save Artifacts --------------------------------------------------------

artifacts_list <- list(
		"reg_data" = house_prices_reg,
		"class_data" = house_prices_class
	)

artifacts_list %>%
	write_rds("artifacts/artifacts_list.rds")


