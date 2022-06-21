# Helper Functions


# Function to check packages already loaded into NAMESPACE
check_namespace <- function(pkgs) {

  pkgs_notloaded <- pkgs[!pkgs %in% loadedNamespaces()]
  if (length(pkgs_notloaded) == 0) {
    res <- NULL
  } else {
    res <- pkgs_notloaded
  }
  return(res)

}


# Function to install and load the specified packages
install_and_load <- function(pkgs, repos = getOption("repos")) {

  pkgs_inst <- pkgs[!pkgs %in% installed.packages()]

  if (length(pkgs_inst) == 0) {
    lapply(pkgs, library, character.only = TRUE, quietly = TRUE)
    check_res <- check_namespace(pkgs)
    if (is.null(check_res)) {
      res <- "All packages correctly installed and loaded."
    } else {
      res <- paste0(
        "Problems loading packages ",
        paste0(check_res, collapse = ", "),
        "."
      )
    }

  } else {

    inst_res <- vector("character", length(pkgs_inst))

    for (i in seq_along(pkgs_inst)) {
      inst_res_tmp <- tryCatch(
        utils::install.packages(pkgs_inst[i], dependencies = TRUE, repos = repos, quiet = TRUE),
        error = function(e) e,
        warning = function(w) w
      )
      if (!is.null(inst_res_tmp)) {
        inst_res[i] <- inst_res_tmp$message
      }
    }

    pkgs_err <- pkgs_inst[!inst_res == ""]
    if (length(pkgs_err) == 0) {
      lapply(pkgs, library, character.only = TRUE, quietly = TRUE)
      check_res <- check_namespace(pkgs)
      if (is.null(check_res)) {
        res <- "All packages correctly installed and loaded."
      } else {
        res <- paste0(
          "Problems loading packages ",
          paste0(check_res, collapse = ", "),
          "."
        )
      }
    } else {
      pkgs_noerr <- pkgs[!pkgs %in% pkgs_err]
      lapply(pkgs_noerr, library, character.only = TRUE, quietly = TRUE)
      check_res <- check_namespace(pkgs_noerr)
      if (is.null(check_res)) {
        res <- paste0(
          "Problems installing packages ",
          paste0(pkgs_err, collapse = ", "),
          "."
        )
      } else {
        res <- c(
          paste0(
            "Problems installing packages ",
            paste0(pkgs_err, collapse = ", "),
            "."
          ),
          paste0(
            "Problems loading packages ",
            paste0(check_res, collapse = ", "),
            "."
          )
        )
      }
    }

  }

  message(toupper(
    paste0(
      "\n\n\n",
      "\n==================================================================",
      "\nResults:\n ",
      res,
      "\n=================================================================="
    )
  ))
  return(invisible(res))

}


# Function to clean & modify House Prices Dataset
prepare_house_prices <- function(data, target) {

	data_cleaned <- data %>%
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
		dplyr::select( # remove some "useless" variables
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
			c(SalePrice, LotFrontage, LotArea, LowQualFinSF, GrLivArea, PoolArea, MiscVal, SpecialAreas, TotalIndoorSF),
			log1p
		)) %>% 
		relocate(all_of(target), .before = everything())
	
	return(data_cleaned)
		
}


# Function to evaluate models
evaluate_model <- function(prediction_results, mode, type) {
	
	if (mode == "regression") {
		res <- prediction_results %>% 
			metrics(truth = Actual, estimate = Pred) %>% 
			dplyr::select(.metric, .estimate) %>% 
			set_names("Metric", "Estimate") %>% 
			add_column("Type" = type)
	} else {
		res_confmat <- prediction_results %>%
			conf_mat(truth = Actual, estimate = Pred)
		res_metrics <- bind_rows(
			prediction_results %>% metrics(truth = Actual, estimate = Pred),
			prediction_results %>% roc_auc(truth = Actual, estimate = Prob_Low)
		) %>% 
			dplyr::select(.metric, .estimate) %>% 
			set_names("Metric", "Estimate") %>% 
			add_column("Type" = type)
		res_roc <- prediction_results %>%
			roc_curve(truth = Actual, Prob_Low) %>% 
			add_column("Type" = type)
		res <- list("confusion" = res_confmat$table, "metrics" = res_metrics, "roc" = res_roc)
	}
	
	return(res)
	
}


# Function to plot models
plot_model <- function(prediction_results, mode) {
	
	if (mode == "regression") {
		p <- prediction_results %>% 
			dplyr::select(-Type) %>% 
			mutate(id = 1:n()) %>% 
			ggplot(aes(x = id)) +
			geom_point(aes(y = Actual, col = "Actual")) +
			geom_point(aes(y = Pred, col = "Pred")) +
			# geom_errorbar(aes(ymin = Lower, ymax = Upper), width = .2, col = "red") +
			scale_color_manual(values = c("black", "red")) +
			labs(x = "", y = "", col = "") +
			theme_minimal()
	} else {
		p <- prediction_results$roc %>%
			ggplot(aes(x = 1 - specificity, y = sensitivity)) +
			geom_path() +
			geom_abline(lty = 3) +
			coord_equal() + 
			theme_minimal() 
	}
	
	res <- plotly::ggplotly(p)
	return(res)
	
}


# Function to fit, evaluate and plot model results on test set
calibrate_evaluate_plot <- function(
	model_fit, 
	y, 
	mode, 
	type = "testing", 
	print = TRUE
) {
	
	if (type == "testing") {
		new_data <- testing(splits)
	} else {
		new_data <- training(splits)
	}
	
	if (mode == "regression") {
		pred_res <- model_fit %>% 
			augment(new_data) %>%
			dplyr::select(all_of(y), .pred) %>% 
			set_names(c("Actual", "Pred")) %>% 
			# bind_cols(
			# 	model_fit %>% 
			# 		predict(new_data, type = "conf_int") %>%
			# 		set_names(c("Lower", "Upper")) 
			# ) %>% 
			add_column("Type" = type)
	} else {
		pred_res <- model_fit %>% 
			augment(new_data) %>%
			dplyr::select(all_of(y), contains(".pred")) %>% 
			set_names(c("Actual", "Pred", "Prob_Low", "Prob_High")) %>% 
			add_column("Type" = type)
	}
	
	pred_met <- pred_res %>% evaluate_model(mode, type)
	
	if (mode == "regression") {
		pred_plot <- pred_res %>% plot_model(mode)
	} else {
		pred_plot <- pred_met %>% plot_model(mode)
	}
	
	if (print) {
		print(pred_met)
		print(pred_plot)	
	}
	
	res <- list(
		"pred_results" = pred_res,
		"pred_metrics" = pred_met
	)
	
	return(invisible(res))
	
}


# Function to evaluate model performance on train and test
collect_results <- function(model_fit, y, mode, method) {
	
	res <- map(
		c("training", "testing"),
		~ calibrate_evaluate_plot(model_fit, y, mode, type = ., FALSE)
	)
	
	if (mode == "regression") {
		res <- list(
			"pred_results" = map(res, "pred_results") %>% bind_rows() %>% add_column("Method" = method, .before = 1),
			"pred_metrics" = map(res, "pred_metrics") %>% bind_rows() %>% add_column("Method" = method, .before = 1)
		)
	} else {
		res <- list(
			"pred_results" = map(res, "pred_results") %>% bind_rows() %>% add_column("Method" = method, .before = 1),
			"pred_metrics" = map(res, "pred_metrics") %>% map("metrics") %>% bind_rows() %>% add_column("Method" = method, .before = 1),
			"pred_roc" = map(res, "pred_metrics") %>% map("roc") %>% bind_rows() %>% add_column("Method" = method, .before = 1)
		)
	}

	return(res)
	
}


# Function to evaluate stacking ensembles
calibrate_evaluate_stacks <- function(
	model_stacks,
	y, 
	mode, 
	type = "testing", 
	print = TRUE
) {
	
	if (type == "testing") {
		new_data <- testing(splits)
	} else {
		new_data <- training(splits)
	}
	
	pred_res <- predict(model_stacks, new_data, members = TRUE) %>% 
		bind_cols(
			new_data %>% dplyr::select(all_of(y)),
			.
		)  %>% 
		rename("Actual" = all_of(y)) %>% 
		rename_with(~ str_replace_all(., ".pred", "Pred"))
	
	if (mode == "regression") {
		metric <- rmse
	} else {
		pred_res <- pred_res %>% rename("Pred" = "Pred_class")
		metric <- accuracy
	}

	pred_met <-	colnames(pred_res) %>%
		map_dfr(metric, truth = Actual, data = pred_res) %>%
		mutate(
			Member = colnames(pred_res),
			Member = ifelse(Member == "Pred", "stack", Member)
		) %>% 
		dplyr::slice(-1) %>% 
		dplyr::select(Member, .metric, .estimate) %>% 
		set_names("Member", "Metric", "Estimate") %>% 
		add_column("Type" = type)
	
	if (print) {
		print(pred_met)
	}
	
	res <- list(
		"pred_results" = pred_res,
		"pred_metrics" = pred_met
	)
	
	return(invisible(res))
	
}


# Mode 
stat_mode <- function(x) {
	ux <- unique(x)
	mode <- ux[which.max(tabulate(match(x, ux)))]
	return(mode)
} 


# Function to finalize and fitting models
finalizing_and_fitting <- function(workfl, param) {
	res <- finalize_workflow(workfl, param) %>% 
		last_fit(splits)
	return(res)
}


# Function to compute simple ensembles
simple_ensemble <- function(
	model_results, 
	workflows,
	y,
	mode, 
	ensemble_fun,
	print = TRUE
) {
	
	methods <- names(model_results)
	
	if (mode == "regression") {
		
		best_models <- model_results %>% map(select_best, metric = "rmse")
		wrkfls_fit_final <-	map2(workflows, best_models, finalizing_and_fitting)
		
		pred_res <- wrkfls_fit_final %>% 
			map(collect_predictions) %>% 
			map(".pred") %>% 
			bind_cols() %>% 
			rename_with(~ str_c("Pred_", .)) %>% 
			dplyr::rowwise() %>% 
			mutate(Ensemble = ensemble_fun(c_across(everything())), .before = 1) %>% 
			ungroup() 

		ensemble_met <- rmse_vec(
			truth = wrkfls_fit_final[[1]] %>% collect_predictions() %>% pull(y), 
			estimate = pred_res$Ensemble
		) 
		members_met <- wrkfls_fit_final %>% 
			map(collect_metrics) %>% 
			map2(
				methods, 
				~ mutate(.x, Member = .y) %>% 
					dplyr::filter(.metric == "rmse") %>% 
					dplyr::select(Member, .metric, .estimate) %>% 
					set_names(c("Member", "Metric", "Estimate"))
			) %>%
			bind_rows()
		pred_met <- bind_rows(
			tibble("Member" = "ensemble", "Metric" = "rmse", "Estimate" = ensemble_met),
			members_met
		)
		
	} else {
		
		best_models <- model_results %>% map(select_best, metric = "accuracy")
		wrkfls_fit_final <-	map2(workflows, best_models, finalizing_and_fitting)
		
		pred_res <- wrkfls_fit_final %>% 
			map(collect_predictions) %>% 
			map(".pred_class") %>% 
			bind_cols() %>% 
			rename_with(~ str_c("Pred_", .)) %>% 
			dplyr::rowwise() %>% 
			mutate(Ensemble = ensemble_fun(c_across(everything())), .before = 1) %>% 
			ungroup() 
		
		ensemble_met <- accuracy_vec(
			truth = wrkfls_fit_final[[1]] %>% collect_predictions() %>% pull(y), 
			estimate = pred_res$Ensemble
		) 
		members_met <- wrkfls_fit_final %>% 
			map(collect_metrics) %>% 
			map2(
				methods, 
				~ mutate(.x, Member = .y) %>% 
					dplyr::filter(.metric == "accuracy") %>% 
					dplyr::select(Member, .metric, .estimate) %>% 
					set_names(c("Member", "Metric", "Estimate"))
			) %>%
			bind_rows()
		pred_met <- bind_rows(
			tibble("Member" = "ensemble", "Metric" = "accuracy", "Estimate" = ensemble_met),
			members_met
		)
		
	}
	
	if (print) {
		print(pred_met)
	}
	
	res <- list(
		"pred_results" = pred_res,
		"pred_metrics" = pred_met
	)
	
	return(invisible(res))
	
}


# Function to evaluate h2o models
evaluate_h2o <- function(h2o_model, metric) {
	
	res <- h2o.performance(h2o_model, test)@metrics[[metric]]
	return(res)
	
}
