#FUNCTION TO CALCULATE BRIER AND INTEGRATED BRIER SCORES

compute_IBS <- function(model, model_type, train_data, test_data, times, status_col, time_col, return_brier_scores = FALSE, test_xgb = NULL) {
  # Load necessary libraries
  library(survival)
  library(ipred)
  
  # Ensure that status and time columns are correctly specified
  status_train <- train_data[[status_col]]
  time_train <- train_data[[time_col]] / 86400  # Adjust time units if necessary
  
  status_test <- test_data[[status_col]]
  time_test <- test_data[[time_col]] / 86400    # Adjust time units if necessary
  
  # Prepare the survival object for the test data
  Surv_test <- Surv(time_test, status_test)
  
  # Estimate the censoring distribution using the Kaplan-Meier estimator
  km_fit <- survfit(Surv(time_train, status_train == 0) ~ 1)
  
  # Compute the censoring probabilities at the specified times
  km_times <- km_fit$time
  km_surv <- km_fit$surv
  
  G_times <- approx(
    x = km_times,
    y = km_surv,
    xout = times,
    method = "constant",
    rule = 2
  )$y
  
  # Initialize the survival probabilities matrix
  surv_probs <- matrix(NA, nrow = nrow(test_data), ncol = length(times))
  
  # Compute survival probabilities based on model type
  if (model_type == "cox" || model_type == "gbm") {
    # Predict linear predictors for the training and test data
    if (model_type == "gbm") {
      library(gbm)
      best_trees <- gbm.perf(model, method = "test", plot.it = FALSE)
      lp_train <- predict(model, newdata = train_data, n.trees = best_trees, type = "link")
      lp_test <- predict(model, newdata = test_data, n.trees = best_trees, type = "link")
    } else if (model_type == "cox") {
      lp_train <- predict(model, newdata = train_data, type = "lp")
      lp_test <- predict(model, newdata = test_data, type = "lp")
    }
    
    # Fit a Cox model with the linear predictors as an offset (for GBM)
    if (model_type == "gbm") {
      cox_fit <- coxph(Surv(time_train, status_train) ~ offset(lp_train), data = train_data)
    } else {
      cox_fit <- model
    }
    
    # Estimate the baseline cumulative hazard function
    basehaz <- basehaz(cox_fit, centered = FALSE)
    
    # Interpolate the baseline hazard at the specified time points
    basehaz_times <- approx(
      x = basehaz$time,
      y = basehaz$hazard,
      xout = times,
      method = "constant",
      rule = 2
    )$y
    
    # Compute survival probabilities for the test data
    surv_probs <- exp(-exp(lp_test) %o% basehaz_times)
    
  } else if (model_type == "aft") {
    # For AFT models, use psurvreg to compute survival probabilities
    lp_test <- predict(model, newdata = test_data, type = "lp")
    for (i in seq_along(times)) {
      t <- times[i]
      surv_probs[, i] <- 1 - psurvreg(
        q = t,
        mean = lp_test,
        scale = model$scale,
        distribution = model$dist
      )
    }
    
  } else if (model_type == "km") {
    # For Kaplan-Meier, survival probabilities are the same for all individuals
    km_model <- survfit(Surv(time_train, status_train) ~ 1)
    km_times <- km_model$time
    km_surv <- km_model$surv
    
    # Interpolate the survival probabilities at the specified time points
    surv_probs_single <- approx(
      x = km_times,
      y = km_surv,
      xout = times,
      method = "constant",
      rule = 2
    )$y
    
    # Repeat the survival probabilities for all individuals
    surv_probs <- matrix(rep(surv_probs_single, each = nrow(test_data)), nrow = nrow(test_data))
    
  } else if (model_type == "xgboost") {
    # XGBoost Survival Model
    library(xgboost)
    library(Matrix)

    # Predict the median survival times
    pred_median <- predict(xgb_model, newdata = test_xgb)
    
    # Compute survival probabilities at specified times
    for (i in seq_along(times)) {
      t <- times[i]
      # Assuming an exponential distribution for simplicity
      # S(t) = exp(-t / lambda), where lambda is the predicted median survival time divided by log(2)
      lambda <- pred_median / log(2)
      surv_probs[, i] <- exp(-t / lambda)
    }
    
  } else if (model_type == "rsf") {
    # Random Survival Forest Model
    library(randomForestSRC)
    
    # Predict survival probabilities at specified times
    rsf_pred <- predict(model, newdata = test_data, importance = "none", proximity = FALSE, outcome = "train", membership = FALSE, type = "surv", times = times)
    
    # Extract survival probabilities
    surv_probs <- rsf_pred$survival  # Rows correspond to individuals, columns to times
    
  } else {
    stop("Invalid model type. Choose from 'cox', 'gbm', 'aft', 'km', 'xgboost', or 'rsf'.")
  }
  
  # Compute the Brier scores
  brier_scores <- numeric(length(times))
  for (i in seq_along(times)) {
    t <- times[i]
    surv_prob_t <- surv_probs[, i]
    brier_scores[i] <- sbrier(
      obj = Surv_test,
      pred = surv_prob_t,
      btime = t
    )
  }
  
  # Compute the integrated Brier score using the trapezoidal rule
  dtimes <- diff(times)
  avg_brier <- (brier_scores[-1] + brier_scores[-length(brier_scores)]) / 2
  area_under_curve <- sum(dtimes * avg_brier)
  integrated_brier_score <- area_under_curve / (max(times) - min(times))
  
  # Return the IBS or both IBS and Brier scores
  if (return_brier_scores) {
    return(list(IBS = integrated_brier_score, brier_scores = brier_scores))
  } else {
    return(integrated_brier_score)
  }
}