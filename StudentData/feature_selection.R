
library(tidyverse)
library(caret)
library(dplyr)
library(ggplot2)
library(forcats)
library(ROSE)
library(doParallel)



#declare index outcome events so that I can load the function in. stupid global variable nonsense
indexEvent = "borrow"
outcomeEvent = "deposit"
source("~/DAR-DeFi-LTM-F24/DeFi_source/dataLoader.R")
source("~/DAR-DeFi-LTM-F24/DeFi_source/survivalData_pipeline/data_preprocessing.R")
source("~/DAR-DeFi-LTM-F24/DeFi_source/survivalData_pipeline/get_classification_cutoff.R")

# remove the superflous data loading
rm(test)
rm(train)

# ================================ DATA LOADERS =====================================

# Copy of this function from dataLoader.R because for some reason the source depends on global variables
loadSurvivalDataset = function(indexEvent, outcomeEvent, 
                                dataPath = "/data/IDEA_DeFi_Research/Data/Survival_Data_F24/", 
                                X_path = "/X_train/",
                                y_path = "y_train.rds"){
  
  
  X_files <- list.files(paste0(dataPath, str_to_title(indexEvent), X_path), pattern = NULL, all.files = FALSE, full.names = TRUE)
  X = data.frame()
  
  for(file in X_files){
    X <- X %>%
      bind_rows(read_rds(file)) %>%
      select(where(not_all_na)) %>%
      select(-starts_with("exo")) %>%
      filter(!is.na(id))
  }
  
  y <- read_rds(paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/", y_path)) %>%
    filter(!is.na(id))
  
  
  
  return(inner_join(y, X, by = "id", relationship = "many-to-many"))
  
}

# Gets the train and test data in classification form for the given index and outcome
load_train_test = function(index, outcome){
  
  # Load in the base datasets
  train = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_train/", y_path = "y_train.rds")
  test = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_test/", y_path = "y_test.rds")
  
  # Grab the optimal cutoff for the index outcome pair
  classification_cutoff = get_classification_cutoff(indexEvent, outcomeEvent)
  
  # Apply the cutoff to the data, creating the classification data
  trainData <- data_processing(train, classification_cutoff)
  testData <- data_processing(test, classification_cutoff)
  
  return(list(train = trainData, test = testData))
  
}

# ================================ RFE =====================================

RFE_driver = function(indexEvent, outcomeEvent, 
                      isVerbose = FALSE, parallel = TRUE, downsample = -1) {

  print(paste("RFE on", indexEvent, "-", outcomeEvent))
  
  # Load the survival dataset
  data = load_train_test(indexEvent, outcomeEvent)
  if(isVerbose) { print("    raw data loaded") }
  
  # Error out if dataset isn't loaded correctly
  if (!exists("data")) {
    stop(paste("RFE_driver: Dataset for", indexEvent, "-", outcomeEvent, "failed to load."))
  }
  
  if(downsample > 0){
    downsampled_data <- data$train %>%
      group_by(event) %>%                                   # Group by the outcome variable
      sample_n(size = downsample, replace = FALSE) %>%      # Sample 500 rows from each group
      ungroup()                                             # Remove grouping
    
    
    data$train = downsampled_data
    if(isVerbose) { print("    data downsampled") }
  }
  
  # separate the response and predictor variables
  response_var = data$train %>% pull(event)
  response_var = as.factor(response_var)
  
  predict_vars <- data$train %>%
    mutate(
      reserve = fct_lump(reserve, n = 20), # Keep top 5 levels, lump the rest
      userReserveMode = fct_lump(userReserveMode, n = 20) # Adjust as needed
    ) %>%
    select(-event) # Exclude 'event' if still needed
  
  # scale and normalize the data
  preProcValues = preProcess(predict_vars, method = c("center", "scale"))
  predict_vars = predict(preProcValues, predict_vars)
  
  if(isVerbose) { print("    data preprocessed") }
  
  
  # Ensure x and y are aligned
  if (nrow(predict_vars) != length(response_var)) {
    stop("Mismatch between number of rows in predictors_transformed and length of response_var.")
  }
  
  # Set up control for rfe, which sets parameters for how the selection is performed
  control = rfeControl(
    functions = rfFuncs,  # Random forest functions
    method = "cv",        # Cross-validation
    verbose = isVerbose   # Should it print out progress etc?
  )
  
  # Set up parallel compute for RFE, this thangamajang takes forever
  start = Sys.time()
  if(parallel){
    if(isVerbose) { print("    initializing cores")}
    cl <- makePSOCKcluster(5) # 5 cores
    registerDoParallel(cl)
  }
  
  
  # Perform Recursive Feature Elimination
  if(isVerbose) {print("    starting RFE")}
  rfe_model = rfe(
    x = predict_vars,
    y = response_var,
    sizes = seq(1, ncol(predict_vars), by = 5),      # Subset sizes for quick runtime
    rfeControl = control
  )
  
  if(parallel){
    stopCluster(cl) # Stop parallel compute here
    registerDoSEQ()
  }

  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  print(paste("    RFE completed in:", round(elapsed, 2), "seconds"))
  
  
  # Print and return the RFE model
  if(isVerbose) { print(rfe_model) }
  
  return(rfe_model)
}

RFE_selection = function(indexEvents, outcomeEvents, downsample = -1){
  
  # Initialize an empty data frame to store results
  rfe_results <- list(index = character(0), outcome = character(0), model = list())
  
  for (indexEvent in indexEvents){
    for (outcomeEvent in outcomeEvents){
      # Do not compare an event with itself
      if( indexEvent == outcomeEvent ){
        next
      }
      
      rfe_result = tryCatch(
        RFE_driver(indexEvent, outcomeEvent, isVerbose = FALSE, parallel = TRUE, downsample = downsample),
        error = function(e) {
          # Print error message and return NULL instead of using next
          print(paste("Error in RFE for", indexEvent, "-", outcomeEvent, ":", e$message))
          return(NULL)
        })
      
      # If rfe_result is not NULL, store the results
      if (!is.null(rfe_result)) {
        
        # Append the new values to the list
        rfe_results$index <- c(rfe_results$index, indexEvent)       # Append to index field
        rfe_results$outcome <- c(rfe_results$outcome, outcomeEvent) # Append to outcome field
        rfe_results$model <- append(rfe_results$model, list(rfe_result))  # Append to model field
      }
    }
  }
  
  return (rfe_results)
}

# ================================ SBF =====================================

SBF_driver = function(indexEvent, outcomeEvent, 
                      isVerbose = FALSE, parallel = TRUE, downsample = -1) {
  
  print(paste("SBF on", indexEvent, "-", outcomeEvent))
  
  # Load the survival dataset
  data = load_train_test(indexEvent, outcomeEvent)
  if(isVerbose) { print("    raw data loaded") }
  
  # Error out if dataset isn't loaded correctly
  if (!exists("data")) {
    stop(paste("SBF_driver: Dataset for", indexEvent, "-", outcomeEvent, "failed to load."))
  }
  
  if(downsample > 0){
    downsampled_data <- data$train %>%
      group_by(event) %>%                                   # Group by the outcome variable
      sample_n(size = downsample, replace = FALSE) %>%      # Sample 500 rows from each group
      ungroup()                                             # Remove grouping
    
    
    data$train = downsampled_data
    if(isVerbose) { print("    data downsampled") }
  }
  
  
  response_var = data$train %>% pull(event)
  response_var = as.factor(response_var)
  predict_vars = data$train %>% select(-event) #, -reserve, -userReserveMode)
  
  # Convert Factors to dummies
  dummies = dummyVars(~ ., data = predict_vars)
  predict_vars = as.data.frame(predict(dummies, newdata = predict_vars))
  
  # Ensure x and y are aligned
  if (nrow(predict_vars) != length(response_var)) {
    stop("Mismatch between number of rows in predictors_transformed and length of response_var.")
  }
  
  
  # Set up parallel compute for SBF
  start = Sys.time()
  if(parallel){
    if(isVerbose) { print("    initializing cores")}
    cl <- makePSOCKcluster(5) # 5 cores
    registerDoParallel(cl)
  }
  
  
  sbf_control = sbfControl(functions = rfSBF, method = "cv", number = 10)
  
  # Perform Recursive Feature Elimination
  if(isVerbose) {print("    starting SBF")}
  sbf_model = sbf(
    x = predict_vars,
    y = response_var,
    sbfControl = sbf_control
  )
  
  if(parallel){
    stopCluster(cl) # Stop parallel compute here
    registerDoSEQ()
  }
  
  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  print(paste("    SBF completed in:", round(elapsed, 2), "seconds"))
  
  
  # Print and return the RFE model
  if(isVerbose) { print(sbf_model) }
  
  return(sbf_model)
}

SBF_selection = function(indexEvents, outcomeEvents, downsample = -1){
  

  # Initialize the list with the required fields (empty vectors for now)
  sbf_results <- list(index = character(0), outcome = character(0), model = list())
  
  
  for (indexEvent in indexEvents){
    for (outcomeEvent in outcomeEvents){
      # Do not compare an event with itself
      if( indexEvent == outcomeEvent ){
        next
      }
      
      sbf_result = tryCatch(
        SBF_driver(indexEvent, outcomeEvent, isVerbose = FALSE, parallel = TRUE, downsample = downsample),
        error = function(e) {
          # Print error message and return NULL instead of using next
          print(paste("Error in SBF for", indexEvent, "-", outcomeEvent, ":", e$message))
          return(NULL)
        })
      
      # If rfe_result is not NULL, store the results
      if (!is.null(sbf_result)) {
        
        # Append the new values to the list
        sbf_results$index <- c(sbf_results$index, indexEvent)       # Append to index field
        sbf_results$outcome <- c(sbf_results$outcome, outcomeEvent) # Append to outcome field
        sbf_results$model <- append(sbf_results$model, list(sbf_result))  # Append to model field
      }
    }
  }
  
  return (sbf_results)
}


# ================================ Visualizer Functions =====================================


# Define the plotting function for a specified subset
# Function to create a horizontal bar chart with different colors for bars
plot_optimal_feature_counts <- function(rfe_results, feature_list) {
  
  # Extract the optimal feature subsets for all models
  optimal_features <- lapply(rfe_results$model, function(x) x$optVariables)
  
  # Count how many times each feature is selected in the optimal subsets
  feature_counts <- sapply(feature_list, function(feature) {
    sum(sapply(optimal_features, function(opt_set) feature %in% opt_set))
  })
  
  # Create a data frame for plotting
  feature_count_df <- data.frame(Feature = feature_list, Count = feature_counts)
  
  # Horizontal bar plot of optimal feature selection counts with varying colors
  ggplot(feature_count_df, aes(y = reorder(Feature, Count), x = Count, fill = Count)) +
    geom_bar(stat = "identity") +
    scale_fill_gradient(low = "skyblue", high = "darkorange") +
    labs(title = "Optimal Feature Selection Counts",
         x = "Number of Times Selected",
         y = "Features") +
    theme_minimal() +
    theme(axis.text.y = element_text(angle = 0, hjust = 1))
}




# Function to create a table with index-outcome pairs, optimal number of features, and highest accuracy
create_optimal_table <- function(rfe_results) {
  
  # Extract data for each index-outcome pair
  data_list <- lapply(seq_along(rfe_results$model), function(i) {
    model <- rfe_results$model[[i]]
    index_outcome <- paste(rfe_results$index[i], rfe_results$outcome[i], sep = " - ")
    
    model$results %>%
      mutate(IndexOutcome = index_outcome)
  })
  
  # Combine all data into one data frame
  accuracy_data <- do.call(rbind, data_list)
  
  # Identify the optimal number of features and highest accuracy for each index-outcome pair
  optimal_table <- accuracy_data %>%
    group_by(IndexOutcome) %>%
    summarise(
      OptimalFeatures = Variables[which.max(Accuracy)],
      HighestAccuracy = max(Accuracy)
    ) %>%
    ungroup()
  
  return(optimal_table)
}

# Function to display the optimal table using kableExtra
display_optimal_table <- function(optimal_table) {
  library(kableExtra)
  
  # Create a nicely formatted table
  optimal_table %>%
    knitr::kable(format = "html", align = c("l", "c", "c"),
                 col.names = c("Index-Outcome Pair", "Optimal # of Features", "Highest Accuracy"),
                 title = "Optimal Features and Accuracy for Each Model") %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                  full_width = FALSE,
                  position = "center") %>%
    column_spec(1, bold = TRUE) %>%
    column_spec(3, color = "white", background = "darkblue")
}



# Function to plot accuracy vs. number of features for each method
plot_accuracy_vs_features <- function(rfe_results) {
  
  # Extract accuracy and number of features for each index-outcome pair
  data_list <- lapply(seq_along(rfe_results$model), function(i) {
    model <- rfe_results$model[[i]]
    index_outcome <- paste(rfe_results$index[i], rfe_results$outcome[i], sep = " - ")
    
    model$results %>%
      mutate(IndexOutcome = index_outcome)
  })
  
  # Combine all data into one data frame
  accuracy_data <- do.call(rbind, data_list)
  
  # Identify the highest accuracy for each index-outcome pair
  highest_accuracy <- accuracy_data %>%
    group_by(IndexOutcome) %>%
    filter(Accuracy == max(Accuracy)) %>%
    ungroup()
  
  # Plot
  ggplot(accuracy_data, aes(x = Variables, y = Accuracy, color = IndexOutcome, group = IndexOutcome)) +
    geom_line() +
    geom_point(data = highest_accuracy, aes(x = Variables, y = Accuracy), shape = 21, fill = "white", size = 3) +
    labs(title = "Accuracy vs. Number of Features",
         x = "Number of Features",
         y = "Accuracy",
         color = "Index-Outcome") +
    theme_minimal() +
    theme(legend.position = "right",
          axis.text.x = element_text(angle = 45, hjust = 1))
}


# Helper function to extract the model from results (either SBF or RFE) for the given index, outcome
get_model <- function(result, index_value, outcome_value) {
  # Find the indices where both index and outcome match the given values
  match_indices <- which(result$index == index_value & result$outcome == outcome_value)
  
  # If there is exactly one match, return the corresponding model
  if(length(match_indices) == 1) {
    return(result$model[[match_indices]])
  } else {
    # If no match found or multiple matches, return a message
    return("No matching model found for the given index and outcome.")
  }
}

# ================================ Main Code =====================================

indexEvents = c("borrow", "deposit", "repay", "withdraw")
outcomeEvents = c("borrow", "deposit", "repay", "withdraw", "liquidation performed")

# We aren't actually using the 
result_SBF = SBF_selection(indexEvents, outcomeEvents, 5000)

# Run with full subset, all features, downsampling to 1000 datapoints
result_RFE_full_subset = RFE_selection(indexEvents, outcomeEvents, 1000)
saveRDS(result_RFE_full_subset, file="result_RFE_full_subset.rds")


data = load_train_test("borrow", "repay")


all_features = colnames(data$train)


# Define the regex patterns
patterns <- list(
  market = "^market",
  time = "^cos|sin|isWeekend|day|time|weekend|quarter",
  user = "^user"
)

# Filter the words based on the regex patterns
features <- list(
  market = grep(patterns$market, all_features, value = TRUE),
  time = grep(patterns$time, all_features, value = TRUE),
  user = grep(patterns$user, all_features, value = TRUE)
)

# Add remaining features not matched by the above patterns
features$info <- setdiff(all_features, unlist(features))






compute_test_accuracy <- function(rfe_results, test_data) {
  
  # Extract test accuracy for each model
  test_accuracies <- lapply(seq_along(rfe_results$model), function(i) {
    model <- rfe_results$model[[i]]$fit
    index_outcome <- paste(rfe_results$index[i], rfe_results$outcome[i], sep = " - ")
    
    # Predict on test data
    predictions <- predict(model, test_data[, model$call$xvars])
    
    # Compute accuracy
    actuals <- test_data[, as.character(rfe_results$outcome[i])]
    accuracy <- mean(predictions == actuals)
    
    data.frame(IndexOutcome = index_outcome, TestAccuracy = accuracy)
  })
  
  # Combine all test accuracies into a single data frame
  test_accuracy_df <- do.call(rbind, test_accuracies)
  return(test_accuracy_df)
}



compute_test_accuracy(result_RFE_full_subset, data$train)




p = plot_accuracy_vs_features(result_RFE_full_subset)
ggsave("~/DAR-DeFi-LTM-F24/StudentData/feature_images/RFE_acc.png", p, width = 8, height = 6, dpi = 400)

opt_table = create_optimal_table(result_RFE_full_subset)
display_optimal_table(opt_table)
saveRDS(opt_table, file = "~/DAR-DeFi-LTM-F24/StudentData/feature_images/opt_table.rds")


p = plot_optimal_feature_counts(result_RFE_full_subset, features$info)
ggsave("~/DAR-DeFi-LTM-F24/StudentData/feature_images/RFE_info_counts.png", p, width = 8, height = 6, dpi = 400)
p = plot_optimal_feature_counts(result_RFE_full_subset, features$user)
ggsave("~/DAR-DeFi-LTM-F24/StudentData/feature_images/RFE_user_counts.png", p, width = 8, height = 6, dpi = 400)
p = plot_optimal_feature_counts(result_RFE_full_subset, features$market)
ggsave("~/DAR-DeFi-LTM-F24/StudentData/feature_images/RFE_market_counts.png", p, width = 8, height = 6, dpi = 400)
p = plot_optimal_feature_counts(result_RFE_full_subset, features$time)
ggsave("~/DAR-DeFi-LTM-F24/StudentData/feature_images/RFE_time_counts.png", p, width = 8, height = 6, dpi = 400)


