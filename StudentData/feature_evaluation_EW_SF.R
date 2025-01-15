library(dplyr)
library(data.table)
library(progress)
library(lubridate)
library(caret)
library(grid)
library(viridis)
# source("~/DAR-DeFi-LTM-F24/DeFi_source/Data_Creation_Functions/createSurvData.R")
source("~/DAR-DeFi-LTM-F24/DeFi_source/survivalData_pipeline/get_classification_cutoff.R")
source("~/DAR-DeFi-LTM-F24/DeFi_source/survivalData_pipeline/data_preprocessing.R")

# helper function
not_all_na <- function(x) any(!is.na(x))
`%notin%` <- Negate(`%in%`)


# OUTDATED - use dataLoader.R
# generates survival datasets for all pairs between "deposit", "withdraw", "borrow", "repay" events
create_survival_data <- function() {
  transactions <- readRDS("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/currentFeatures_DO_NOT_OVERWRITE.rds")

  ## Helper functions
  not_all_na <- function(x) any(!is.na(x))
  `%notin%` <- Negate(`%in%`)
  
  # These will be the default settings for subjects unless otherwise specified:
  subjects <- c("user", "reserve")
  indexCovariates <- names(transactions)
  
  outcomeCovariates = c()
  
  basicTransactions <- transactions %>%
    filter(type != "collateral",
           type != "flashLoan",
           type != "swap")
  
  basicEventTypes <- basicTransactions %>%
    dplyr::select(type) %>%
    distinct()
  

  dataPath = "/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/temp_survival_data/"

  # Initialize the progress bar
  totalSteps <- length(basicEventTypes$type) ^ 2 - length(basicEventTypes$type) # for index-outcome pairs excluding same event
  pb <- progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = totalSteps,
    clear = FALSE,
    width = 60
  )
  
  for(indexEvent in basicEventTypes$type){
    # Each index event should have its own directory. We create the appropriate directory
    # here, in case it doesn't already exist:
    
    dir.create(paste0(dataPath, str_to_title(indexEvent), "/"), recursive = TRUE)
    
    for(outcomeEvent in basicEventTypes$type){
      if(indexEvent == outcomeEvent){
        next # Skip the case when the index and outcome events are the same
      }
      
      # Each outcome event should have its own directory within the index event's folder:
      dir.create(paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/"))
      
      survData <- createSurvData(indexEventSet = c(indexEvent), 
                                 outcomeEventSet = c(outcomeEvent), 
                                 basicTransactions, 
                                 subjects = subjects,
                                 indexCovariates = indexCovariates,
                                 outcomeCovariates = outcomeCovariates)
      
      # Drop columns that are entirely NA
      survData <- survData %>% select(where(~ !all(is.na(.))))
      
      saveRDS(survData, paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/","survivalData.rds"))
      
      # Update the progress bar
      pb$tick()
    }
  }
}


# OUTDATED - use dataLoader.R
# returns balanced accuracy and importance
get_train_test <- function(index, outcome, threshold = 1, max_attempts = 10) {
  gc()

  survivalData = read_rds(paste0("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/temp_survival_data/", 
                                 str_to_title(index), "/", str_to_title(outcome), "/survivalData.rds"))
  
  attempts <- 0  # Initialize a counter for attempts
  
  repeat {
    # Drop censored events with less than the threshold of observation time
    survivalData <- survivalData %>%
      filter(!(timeDiff < threshold & status == 0)) # Filter out censored events with timeDiff less than threshold
    
    survivalDataForClassification <- survivalData %>%
      mutate(event = case_when(timeDiff <= threshold ~ "yes", # if timeDiff is below threshold
                               TRUE ~ "no"))  # Otherwise set event = "no"

    # Drop irrelevant features
    featuresToDrop = c("indexTime", "outcomeTime", "id", "Index Event", 
                       "Outcome Event", "timeDiff", "deployment", 
                       "version", "indexID", "user", "status", 
                       "type")
    
    survivalDataForClassification <- survivalDataForClassification %>%
      select(-any_of(featuresToDrop)) %>%
      mutate(across(matches("AvgAmount"), ~replace(., is.na(.), 0)))
    
    survivalDataForClassification <- survivalDataForClassification %>%
      drop_na()
    
    # Convert relevant columns to factors
    survivalDataForClassification[] <- lapply(survivalDataForClassification, 
                                              function(x) if(is.character(x)) as.factor(x) else x)
    
    # decrease factor count for randomForestModel to be able to handle it
    survivalDataForClassification <- survivalDataForClassification %>%
      mutate(across(where(is.factor), ~ fct_lump(.x, n = 51)))

    # Compute the percentage of events in "yes" and "no"
    pctPerEvent <- survivalDataForClassification %>%
      group_by(event) %>%
      dplyr::summarize(numPerEvent = n(), .groups = 'drop') %>%
      mutate(total = sum(numPerEvent, na.rm = TRUE)) %>%
      mutate(percentage = numPerEvent / total) %>%
      dplyr::select(event, percentage)
    
    # Check if "yes" events are present
    yes_event_percentage <- ifelse(any(pctPerEvent$event == "yes"), 
                                   pctPerEvent$percentage[pctPerEvent$event == "yes"], 
                                   0)
    
    # If the percentage of "yes" events is >= 10, exit loop
    if (yes_event_percentage >= 0.1) {
      break
    }
    
    # If there are no "yes" events and we've reached the max attempts, exit the loop
    attempts <- attempts + 1
    if (attempts >= max_attempts) {
      warning("Max attempts reached; no 'yes' events found.")
      break
    }
    
    # If the percentage of "yes" events is under 10%, multiply the threshold by 10
    threshold <- threshold * 5
  }
  
  # Split data into training and testing sets
  set.seed(123) # For reproducibility
  if (nrow(survivalDataForClassification) == 0) {
    stop("No data available for training and testing after filtering.")
  }
  trainIndex <- createDataPartition(survivalDataForClassification$event, 
                                    p = 0.8, list = FALSE) # Split the data 80/20
  trainData <- survivalDataForClassification[trainIndex, ]
  testData <- survivalDataForClassification[-trainIndex, ]
  
  return(list(train = trainData, test = testData))
}


# OUTDATED - use evaluate_features_pipeline
# Function to compute feature importances for all index-outcome combinations and return their averages
evaluate_features <- function(index, outcome) {
  data = get_train_test(index, outcome)
  trainData = data$train 
  testData = data$test
  
  num_trees <- 1
  rfModel <- randomForest(event ~ ., data = trainData, ntree = num_trees, do.trace = num_trees/10)
  rfPredictions <- predict(rfModel, testData)
  
  # Confusion Matrix for the Random Forest
  rfConfMatrix <- table(Predicted = rfPredictions, Actual = testData$event)
  
  # Extract values from the confusion matrix
  rfTN <- rfConfMatrix[1, 1]  # True Negatives
  rfFP <- rfConfMatrix[1, 2]  # False Positives
  rfFN <- rfConfMatrix[2, 1]  # False Negatives
  rfTP <- rfConfMatrix[2, 2]  # True Positives
  
  # Class Accuracy (Specificity): TN / (TN + FP)
  rfClassAccuracy <- rfTN / (rfTN + rfFP)
  
  # Negative 1 Accuracy (Sensitivity/Recall): TP / (TP + FN)
  rfNegative1Accuracy <- rfTP / (rfTP + rfFN)
  
  # Balanced Accuracy: Average of Sensitivity and Specificity
  rfBalancedAccuracy <- (rfClassAccuracy + rfNegative1Accuracy) / 2 
  
  return(list(balanced_accuracy = rfBalancedAccuracy, importance = as.data.frame(importance(rfModel))))
}


# Function to lump levels of factors together for both train and test based only on the train data (prevents data leakage)
consistently_lump_factors <- function(train_df, test_df, n = 50) {
  # Convert to data.frame to avoid data.table join issues
  train_df <- as.data.frame(train_df)
  test_df <- as.data.frame(test_df)
  
  # Identify factor columns
  factor_cols <- names(train_df)[sapply(train_df, is.factor)]
  
  # Store level mapping for each factor column
  level_mapping <- lapply(train_df[, factor_cols, drop = FALSE], function(column) {
    lumped_column <- fct_lump(column, n = min(n, 51))  # Ensure no more than 53 levels
    levels(lumped_column)
  })
  
  print(level_mapping)
  
  # Consistent lumping for both train and test
  train_lumped <- train_df %>%
    mutate(across(all_of(factor_cols), function(col) {
      col_name <- cur_column()
      keep_levels <- level_mapping[[col_name]]
      fct_collapse(col, Other = setdiff(levels(col), keep_levels))
    }))
  
  test_lumped <- test_df %>%
    mutate(across(all_of(factor_cols), function(col) {
      col_name <- cur_column()
      keep_levels <- level_mapping[[col_name]]
      fct_collapse(col, Other = setdiff(levels(col), keep_levels))
    }))
  
  return(list(train = train_lumped, test = test_lumped))
}


# Copy of this function from dataLoader.R because for some reason the source depends on global variables
loadSurvivalDataset <- function(indexEvent, outcomeEvent, 
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
  
  
  
  return(inner_join(y, X, by = "id"))
  
}

# Function to compute feature importances for all index-outcome combinations and return their averages
evaluate_features_pipeline <- function(indexEvent, outcomeEvent, doNotUseNewFeatures = FALSE) {
  train = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_train/", y_path = "y_train.rds")
  test = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_test/", y_path = "y_test.rds")
  
  classification_cutoff = get_classification_cutoff(indexEvent, outcomeEvent)

  trainData <- data_processing(train, classification_cutoff)
  testData <- data_processing(test, classification_cutoff)

  # decrease factor count for RF model
  # for some reason this doesn't work when using n > 41
  allData <- bind_rows(trainData, testData) %>%
    mutate(across(where(is.factor), ~ fct_lump(.x, n = 41)))
  
  if (doNotUseNewFeatures) {
    allData <- allData %>%
      select(-starts_with("user"), -starts_with("market"), -starts_with("sin"), 
             -starts_with("cos"), -starts_with("day"), -starts_with("isWeekend"), 
             -starts_with("quarter"), -starts_with("timeOfDay"), -starts_with("log"), 
             -starts_with("coinType"))
  }
  
  trainData <- allData[1:nrow(trainData), ]
  testData <- allData[(nrow(testData) + 1):nrow(allData), ]
  
  

  num_trees <- 3
  rfModel <- randomForest(event ~ ., data = trainData, ntree = num_trees, do.trace = num_trees/10)
  rfPredictions <- predict(rfModel, testData)
  
  # Confusion Matrix for the Random Forest
  rfConfMatrix <- table(Predicted = rfPredictions, Actual = testData$event)
  
  # Extract values from the confusion matrix
  rfTN <- rfConfMatrix[1, 1]  # True Negatives
  rfFP <- rfConfMatrix[1, 2]  # False Positives
  rfFN <- rfConfMatrix[2, 1]  # False Negatives
  rfTP <- rfConfMatrix[2, 2]  # True Positives
  
  # Class Accuracy (Specificity): TN / (TN + FP)
  rfClassAccuracy <- rfTN / (rfTN + rfFP)
  
  # Negative 1 Accuracy (Sensitivity/Recall): TP / (TP + FN)
  rfNegative1Accuracy <- rfTP / (rfTP + rfFN)
  
  # Balanced Accuracy: Average of Sensitivity and Specificity
  rfBalancedAccuracy <- (rfClassAccuracy + rfNegative1Accuracy) / 2 
  
  return(list(balanced_accuracy = rfBalancedAccuracy, importance = as.data.frame(importance(rfModel))))
}


# Function to compute feature importances for all index-outcome combinations and return their averages
average_feature_importances <- function(eventTypes) {
  all_importances <- list()
  all_features <- character()
  
  num_events <- length(eventTypes)
  accuracy_matrix <- matrix(NA, nrow = num_events, ncol = num_events,
                            dimnames = list(eventTypes, eventTypes))
  
  totalSteps <- length(eventTypes) * (length(eventTypes) - 1)
  step <- 0
  
  for (indexEvent in eventTypes) {
    for (outcomeEvent in eventTypes) {
      if (indexEvent == outcomeEvent) {
        next
      }
      
      step <- step + 1
      print(paste("Training model", step, "of", totalSteps))
      
      # Use tryCatch to handle errors gracefully
      eval_result <- tryCatch({
        evaluate_features_pipeline(indexEvent, outcomeEvent)
      }, error = function(e) {
        # Print error message and return NULL instead of using next
        print(paste("Error in model for", indexEvent, "vs", outcomeEvent, ":", e$message))
        return(NULL)
      })
      
      # Skip if eval_result is NULL due to an error
      if (is.null(eval_result)) {
        next
      }
      
      feature_importance <- eval_result$importance
      accuracy <- eval_result$balanced_accuracy
      
      accuracy_matrix[indexEvent, outcomeEvent] <- accuracy
      
      if (length(feature_importance) > 0) {
        importance_df <- data.frame(
          Feature = rownames(feature_importance),
          Importance = feature_importance[,1],
          Pair = paste(indexEvent, outcomeEvent, sep = "-"),
          stringsAsFactors = FALSE
        )
        all_importances[[paste(indexEvent, outcomeEvent, sep = "-")]] <- importance_df
        all_features <- union(all_features, importance_df$Feature)
      }
    }
  }
  
  # Combine all feature importances
  combined_importances <- do.call(rbind, all_importances)
  
  # Pivot the data to wide format
  wide_importances <- combined_importances %>%
    pivot_wider(names_from = Pair, values_from = Importance, values_fill = list(Importance = 0))
  
  # Ensure all features are present
  missing_features <- setdiff(all_features, wide_importances$Feature)
  if (length(missing_features) > 0) {
    missing_df <- data.frame(Feature = missing_features)
    wide_importances <- bind_rows(wide_importances, missing_df)
  }
  
  # Calculate mean importance for each feature
  wide_importances$AverageImportance <- rowMeans(wide_importances[,-1], na.rm = TRUE)
  
  # Create the final data frame of average importances
  average_importances_df <- wide_importances %>%
    select(Feature, AverageImportance) %>%
    arrange(desc(AverageImportance))
  
  return(list(AverageImportances = average_importances_df, AccuracyMatrix = accuracy_matrix, AllImportances = all_importances))
}


eventTypes <- c("borrow", "deposit", "withdraw", "repay")
ret <- average_feature_importances(eventTypes)
pander(ret$AccuracyMatrix)
average_importances <- ret$AverageImportances
all_importances <- ret$AllImportances


create_category_heatmap_agglomerative <- function(data, category) {
  # Filter data for the specific category
  category_data <- data %>%
    filter(feature_category == category) %>%
    pivot_wider(names_from = Pair, values_from = Importance, values_fill = list(Importance = 0)) %>%
    pivot_longer(cols = -c(Feature, feature_category), names_to = "IndexOutcomePair", values_to = "Importance")
  
  library(scales)
  library(stats)
  
  # Custom symlog transformation function
  symlog_trans <- function(base = 10) {
    trans <- function(x) sign(x) * log(abs(x) + 1, base)
    inv <- function(x) sign(x) * (exp(abs(x)) - 1)
    
    scales::trans_new(
      name = "symlog",
      transform = trans,
      inverse = inv
    )
  }
  
  # Clamp Importance values to a maximum of 5
  category_data <- category_data %>%
    mutate(Importance = pmin(Importance, 4))
  
  # Prepare data matrix for clustering
  heatmap_matrix <- category_data %>%
    pivot_wider(
      id_cols = Feature, 
      names_from = IndexOutcomePair, 
      values_from = Importance
    )
  
  # Separate feature names
  feature_names <- heatmap_matrix$Feature
  
  # Remove Feature column for clustering
  heatmap_matrix <- as.matrix(heatmap_matrix[,-1])
  rownames(heatmap_matrix) <- feature_names
  
  # Perform hierarchical clustering on rows and columns
  row_dist <- dist(heatmap_matrix)
  col_dist <- dist(t(heatmap_matrix))
  
  row_hclust <- hclust(row_dist, method = "complete")
  col_hclust <- hclust(col_dist, method = "complete")
  
  # Reorder the data based on clustering
  reordered_data <- category_data %>%
    mutate(
      Feature = factor(Feature, levels = feature_names[row_hclust$order]),
      IndexOutcomePair = factor(
        IndexOutcomePair, 
        levels = c(
          # Exclude "Average" from initial sorting
          setdiff(colnames(heatmap_matrix)[col_hclust$order], "Average"),
          "Average"  # Always place "Average" at the end
        )
      )
    )
  
  # Custom breaks and labels for the legend
  custom_breaks <- c(0, 1, 2, 3, 4)
  custom_labels <- c("0", "1", "2", "3", "4+")
  
  heatmap_plot <- ggplot(reordered_data,
                         aes(x = IndexOutcomePair,
                             y = Feature,
                             fill = Importance)) +
    geom_tile() +
    scale_fill_viridis(
      option = "viridis",
      trans = "symlog",
      limits = c(0, 4),
      breaks = custom_breaks,
      labels = custom_labels,
      name = NULL
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_blank())
  
  return(heatmap_plot)
}

normalized_importances <- lapply(all_importances, function(df) {
  # Calculate the mean importance for the current dataframe
  mean_importance <- mean(df$Importance, na.rm = TRUE)

  # Create a copy of the dataframe and normalize the Importance column
  df_normalized <- df
  df_normalized$Importance <- df$Importance / mean_importance
  
  mean_importance <- mean(df_normalized$Importance, na.rm = TRUE)

  return(df_normalized)
})

# Combine all feature importances from all models
all_importances_combined <- do.call(rbind, normalized_importances)

all_importances_combined$feature_category <- case_when(
  grepl("^market", all_importances_combined$Feature) ~ "market",
  grepl("^cos|sin|isWeekend|day|time|weekend|quarter", all_importances_combined$Feature) ~ "time",
  grepl("^user", all_importances_combined$Feature) ~ "user",
  TRUE ~ "info"
)

# Calculate average importance for each feature across all pairs
average_importances <- all_importances_combined %>%
  group_by(Feature, feature_category) %>%
  summarize(Importance = mean(Importance), .groups = 'drop') %>%
  mutate(Pair = "Average")

# Combine with original data
all_importances_with_average <- bind_rows(all_importances_combined, average_importances)

# Create heatmaps for each category
market_heatmap <- create_category_heatmap_agglomerative(all_importances_with_average, "market")
time_heatmap <- create_category_heatmap_agglomerative(all_importances_with_average, "time")
user_heatmap <- create_category_heatmap_agglomerative(all_importances_with_average, "user")
info_heatmap <- create_category_heatmap_agglomerative(all_importances_with_average, "info")

ggsave("info_heatmap.png", info_heatmap, width = 7, height = 2.5)
ggsave("time_heatmap.png", time_heatmap, width = 7, height = 3.75)
ggsave("user_heatmap.png", user_heatmap, width = 7.5, height = 7)
ggsave("market_heatmap.png", market_heatmap, width = 7.5, height = 6.75)

