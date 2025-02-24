library(dplyr)
library(data.table)
library(ggplot2)
library(scales)
library(reshape2)
library(caret)
library(pROC)

calculate_model_metrics <- function(confusion_matrix, binary_predictions, model_name) {
  # Load the required package for AUC calculation (uncomment if not already loaded)
  # library(pROC)
  
  # Extract True Negatives, False Positives, False Negatives, and True Positives from the confusion matrix
  TN <- confusion_matrix[1, 1] # True Negatives
  FP <- confusion_matrix[1, 2] # False Positives
  FN <- confusion_matrix[2, 1] # False Negatives
  TP <- confusion_matrix[2, 2] # True Positives
  
  # Calculate Specificity (True Negative Rate)
  specificity <- TN / (TN + FP)
  
  # Calculate Sensitivity (Recall, True Positive Rate)
  sensitivity <- TP / (TP + FN)
  
  # Calculate Balanced Accuracy as the average of Sensitivity and Specificity
  balanced_accuracy <- (specificity + sensitivity) / 2
  
  # Calculate Precision
  precision <- TP / (TP + FP)
  
  # Calculate F1 Score as the harmonic mean of Precision and Sensitivity
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  # Recover the actual labels from the confusion matrix
  # In this confusion matrix, the rows represent predicted values and the columns represent actual values.
  actual_labels <- c(rep(0, TN + FP), rep(1, FN + TP)) # 0 represents "no", 1 represents "yes"
  
  # Ensure that binary_predictions are treated as probability values
  predicted_probabilities <- as.numeric(binary_predictions) # Convert predictions to numeric
  
  # Calculate the AUC (Area Under the Curve) Score
  auc_score <- NaN # Default value in case the AUC cannot be calculated
  if (length(actual_labels) == length(predicted_probabilities) && length(unique(actual_labels)) > 1) {
    roc_curve <- roc(actual_labels, predicted_probabilities)
    auc_score <- auc(roc_curve)
  }
  
  if (is.nan(balanced_accuracy)) balanced_accuracy <- 0.50
  if (is.nan(f1_score)) f1_score <- 0.50
  if (is.nan(auc_score)) auc_score <- 0.50
  
  # Print all performance metrics with labels
  print(paste(model_name, "model prediction accuracy:"))
  cat("Balanced Accuracy:", sprintf("%.2f%%", balanced_accuracy * 100), "\n")
  cat("F1 Score:", sprintf("%.2f%%", f1_score * 100), "\n")
  cat("AUC Score:", sprintf("%.2f%%", auc_score * 100), "\n")
  
  # Return all computed metrics in a list
  return(list(
    balanced_accuracy = balanced_accuracy, 
    f1_score = f1_score,
    auc_score = auc_score
  ))
}

get_dataframe <- function(model_name, metrics) {
  metrics_dataframe <- data.frame(
    Model = model_name, 
    # Balanced_Accuracy = sprintf("%.2f%%", metrics$balanced_accuracy * 100),
    AUC_Score = sprintf("%.2f%%", metrics$auc_score * 100),
    F1_Score = sprintf("%.2f%%", metrics$f1_score * 100)
  )
  return (metrics_dataframe)
}

combine_classification_results <- function(accuracy_dataframe_list, data_combination) {
  # apply the data combination description to each dataframe in the list
  accuracy_dataframe_list <- lapply(accuracy_dataframe_list, function(df) {
    # add a new column `Data_Combination` to store the combination description
    # this allows each dataframe to retain information about the specific data combination it
    # represents
    df$Data_Combination <- data_combination
    # return the modified dataframe with the new column added
    return(df)
  })
  
  # combine all the modified dataframes into one large dataframe
  # `do.call` applies `rbind` to all dataframes in the list, effectively stacking them by rows
  combined_dataframe <- do.call(rbind, accuracy_dataframe_list)
  
  # return the combined dataframe
  return(combined_dataframe)
}

get_percentage <- function(survivalDataForClassification, indexEvent, outcomeEvent) {
  # indexEvent and outcomeEvent is a string type
  pctPerEvent <- survivalDataForClassification %>%
    group_by(event) %>%
    dplyr::summarize(numPerEvent = n()) %>%
    mutate(total = sum(numPerEvent)) %>%
    mutate(percentage = numPerEvent / total) %>%
    dplyr::select(event, percentage)
  # create a bar plot for event percentages
  # stat = "identity": percentages used directly to draw the bar chart
  print(ggplot(pctPerEvent, aes(x = event, y = percentage, fill = event)) +
          geom_bar(stat = "identity") +
          scale_y_continuous(labels = scales::percent_format()) +  # show y-axis in percentage
          labs(title = "Percentage of Events: 'Yes' event vs 'No' event",
               x = paste(indexEvent, "and", outcomeEvent),
               y = "Percentage") +
          geom_text(aes(label = scales::percent(percentage)), 
                    vjust = -0.5, size = 3.5) +  # show percentages on top of bars
          theme_minimal())
}

accuracy_comparison_plot <- function(metrics_list) {
  # initialize an empty data frame to store the metrics for all models
  accuracy_table <- data.frame()
  
  # loop over each element in metrics_list (each element is a list containing metrics and model name)
  for (metrics in metrics_list) {
    # Extract metrics and model name from each "tuple"
    model_metrics <- metrics[[1]]
    model_name <- metrics[[2]]
    
    # create a temporary dataframe for this model
    temp_df <- data.frame(
      Model = model_name, 
      # BalancedA = model_metrics$balanced_accuracy,
      AUC_score = model_metrics$AUC_score,
      F1_score = model_metrics$f1_score
    )
    
    # append the temporary dataframe to the main accuracy_table
    accuracy_table <- rbind(accuracy_table, temp_df)
  }
  
  # melt the dataframe into long format for plotting
  accuracy_results_melted <- reshape2::melt(accuracy_table, id.vars = "Model")
  
  # generate the plot with faceted bars
  ggplot(accuracy_results_melted, aes(x = Model, y = value, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ variable, scales = "free_y") +  # Facet by each metric
    labs(title = "Comparison of Accuracy Metrics Across Models",
         x = "Model",
         y = "Value") +
    # add percentage labels on top of each bar
    geom_text(aes(label = scales::percent(value, accuracy = 0.1)),
              position = position_dodge(width = 0.9),
              vjust = 0.5, size = 2.0) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

specific_accuracy_statistics <- function(event_pair, accuracy_type, metrics_list) {
  # initialize the result list
  results <- list()
  
  # add accuracy_type as the left front corner and event_pair is the first column of the data
  results[[accuracy_type]] <- event_pair
  
  # traverse metrics_list and extract the specified accuracy type for each model
  for (metric_item in metrics_list) {
    # get accuracy data
    accuracy_data <- metric_item[[1]]
    # get the model name
    model_name <- metric_item[[2]]
    if (accuracy_type == "balanced_accuracy") {
      results[[model_name]] <- round(accuracy_data$balanced_accuracy * 100, 2)
    }
    else if (accuracy_type == "auc_score") {
      results[[model_name]] <- round(accuracy_data$auc_score * 100, 2)
    }
    else if (accuracy_type == "f1_score") {
      results[[model_name]] <- round(accuracy_data$f1_score * 100, 2)
    }
    else {
      # An error message is displayed if the specified accuracy_type does not exist.
      stop(paste("Invalid accuracy type:", accuracy_type))
    }
  }
  
  # convert the result to a DataFrame and set row.names = NULL
  df <- as.data.frame(results, row.names = NULL)
  return(df)
}

combine_accuracy_dataframes <- function(df_list) {
  # check if the input is a list
  if (!is.list(df_list)) {
    stop("Input must be a list of data.frames.")
  }
  
  # Use do.call and rbind to combine all data.frames in a list.
  combined_df <- do.call(rbind, df_list)
  
  # returns the merged data.frame
  return(combined_df)
}