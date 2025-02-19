library(dplyr)
library(data.table)
library(ggplot2)
library(scales)
library(reshape2)
library(caret)

calculate_model_metrics <- function(confusion_matrix, binary_predictions, model_name) {
  TN <- confusion_matrix[1, 1] # True Negatives
  FP <- confusion_matrix[1, 2] # False Positives
  FN <- confusion_matrix[2, 1] # False Negatives
  TP <- confusion_matrix[2, 2] # True Positives
  
  # positive Class accuracy (Specificity): TN / (TN + FP)
  class_accuracy <- TN / (TN + FP)
  
  # negative 1 accuracy (Sensitivity/Recall): TP / (TP + FN)
  negative_1_accuracy <- TP / (TP + FN)
  
  # balanced accuracy: Average of Sensitivity and Specificity
  balanced_accuracy <- (class_accuracy + negative_1_accuracy) / 2
  
  # precision
  precision <- TP / (TP + FP)
  
  # f1 score
  f1_score <- 2 * (precision * negative_1_accuracy) / (precision + negative_1_accuracy)
  
  # print out all the accuracy records
  print(paste(model_name, "model prediction accuracy:"))
  cat("Balanced accuracy:", sprintf("%.2f%%", balanced_accuracy * 100), "\n")
  cat("F1 score:", sprintf("%.2f%%", f1_score * 100), "\n")
  
  return (list(balanced_accuracy = balanced_accuracy, 
               f1_score = f1_score))
}

get_dataframe <- function(model_name, metrics) {
  metrics_dataframe <- data.frame(
    Model = model_name, 
    Balanced_Accuracy = sprintf("%.2f%%", metrics$balanced_accuracy * 100), 
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
      BalancedA = model_metrics$balanced_accuracy, 
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
      results[[model_name]] <- round(accuracy_data$balanced_accuracy * 100, 1)
    }
    else if (accuracy_type == "f1_score") {
      results[[model_name]] <- round(accuracy_data$f1_score * 100, 1)
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