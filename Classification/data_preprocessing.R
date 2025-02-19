library(readr)
library(stringr)
library(dplyr)
library(tidyverse)
library(tidyr)
library(ROSE)

get_train_test_data <- function(indexEvent, outcomeEvent) {
  source("~/KDD_DeFi_Survival_Dataset_And_Benchmark/dataLoader.R")
}

data_processing <- function(survivalData, set_timeDiff) {
  # filter out invalid records where `timeDiff` is <= 0 early
  survivalData <- survivalData %>% filter(timeDiff > 0)
  
  # filter out records based on the `set_timeDiff` threshold and `status`
  survivalData <- survivalData %>% filter(!(timeDiff / 86400 <= set_timeDiff & status == 0))
  
  # create a new binary column `event` based on `timeDiff`
  survivalDataForClassification <- survivalData %>%
    mutate(event = case_when(
      timeDiff / 86400 <= set_timeDiff ~ "yes",
      timeDiff / 86400 > set_timeDiff ~ "no"
    ))
  
  featuresToDrop <- c("indexTime", "outcomeTime", "id", "Index Event", "Outcome Event",
                      "timeDiff", "status", "deployment", "version", "indexID", "user", 
                      "liquidator", "pool", "timestamp", "type", "datetime", "quarter_start_date")
  
  # remove only columns that actually exist in the dataset
  featuresToDrop <- intersect(featuresToDrop, colnames(survivalDataForClassification))
  
  survivalDataForClassification <- survivalDataForClassification %>%
    # drop unnecessary columns
    select(-any_of(featuresToDrop)) %>%
    # remove columns with only NA values
    select(where(~ !all(is.na(.)))) %>%
    # replace NA in numeric columns with -999
    mutate(across(where(is.numeric), ~ replace_na(., -999))) %>%
    # replace NA in character columns with "missing"
    mutate(across(where(is.character), ~ replace_na(., "missing"))) %>%
    # convert character columns to factors
    mutate(across(where(is.character), as.factor))
  
  # return the processed dataset
  return(survivalDataForClassification)
}

smote_data <- function(train_data, target_var = "event", seed = 123) {
  # library(ROSE)
  # check if the input data contains the target variable
  if (!target_var %in% colnames(train_data)) {
    stop(paste("Target variable", target_var, "not found in the dataset"))
  }
  
  # set the random seed (if provided)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # dynamic formula creation to adapt to different target variables
  formula <- as.formula(paste(target_var, "~ ."))
  
  # applying ROSE Balance Data
  train_data_balanced <- ROSE(formula, data = train_data, seed = seed)$data
  
  # return the balanced dataset
  return(train_data_balanced)
}