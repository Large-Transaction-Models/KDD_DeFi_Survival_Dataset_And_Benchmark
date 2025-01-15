preprocessing <- function(train, test){
  # Convert NA to 0 in selected columns from the training set
  columns_to_fill <- c(
    "userDepositAvgAmountETH", "userDepositAvgAmountUSD", "userDepositAvgAmount",
    "userLiquidationAvgAmountETH", "userLiquidationAvgAmount", 
    "userLiquidationAvgAmountUSD",
    "userFlashloanAvgAmount", "userRepayAvgAmountETH", "userRepayAvgAmountUSD",
    "userRepayAvgAmount", "userWithdrawAvgAmountETH", "userWithdrawAvgAmountUSD",
    "userWithdrawAvgAmount"
  )
  
  train <- train %>%
    mutate(across(
      all_of(intersect(columns_to_fill, names(train))),  # Only columns that exist
      ~ nafill(., fill = 0)  # Fill NAs with 0
    ))
  
  # Convert NA to 0 in selected columns from the testing set
  test <- test %>%
    mutate(across(
      all_of(intersect(columns_to_fill, names(train))),  # Only columns that exist
      ~ nafill(., fill = 0)  # Fill NAs with 0
    ))
  
  # Convert character features to factors in training set for optimal processing
  train[] <- lapply(train, 
                    function(x) if(is.character(x)) as.factor(x) else x)
  
  # Convert character features to factors in testing set for optimal processing
  test[] <- lapply(test, 
                   function(x) if(is.character(x)) as.factor(x) else x)
  
  # Process the data
  train <- train %>%
    group_by(reserve) %>%
    filter(!any(sapply(across(everything(), ~ length(unique(status)) == 1), identity))) %>%
    ungroup()
  
  # Names of factor columns that won't be compared or modified
  excluded_columns <- c("status","id","user")
  
  # Apply level matching only to non-excluded columns that are factors
  test <- test %>%
    mutate(across(
      .cols = where(is.factor) & !all_of(excluded_columns),
      .fns = ~ factor(., levels = levels(train[[cur_column()]]))
    ))
  
  test <- test[test$timeDiff > 0, ]
  train <- train[train$timeDiff >0, ]
  
  infinite_cols <- sapply(train, function(x) any(is.infinite(x)))
  
  # Show columns with infinite values
  exc_cols<-names(train)[infinite_cols]
  
  # Remove invalid features
  train <- train %>% select(-any_of(c("user", "outcomeTime", 
                                      "indexTime", "Index Event", "Outcome Event",
                                      "pool","timestamp","type","id","indexID",
                                      "datetime",
                                      "quarter_start_date", "userReserveMode",
                                      "userCoinTypeMode", "coinType", "userIsNew", 
                                      "userDepositSum", "userDepositSumUSD",
                                      "userDepositAvgAmountUSD", "userDepositSumETH",
                                      "userDepositAvgAmountETH", "userWithdrawSum",
                                      "userWithdrawSumUSD", "userWithdrawAvgAmountUSD",
                                      "userWithdrawSumETH", "userWithdrawAvgAmountETH",
                                      "userBorrowSum","userBorrowSumUSD", 
                                      "userBorrowAvgAmountUSD", "userBorrowSumETH",
                                      "userBorrowAvgAmountETH", "userRepaySum",
                                      "userRepaySumUSD", "userRepayAvgAmountUSD",
                                      "userRepaySumETH", "userRepayAvgAmountETH",
                                      "userLiquidationSum", 
                                      "userLiquidationSumUSD",
                                      "userLiquidationAvgAmountUSD",
                                      "userLiquidationSumETH",
                                      "userLiquidationAvgAmountETH",
                                      "userBorrowAvgAmount","priceInUSD",
                                      "cosQuarter",
                                      exc_cols)))
  
  test <- test %>% select(-any_of(c("user", "outcomeTime", 
                                    "indexTime", "Index Event", "Outcome Event",
                                    "pool","timestamp","type","id","indexID",
                                    "datetime",
                                    "quarter_start_date", "userReserveMode",
                                    "userCoinTypeMode", "coinType", "userIsNew", 
                                    "userDepositSum", "userDepositSumUSD",
                                    "userDepositAvgAmountUSD", "userDepositSumETH",
                                    "userDepositAvgAmountETH", "userWithdrawSum",
                                    "userWithdrawSumUSD", "userWithdrawAvgAmountUSD",
                                    "userWithdrawSumETH", "userWithdrawAvgAmountETH",
                                    "userBorrowSum","userBorrowSumUSD", 
                                    "userBorrowAvgAmountUSD", "userBorrowSumETH",
                                    "userBorrowAvgAmountETH", "userRepaySum",
                                    "userRepaySumUSD", "userRepayAvgAmountUSD",
                                    "userRepaySumETH", "userRepayAvgAmountETH",
                                    "userLiquidationSum", 
                                    "userLiquidationSumUSD",
                                    "userLiquidationAvgAmountUSD",
                                    "userLiquidationSumETH",
                                    "userLiquidationAvgAmountETH",
                                    "userBorrowAvgAmount","priceInUSD",
                                    "cosQuarter",
                                    exc_cols)))
  
  l <- list(train, test)
  
  return(l)
}
