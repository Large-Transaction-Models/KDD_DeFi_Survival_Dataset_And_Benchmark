concordanceIndex <- function(predictions, test, model_type = c('cox', 'aft', 'gbm', 'xgb', 'rsf')){
  #Install required packages if needed
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  
  model_type <- match.arg(model_type) 
  
  #Make sure testing data doesn't have timeDiff = 0
  test <- test[test$timeDiff > 0,]
  
  # Compute concordance index using concordance() function
  cindex <- concordance(Surv((timeDiff/86400), status) ~ predictions, data = test)$concordance
  
  # Depending of the model, we return cindex or 1-cindex
  if (model_type %in% c("aft")) {
    return(cindex)
  } else {
    return(1-cindex)
  }
}