rsf_regression <- function(trainData, testData) {
  
  # Install required packages if needed
  if (!require("randomForestSRC")) {
    install.packages("randomForestSRC")
    library(randomForestSRC)
  }
  
  # Define the RSF model with the specified features
  rsf_model <- rfsrc(Surv(timeDiff / 86400, status) ~ .,
                     ntree = 5,       # Number of trees
                     nodedepth = 3,    # Tree depth
                     data = trainData,
                     importance = TRUE)  # Calculate variable importance
  
  # Predict on the test data
  rsf_predictions <- predict(rsf_model, newdata = testData)
  
  # Extract the predicted survival times
  predicted_times <- rsf_predictions$predicted
  
  # Return the predicted times and the model
  return(list(predictions = predicted_times, model = rsf_model))
}
