# as.factor()

# Make sure all categorical variables are of type `factor`
data <- read.csv("data.csv", sep = ";")
categorical_variables <- c(
  "marital", "education", "default", "housing",
  "loan", "contact", "poutcome", "class"
)
data[, categorical_variables] <- lapply(
  data[, categorical_variables],
  as.factor
)
head(data)

# Exploratory data analysis
# Overall summary
summary(data)

# Search for any missing values
missing_count <- colSums(is.na(data))
print(missing_count) # => No missing data in our database

# Distributions of observations
hist_obj <- hist(data$balance, plot = FALSE)
hist(data$balance,
  main = "Histogram with Density Estimate", xlab = "Balance", ylab = "Frequency"
)
lines(density(data$balance), col = "blue", lwd = 2)

# Take EDA from Mathieu... + ANOVA


library(rpart)

# Split data into train and test
set.seed(1234)
index <- sample(1:nrow(data), round(0.70 * nrow(data)))
train <- data[index, ]
test <- data[-index, ]
nrow(train) # 5505
nrow(test) # 2359

# Define the hyperparameter grid
param_grid <- expand.grid(
  minsplit = seq(2, 14, by = 2),
  minbucket = seq(1, 10, by = 1),
  cp = seq(0, 0.003, by = 0.001)
)

# Check trainControl for automatic CV

# Initialize an empty vector to store cross-validation results
cv_accuracies <- numeric(nrow(param_grid))
cv_mses <- numeric(nrow(param_grid))

# Perform k-fold cross-validation
k <- 5

for (i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]

  # Set up cross-validation
  set.seed(123) # For reproducibility
  folds <- sample(1:k, nrow(data), replace = TRUE)

  # Initialize vector to store accuracy for each fold
  fold_accuracies <- numeric(k)
  fold_predictions <- list()

  for (j in 1:k) {
    # Split the data into training and validation sets
    validation_data <- data[folds == j, ]
    training_data <- data[folds != j, ]

    # Fit the decision tree model
    tree_model <- rpart(class ~ .,
      data = training_data,
      method = "class",
      minsplit = params$minsplit,
      minbucket = params$minbucket,
      cp = params$cp
    )
    # You have access to: tree_model$cptable

    # Make predictions on the validation set
    predictions <- predict(tree_model,
      newdata = validation_data,
      type = "class"
    )

    # Calculate accuracy for this fold
    fold_accuracy <- mean(predictions == validation_data$class)
    fold_accuracies[j] <- fold_accuracy
    fold_predictions[[j]] <- predictions
    # Calculate other metrics
    # xerror, xstd...
  }

  # Calculate the mean accuracy across folds
  cv_accuracies[i] <- mean(fold_accuracies)
  cv_mses[i] <- mean((fold_accuracies - 1)^2)
}

# Calculate standard deviation of accuracy
cv_sds <- apply(sapply(fold_predictions, as.numeric), 1, sd)

# Create a data frame to store results
results_table <- cbind(
  param_grid,
  Mean_CV_Accuracy = cv_accuracies,
  Mean_CV_MSE = cv_mses
  # SD_Accuracy = cv_sds
)

# Print or inspect the results table
print(results_table)
