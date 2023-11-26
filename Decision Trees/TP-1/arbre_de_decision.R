library(dplyr)
library(rpart)
library(caret)
library(ggplot2)

data = read.csv("7 bank_marketing.csv", sep = ";")
data


# Check for missing values in a dataset
missing_values_count <- colSums(is.na(data))
#no missing values

#boxplot des variables quantitatives
boxplot(data$age, main = "Simple Boxplot", ylab = "Values")
boxplot(data$balance, main = "Simple Boxplot", ylab = "Values")

#
data$loan <- as.factor(data$loan)
data$class <- as.factor(data$class)
data$poutcome <- as.factor(data$poutcome)
data$contact <- as.factor(data$contact)
data$housing <- as.factor(data$housing)
data$default <- as.factor(data$default)
data$education <- as.factor(data$education)
data$marital <- as.factor(data$marital)

loan_frequency_table <- table(factor_loan)
barplot(loan_frequency_table, main = "Categorical Data Distribution", xlab = "Categories", ylab = "Count")
percentages <- round(100 * loan_frequency_table / sum(loan_frequency_table), 1)
pie(loan_frequency_table, labels = paste(names(loan_frequency_table), percentages, "%"),main = "Categorical Data Distribution")

class_frequency_table <- table(factor_class)
barplot(class_frequency_table, main = "Categorical Data Distribution", xlab = "Categories", ylab = "Count")
percentages <- round(100 * class_frequency_table / sum(class_frequency_table), 1)
pie(class_frequency_table, labels = paste(names(class_frequency_table), percentages, "%"),main = "Categorical Data Distribution")

poutcome_frequency_table <- table(factor_loan)
barplot(poutcome_frequency_table, main = "Categorical Data Distribution", xlab = "Categories", ylab = "Count")
percentages <- round(100 * poutcome_frequency_table / sum(poutcome_frequency_table), 1)
pie(poutcome_frequency_table, labels = paste(names(poutcome_frequency_table), percentages, "%"),main = "Categorical Data Distribution")

classes <- sapply(data, class)
classes

# decoupage partie train et partie test
set.seed(1234)
index <- sample(1:nrow(data),round(0.70*nrow(data)))
train <- data[index,]
test <- data[-index,]
nrow(train)
nrow(test)

# Create a sequence from start to end with a specified step size
cp_values <- seq(from = 0.0005, to = 0.005, by = 0.0001)

### Construction de l'arbre
fulltree<-rpart(class~., data=train, method="class", cp)
rpart.plot::rpart.plot(fulltree)
summary(fulltree)
printcp(fulltree)
varImp(fulltree)



# Initialize an empty vector to store cross-validation results
cv_results <- numeric(length(cp_values))

# Perform k-fold cross-validation (e.g., 5-fold)
k <- 5

for (i in 1:length(cp_values)) {
  cp <- cp_values[i]
  
  # Set up cross-validation
  set.seed(123)  # For reproducibility
  folds <- sample(1:k, nrow(data), replace = TRUE)
  
  # Initialize vector to store accuracy for each fold
  fold_accuracies <- numeric(k)
  
  for (j in 1:k) {
    # Split the data into training and validation sets
    validation_data <- data[folds == j, ]
    training_data <- data[folds != j, ]
    
    # Fit the decision tree model
    tree_model <- rpart(class~., data = training_data, method = "class", cp = cp)
    
    # Make predictions on the validation set
    predictions <- predict(tree_model, newdata = validation_data, type = "class")
    
    # Calculate accuracy for this fold
    fold_accuracy <- mean(predictions == validation_data$class)
    fold_accuracies[j] <- fold_accuracy
  }
  
  # Calculate the mean accuracy across folds
  cv_results[i] <- mean(fold_accuracies)
}

# Create a data frame to store results
results_table <- data.frame(CP = cp_values, Mean_CV_Accuracy = cv_results)

# Print or inspect the results table
print(results_table)






#ggplot(data, aes(x = data$poutcome, fill = data$class)) +geom_bar()
