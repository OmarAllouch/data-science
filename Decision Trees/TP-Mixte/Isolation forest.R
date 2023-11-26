data <- read.csv("Income_Inequality.csv", sep = ";")

library("solitude")
library("tidyverse")
library("mlbench")

data <- data[, c(4:20)]
data <- na.omit(data)
colnames(data)

iso <- isolationForest$new(sample_size = 245)
iso$fit(data)

scores <- data %>%
  iso$predict() %>%
  arrange(desc(anomaly_score))

scores
