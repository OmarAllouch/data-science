library(ggplot2)

# 1.
data_app <- read.table("Data_app.txt", header = TRUE)

model <- lm(ventes ~ temperature, data = data_app)

summary(model)

data_test <- read.table("Data_test.txt", header = TRUE)
sales_pred <- predict(model, newdata = data_test)

ventes_test <- read.table("Ventes_test.txt", header = TRUE)

rmse <- sqrt(mean((ventes_test$ventes - sales_pred)^2))
cat("RMSE:", rmse, "\n")

ggplot(data_app, aes(x = temperature, y = ventes)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(x = "Temperature", y = "Ventes")

residuals <- ventes_test$ventes - sales_pred
ggplot(data.frame(Predicted = sales_pred, Residuals = residuals), 
       aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Predicted Sales", y = "Residuals")

# 2.
# On utilise une regression lineaire multiple en utilisant 2 variables:
# - temperature
# - jour

# 3.
data_app$we <- ifelse(data_app$jour %in% c("S", "D"), 1, 0)
data_app$dim <- ifelse(data_app$jour == "D", 1, 0)

model_multiple <- lm(ventes ~ temperature + we + dim, data = data_app)

summary(model_multiple)

data_test <- read.table("Data_test.txt", header = TRUE)
data_test$we <- ifelse(data_test$jour %in% c("S", "D"), 1, 0)
data_test$dim <- ifelse(data_test$jour == "D", 1, 0)
sales_pred_multiple <- predict(model_multiple, newdata = data_test)

rmse_multiple <- sqrt(mean((ventes_test$ventes - sales_pred_multiple)^2))
cat("RMSE for Multiple Regression:", rmse_multiple, "\n")

cat("RMSE for Simple Linear Regression (previously calculated):", rmse, "\n")


# 4.
model_interaction <- lm(ventes ~ temperature * we + dim, data = data_app)

summary(model_interaction)

sales_pred_interaction <- predict(model_interaction, newdata = data_test)

rmse_interaction <- sqrt(mean((ventes_test$ventes - sales_pred_interaction)^2))
cat("RMSE for Multiple Regression with Interaction:", rmse_interaction, "\n")
