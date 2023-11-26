# Omar ALLOUCH - Mathieu SROUR
rm(list = ls())

# Partie 1 - Analyse Descriptive
library(quantmod)
start_date <- as.Date("2020-01-01")
end_date <- as.Date("2022-12-31")
# Importing dataset
getSymbols("GOLD", src="yahoo", from = start_date, to = end_date)

# Extracting necessary data
gold = GOLD$GOLD.Close

plot(gold)

mean(gold) # 21.30594 ($)

sd(gold) # 3.967074 ($)

summary(gold)
#       Index              GOLD.Close   
# Min.   :2020-01-02   Min.   :13.10  
# 1st Qu.:2020-09-30   1st Qu.:18.44  
# Median :2021-07-01   Median :20.72  
# Mean   :2021-07-01   Mean   :21.31  
# 3rd Qu.:2022-03-31   3rd Qu.:24.17  
# Max.   :2022-12-30   Max.   :30.46

# Le prix minimum est $13.10
# Le prix maximum est $30.46
# La moitie des donnees est au-dessous/au-dessus de $20.72
# 25% des donnees est au-dessous de $18.44
# 75% des donnees est au-dessous de $24.17

boxplot(gold) # On constate que la serie n'a aucune valeur aberrante.

# La serie ne presente pas de saisonnalite. Malgre que le comportement generale de la serie pendant un annee est d'augmenter puis de diminuer,
# mais cette augmentation/diminution:
# 1) n'est pas la meme
# 2) n'est pas pendant les memes periodes de l'annee

# Il n'y a pas de tendance generale sur toute la duree, mais si on prend des intervalles precis on peut distinguer quelques tendances:
# Jan 2020 -> Sept 2020: Tendance croissante
# Oct 2020 -> Feb 2022: Tendance decroissante
# Mar 2022 -> May 2022: Tendance croissante
# Jun 2022 -> Dec 2022: Tendance decroissante

# La serie presente des fluctuations assez constantes sauf sur des periodes precises:
# April - May 2020: augmentation considerable
# May - August 2022: diminution importante

################################################################################

# Partie 2 - Modelisation
end_date_train <- as.Date("2022-10-01")

# Splitting data
training_data = subset(GOLD, index(GOLD) >= start_date & index(GOLD) <= end_date_train)
test_data = subset(GOLD, index(GOLD) > end_date_train)

plot(training_data$GOLD.Close)
plot(test_data$GOLD.Close)

acf_result <- acf(training_data$GOLD.Close) # MA(0)
pacf_result <- pacf(training_data$GOLD.Close) # AR(1)
# Mais la serie n'est pas stationnaire, alors il est essentiel de la rendre stationnaire pas differentiation.

# Differentiation order 1
diff_data = diff(training_data$GOLD.Close, differences = 1)
diff_data = na.omit(diff_data)
plot(diff_data) # stationnaire  (moyenne constante)
acf_result <- acf(diff_data) # MA(0)
pacf_result <- pacf(diff_data) # AR(0)
# Les ACF / PACF suggerent un modele ARIMA(0, 0, 0)

library(tseries)
library(forecast)

# Finding optimal model
model = auto.arima(diff_data)
model # ARIMA(0, 0, 0) => En accord avec acf/pacf
# Conclusion initiale: bruit blanc, pas de correlations considerables


# Initial data
ts_data <- ts(na.omit(training_data$GOLD.Close), frequency = length(training_data$GOLD.Close)/2.75)
decomposition <- decompose(ts_data)
plot(decomposition)

# Residual series
r_t=decomposition$random

# Removing NA values
r_t=na.omit(r_t)
plot(r_t)

# Volatility test
adf=adf.test(r_t)
adf # non-stationnary (p-value = 0.943)

# Independence test
ljung_box_test <- Box.test(r_t, lag = 5, type = "Ljung-Box")
ljung_box_test # dependence (p-value = zero machine)

# Finding optimal model for residual
model <- auto.arima(r_t)
model # ARIMA(0, 1, 0) => Which is to be expected.
plot(r_t)
lines(fitted(model), col = 'red')


# Differentiated data (order 1)
diff_ts <- diff(ts_data, differences = 1)
decomp <- decompose(diff_ts)
plot(decomp)

# Residual series
diff_rt=decomp$random

diff_rt=na.omit(diff_rt)
plot(diff_rt)

# Volatility test
adf <- adf.test(diff_rt)
adf # Stationary

# Independance test
ljung_box_test <- Box.test(diff_rt, lag = 5, type = "Ljung-Box")
ljung_box_test # Independance

# Finding optimal model for residual
model <- auto.arima(diff_rt)
model # ARIMA(0, 0, 0) => Bruit blanc (a nouveau)
plot(diff_rt)
lines(fitted(model), col="red")
# Conclusion initiale est confirmee, apres differentiation d'ordre 1 pour la rendre stationnaire,
# la serie ne presente aucune correlation => bruit blanc.


# Testing ARIMA(0, 1, 0) and estimating "goodness of fit"
model <- arima(ts_data, order = c(0, 1, 0))

forecast_values = forecast(model, h=length(test_data$GOLD.Close))
forecasted_data <- forecast_values$mean
comparison_df <- data.frame(Actual = test_data$GOLD.Close, Forecasted = forecasted_data)

plot(index(test_data$GOLD.Close), test_data$GOLD.Close, type = 'l')
lines(index(test_data$GOLD.Close), forecast_values$mean, col='red')

plot(index(training_data$GOLD.Close), training_data$GOLD.Close, type = 'l')
lines(index(test_data$GOLD.Close), test_data$GOLD.Close, type = 'l', col='cyan')
lines(index(test_data$GOLD.Close), comparison_df$Forecasted, col='red')

# MSE
squared_differences <- (comparison_df$GOLD.Close - comparison_df$Forecasted)^2
mse <- mean(squared_differences)
mse # 35.67165 => Bad fit
# Likelihood
log_likelihood <- logLik(model, data=test_data$GOLD.Close)
log_likelihood # -1901.824 => Bad fit
