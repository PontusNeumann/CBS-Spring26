# ----------------------------------------
# 1. Install & load required packages safely
# ----------------------------------------

if (!require("tseries", quietly = TRUE)) {
  install.packages("tseries")
  library(tseries)
} else {
  library(tseries)
}

if (!require("forecast", quietly = TRUE)) {
  install.packages("forecast")
  library(forecast)
} else {
  library(forecast)
}

# ----------------------------------------
# 2. Import data from CSV file
# ----------------------------------------

# Setting working directory
setwd("C:/Users/tce.eco/OneDrive - CBS - Copenhagen Business School/01 DOCUMENTS/02 TEACHING/02 PREDICTIVE ANALYTICS - F26/Lectures/Lecture 6")

data <- read.csv("Simulated ARIMA 2-1-1 data.csv", header = TRUE)
head(data)

# ----------------------------------------
# 3. Rename columns
# ----------------------------------------

colnames(data) <- c("Time", "X")

# ----------------------------------------
# 4. Convert to time series object
# ----------------------------------------

ts_data <- ts(data$X, start = 1, frequency = 1)

# ----------------------------------------
# 5. Plot the original (non-stationary) time series - ARIMA(2,1,1) 
# with AR1 = 0.7, AR2 = -0.4, MA1 = 0.3, std. = 1
# ----------------------------------------

plot(ts_data,
     main = "Original Time Series",
     ylab = "X",
     xlab = "Time")

# ----------------------------------------
# 6. ACF and PACF of original series
# ----------------------------------------

acf(ts_data, main = "ACF of Original (Non-Stationary) Series")
pacf(ts_data, main = "PACF of Original (Non-Stationary) Series")

# ----------------------------------------
# 7. Stationarity tests on original series
# ----------------------------------------

adf.test(ts_data)  # H0: non-stationary (unit root)
kpss.test(ts_data, null = "Level")  # H0: stationary

# ----------------------------------------
# 8. First differencing
# ----------------------------------------

diff_ts <- diff(ts_data, differences = 1)

# ----------------------------------------
# 9. Plot the differenced series
# ----------------------------------------

plot(diff_ts,
     main = "First-Differenced Series",
     ylab = "Change in X",
     xlab = "Time")

# ----------------------------------------
# 10. ACF and PACF of differenced series
# ----------------------------------------

acf(diff_ts, main = "ACF of Differenced Series")
pacf(diff_ts, main = "PACF of Differenced Series")

# ----------------------------------------
# 11. Stationarity tests on differenced series
# ----------------------------------------

adf.test(diff_ts)  # H0: non-stationary
kpss.test(diff_ts, null = "Level")  # H0: stationary

# ----------------------------------------
# 12. Estimate ARMA(2,2) model on differenced series
# ----------------------------------------

arma22_model <- Arima(diff_ts, order = c(2, 0, 2))
summary(arma22_model)

# ----------------------------------------
# 13. Diagnostics for ARMA(2,2)
# ----------------------------------------

tsdisplay(residuals(arma22_model),
          main = "Residual Diagnostics for ARMA(2,2)",
          lag.max = 30)

Box.test(residuals(arma22_model), lag = 20, type = "Ljung-Box")

# ----------------------------------------
# 14. Estimate ARMA(2,1) model on differenced series
# ----------------------------------------

arma21_model <- Arima(diff_ts, order = c(2, 0, 1))
summary(arma21_model)

# ----------------------------------------
# 15. Diagnostics for ARMA(2,1)
# ----------------------------------------

tsdisplay(residuals(arma21_model),
          main = "Residual Diagnostics for ARMA(2,1)",
          lag.max = 30)

Box.test(residuals(arma21_model), lag = 20, type = "Ljung-Box")

# ----------------------------------------
# 16. Compare model fit: AIC and BIC
# ----------------------------------------

model_comparison <- data.frame(
  Model = c("ARMA(2,2)", "ARMA(2,1)"),
  AIC   = c(AIC(arma22_model), AIC(arma21_model)),
  BIC   = c(BIC(arma22_model), BIC(arma21_model))
)

print(model_comparison)
