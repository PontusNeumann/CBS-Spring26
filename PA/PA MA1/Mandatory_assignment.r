# Installing packages
# install.packages("AER")
# install.packages("fpp3")

# Load packages
library(AER)
library(fpp3)

# Load data
data("USMacroG", package = "AER")

# Extract GDP
gdp <- USMacroG$gdp

# Plot GDP
plot(gdp)

# ACF
acf(gdp)

# Growth rate
gdp_growth <- diff(log(gdp))

# Plot growth
plot(gdp_growth)

# ACF growth
acf(gdp_growth)

# Ljung-Box test
Box.test(gdp_growth, lag = 12, type = "Ljung-Box")