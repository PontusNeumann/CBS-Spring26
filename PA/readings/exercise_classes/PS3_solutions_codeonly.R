setwd("/Users/barnabasfabianbakay/Documents/Documents - Barnabas’s MacBook Pro/PhD/Teaching/Predictive Analytics/PS3")
## -----------------------------------------------------------------------------
library(fpp3)

## 3.2-----------------------------------------------------------------------------
# United States GDP

us_economy <- global_economy %>%
  filter(Country == "United States")
us_economy %>%
  autoplot(GDP)


us_economy %>%
  autoplot(box_cox(GDP, 0))


us_economy %>%
  autoplot(box_cox(GDP, 0.3))


us_economy %>%
  features(GDP, features = guerrero)


us_economy %>%
  autoplot(box_cox(GDP, 0.2819714))


## -----------------------------------------------------------------------------
# Slaughter of Victorian “Bulls, bullocks and steers”

vic_bulls <- aus_livestock %>%
  filter(State == "Victoria", Animal == "Bulls, bullocks and steers")
vic_bulls %>%
  autoplot(Count)


vic_bulls %>%
  autoplot(log(Count))


vic_bulls %>%
  features(Count, features = guerrero)


## -----------------------------------------------------------------------------
# Victorian Electricity Demand
vic_elec %>%
  autoplot(Demand)


vic_elec %>%
  autoplot(box_cox(Demand, 0))



## -----------------------------------------------------------------------------
#Australian Gas production

aus_production %>%
  autoplot(Gas)


aus_production %>%
  autoplot(box_cox(Gas, 0))


aus_production %>%
  features(Gas, features = guerrero)


aus_production %>%
  autoplot(box_cox(Gas, 0.110))



## -----------------------------------------------------------------------------
china <- global_economy %>%
  filter(Country == "China")
china %>% autoplot(GDP)


## -----------------------------------------------------------------------------
china %>% autoplot(box_cox(GDP, 0.2))

## -----------------------------------------------------------------------------
china %>% features(GDP, guerrero)
china %>% autoplot(box_cox(GDP, -0.0345))

# Lambda = -0.034

## -----------------------------------------------------------------------------
fit <- china %>% 
  model(
    ets = ETS(GDP),
    ets_damped = ETS(GDP ~ trend("Ad")),
    ets_bc = ETS(box_cox(GDP, 0.2)),
    ets_log = ETS(log(GDP))
  )

fit

## -----------------------------------------------------------------------------
fit %>%
  forecast(h = "20 years") %>%
  autoplot(china, level = NULL)


## -----------------------------------------------------------------------------
aus_trips <- tourism %>%
  summarise(Trips = sum(Trips))
aus_trips %>%
  autoplot(Trips)


## -----------------------------------------------------------------------------
dcmp <- aus_trips %>%
  model(STL(Trips)) %>%
  components()
dcmp %>%
  as_tsibble() %>%
  autoplot(season_adjust)

## -----------------------------------------------------------------------------
stletsdamped <- decomposition_model(
  STL(Trips),
  ETS(season_adjust ~ error("A") + trend("Ad") + season("N"))
)
aus_trips %>%
  model(dcmp_AAdN = stletsdamped) %>%
  forecast(h = "2 years") %>%
  autoplot(aus_trips)

## -----------------------------------------------------------------------------
stletstrend <- decomposition_model(
  STL(Trips),
  ETS(season_adjust ~ error("A") + trend("A") + season("N"))
)
aus_trips %>%
  model(dcmp_AAN = stletstrend) %>%
  forecast(h = "2 years") %>%
  autoplot(aus_trips)


## -----------------------------------------------------------------------------
aus_trips %>%
  model(ets = ETS(Trips)) %>%
  forecast(h = "2 years") %>%
  autoplot(aus_trips)

## -----------------------------------------------------------------------------
fit <- aus_trips %>%
  model(
    dcmp_AAdN = stletsdamped,
    dcmp_AAN = stletstrend,
    ets = ETS(Trips)
  )
accuracy(fit)


## -----------------------------------------------------------------------------
fit %>%
  forecast(h = "2 years") %>%
  autoplot(aus_trips, level = NULL)
# The forecasts are almost identical. So I’ll use the decomposition model with 
# additive trend as it has the smallest RMSE. - (Root Mean Square Error. How well a statistical model fits a dataset.
#Distance between actual values and predicted values.)

## -----------------------------------------------------------------------------
best <- fit %>%
  select(dcmp_AAN)
augment(best) %>% gg_tsdisplay(.resid, lag_max = 24, plot_type = "histogram")

## -----------------------------------------------------------------------------
augment(best) %>%
  features(.innov, ljung_box, lag = 24, dof = 4)


augment(best) %>%
  features(.innov, ljung_box, lag = 12, dof = 4)


