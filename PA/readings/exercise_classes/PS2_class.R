
rm(list =ls())

##2.9 

# Load libraries:
library(fpp3) # here we only need fpp3 the library of the book that includes all the tidyverse family.

#Total Private Employment in the US
us_employment %>%
  filter(Title == "Total Private") %>%
  autoplot(Employed)


###
us_employment %>%
  filter(Title == "Total Private") %>%
  gg_season(Employed)


us_employment %>%
  filter(Title == "Total Private") %>%
  gg_subseries(Employed)

us_employment %>%
  filter(Title == "Total Private") %>%
  gg_lag(Employed)

us_employment %>%
  filter(Title == "Total Private") %>%
  ACF(Employed) %>%
  autoplot()


# Brick production in Australia

aus_production %>%
  autoplot(Bricks)



aus_production %>%
  gg_season(Bricks)



aus_production %>%
  gg_subseries(Bricks)


aus_production %>%
  gg_lag(Bricks, geom='point')

aus_production %>%
  ACF(Bricks) %>% autoplot()



# Snow hare trappings in Canada

pelt %>%
  autoplot(Hare)

pelt %>%
  gg_lag(Hare, geom='point')

pelt %>%
  ACF(Hare) %>% autoplot()

####--
h02 <- PBS %>%
  filter(ATC2 == "H02") %>%
  group_by(ATC2) %>%
  summarise(Cost = sum(Cost)) %>%
  ungroup()

h02 %>%
  autoplot(Cost)

h02 %>%
  gg_season(Cost)

h02 %>%
  gg_subseries(Cost)


h02 %>%
  gg_lag(Cost, geom='point', lags=1:16)

h02 %>%
  ACF(Cost) %>% autoplot()



# US gasoline sales

us_gasoline %>%
  autoplot(Barrels)


us_gasoline %>%
  gg_season(Barrels)


us_gasoline %>%
  gg_subseries(Barrels)


us_gasoline %>%
  gg_lag(Barrels, geom='point', lags=1:16)

us_gasoline %>%
  ACF(Barrels, lag_max = 150) %>% autoplot()

## 3.9

## a)


## b)



##5.10

## a)
takeaway <- aus_retail %>%
  filter(Industry == "Takeaway food services") %>%
  summarise(Turnover = sum(Turnover))
train <- takeaway %>%
  filter(Month <= max(Month) - 4 * 12)

## b)

fit <- train %>%
  model(
    naive = NAIVE(Turnover),
    drift = RW(Turnover ~ drift()),
    mean = MEAN(Turnover),
    snaive = SNAIVE(Turnover)
  )
fc <- fit %>% forecast(h = "4 years")

## c)

fc %>%
  accuracy(takeaway) %>%
  arrange(MASE)



## d)

fit %>%
  select(naive) %>%
  gg_tsresiduals()

