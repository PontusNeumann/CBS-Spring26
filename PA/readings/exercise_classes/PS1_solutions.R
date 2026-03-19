## ----setup, include=FALSE-----------------------------------------------------
knitr::knit_hooks$set(purl = knitr::hook_purl)
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_knit$set(root.dir = "/Users/barnabasfabianbakay/Documents/Documents - Barnabas’s MacBook Pro/PhD/Teaching/Predictive Analytics/PS1")


## ----eval=FALSE,echo=T--------------------------------------------------------
#  library(readr) #load the readr package
#  caschool <- read_csv("caschool.csv")

## ----eval=FALSE,echo=T--------------------------------------------------------
#  print(caschool)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  names(caschool)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  summary(caschool)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Simple linear regression
#  model.1 <- lm(testscr ~ meal_pct, caschool)
#  summary(model.1)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Multivariate regression
#  model.2 <- lm(testscr ~ meal_pct + comp_stu, data = caschool)
#  summary(model.2)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  library(tidyverse,warn.conflicts=FALSE) #
#  p <- ggplot(caschool, aes(x = meal_pct, y = testscr)) + #The basis for the plot
#  geom_point() + #Add points for a scatterplot
#  geom_smooth(method = lm) + #Add a regression line
#  labs(x = "Percentage of Students Receiving a Free Lunch",
#  y = "Test Score",
#  title = "California Schools")
#  print(p)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  plot(caschool$meal_pct, y = caschool$testscr);abline(lm(testscr~meal_pct
#                                                          , data = caschool), col = "red")

## ----eval=FALSE,echo=T--------------------------------------------------------
#  ?mean

## ----eval=FALSE,echo=T--------------------------------------------------------
#  x <- rnorm(n = 1000, mean = 0, sd = 2)
#  norm.mean <- mean(x)
#  print(norm.mean)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  rnorm(n = 1000, mean = 0, sd = 2) %>%
#  mean(.) %>% #Take the mean
#  print(.) #Print

## ----eval=FALSE,echo=T--------------------------------------------------------
#  caschool %>%
#  filter(county == "Los Angeles") %>% #filter data
#  lm(testscr ~ meal_pct, data = .) %>% #run regression
#  summary %>% #take summary of regression output
#  print #print regress output
#  

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #Vector from 1 to 10 -- a few different ways
#  x <- 1:10
#  x <- seq(from=1,to=10,by=1)
#  x <- c(1,2,3,4,5,6,7,8,9,10)
#  #Length of x
#  length(x)
#  #Square each element using sapply
#  sapply(x,function(number) number^2)
#  #also can just use
#  x^2
#  #vector multiply x'x
#  t(x)%*%x
#  #print the first and last elements
#  x[1]
#  x[length(x)]
#  x[10] ## length(x) == 10

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #Load in some preset data. The height and weight for
#  # 507 individuals
#  install.packages("openintro")
#  library(openintro)
#  data(bdims)
#  bdims = subset(bdims, select = c(wgt,hgt))
#  bdims <- bdims %>%
#    rename(
#      weight = wgt,
#      height = hgt
#    )
#  #view the data
#  bdims
#  #view the first six observations
#  head(bdims)
#  #view the last six observations
#  tail(bdims)
#  #view the class of the dataset
#  class(bdims)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #Select the first column in the "dataframe"
#  bdims[,1]
#  bdims[,"height"]
#  bdims$height
#  #Select the second column in the dataframe
#  bdims[,2]
#  bdims[,"weight"]
#  bdims$weight
#  #Select the first row
#  bdims[1,]
#  #Select the last row
#  bdims[15,]
#  bdims[nrow(bdims),]
#  #Note: that bdims[i,j] selects the ith row and the jth column
#  bdims[10,2]

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #summarize the data
#  summary(bdims)
#  #get the mean of the data
#  apply(bdims,2,mean) #2 applies the function on columns.
#  sapply(bdims,mean)
#  lapply(bdims, mean) #Results will be a list
#  #the standard deviation
#  apply(bdims,2,sd)
#  sapply(bdims,sd)
#  lapply(bdims, sd)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #Run a linear regression of height on weight
#  bdims.model <- lm(height ~ weight,data=bdims)
#  #Look at a summary of the output
#  summary(bdims.model)
#  ##White standard errors
#  
#  # Here we need to install the sandwich package for the white standard errors
#  # install. packages(“sandwich”), if you have it installed just load the package with library(sandwich) or require(sandwich)
#  
#  require(lmtest);require(sandwich)
#  
#  
#  coeftest(bdims.model,vcov=sandwich)
#  # vcov = sandwich means that the variance / covariance matrix uses White
#  # standard errors (robustness checks)
#  #plot the data
#  plot(bdims)
#  pairs(bdims)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #BMI function
#  bmi.func <- function(cm,kg) {
#  bmi <- kg/((cm/100)^2)
#  return(bmi)
#  }
#  bdims$bmi <- bmi.func(bdims$height,bdims$weight)
#  bdims

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #clear the workspace
#  rm(list=ls())
#  #Set the seed
#  set.seed(123456)
#  #the number of observations
#  n <- 100
#  #Two random series from the normal distribution
#  x1 <- rnorm(n)
#  x2 <- rnorm(n)
#  #(a) the model
#  model <- lm(x1 ~ x2)
#  #The output we will try and replicate
#  summary(model)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #The dependent variable
#  Y <- x1
#  #The matrix of independent variables. Notice how we include the constant
#  X <- matrix(c(rep(1,n),x2),nrow=n,ncol=2)
#  #The regression estimator
#  bhat <- inv(t(X)%*%X)%*%t(X)%*%Y # (X'X)^(-1)X'Y
#  bhat #same as output from R!
#  #Get the residuals
#  residuals <- Y - X%*%bhat
#  cor(residuals,model$residuals) #equals 1 same as R!

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #Get the variance of the residuals --> note that we have two
#  #independent variables
#  resid.var <- as.numeric(t(residuals)%*%residuals)*(1/(n-2))
#  #Get the standard error of the residuals --> in regression analysis,
#  #this is just the square root of the variance of the residuals
#  se.resid <- sqrt(resid.var)
#  se.resid #Same as R!
#  #Get the variance-cov matrix
#  varCov <- inv(t(X)%*%X)*resid.var
#  varCov
#  #Compare this to the var-cov matrix from R
#  vcov(model)
#  #They are the same!
#  #Get the standard errors using our varCov matrix
#  se <- sqrt(diag(varCov))
#  se
#  #compare these to the standard errors from R
#  sqrt(diag(vcov(model))) #Same!!
#  #or
#  summary(model)

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #Now get the t-stats. Note: our null is that the regression estimators
#  #are equal to zero. This will be a two-sided test
#  tstat <- bhat/se
#  tstat
#  #Compare these to R
#  summary(model) #Same!
#  #p-values. this is a two sided test
#  #First get the degrees of freedom. df = n-2 since
#  #we one independent variable plus a constant
#  df <- n-2
#  #Use the cdf of the t distribution
#  pvals <- 2*(1 - pt(abs(tstat),df))
#  pvals #same as R!
#  #The F-statistic
#  #RSS under the null
#  RSS_0 <- as.numeric(t((Y-mean(Y)))%*%(Y-mean(Y)))
#  #RSS under the alternative
#  RSS_1 <- as.numeric(t(residuals)%*%residuals)
#  #The numerator of the F-stat
#  Fnum <- (RSS_0 - RSS_1)/(2-1)
#  #The denominator of the F-stat
#  Fdenom <- (RSS_1)/(n-2)
#  #The F-stat
#  F <- Fnum/Fdenom
#  F #Same as R!
#  #The p-value for the f-stat
#  dfnum <- 2-1 #the degrees of freedom for the numerator
#  dfdenom <- n-2 #the df for the denominator
#  Fpval <- 1 - pf(F,dfnum,dfdenom)
#  Fpval #Same as R

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  #The R^2
#  #We already defined the Residual sum of squares above as RSS_1
#  #Define the total sum of squares
#  TSS <- as.numeric(t((Y-mean(Y)))%*%(Y-mean(Y)))
#  R2 <- 1 - RSS_1/TSS
#  R2 #same as R!
#  #The adjusted R^2 -- For the adjusted r-squared we
#  #divide the numerator and the denominator by the degrees
#  #of freedom.
#  R2_adj <- 1 - ((1/(n-2))*RSS_1)/((1/(n-1))*TSS)
#  R2_adj # Same as R!

## ----eval=FALSE,echo=T--------------------------------------------------------
#  # Load the fpp3 package that corresponds to the book
#  library(fpp3)
#  #Solution
#  summary(gafa_stock)
#  
#  gafa_stock %>%
#    group_by(Symbol) %>%
#    filter(Close == max(Close)) %>%
#    ungroup() %>%
#    select(Symbol, Date, Close)
#  

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  download.file("http://OTexts.com/fpp3/extrafiles/tourism.xlsx",
#                tourism_file <- tempfile())

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  my_tourism <- readxl::read_excel(tourism_file) %>%
#    mutate(Quarter = yearquarter(Quarter)) %>%
#    as_tsibble(
#      index = Quarter,
#      key = c(Region, State, Purpose)
#    )
#  
#  my_tourism
#  

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  my_tourism %>%
#    as_tibble() %>%
#    group_by(Region, Purpose) %>%
#    summarise(Trips = mean(Trips)) %>%
#    ungroup() %>%
#    filter(Trips == max(Trips))

## ----eval=FALSE,echo=T--------------------------------------------------------
#  #Solution
#  state_tourism <- my_tourism %>%
#    group_by(State) %>%
#    summarise(Trips = sum(Trips)) %>%
#    ungroup()
#  state_tourism

