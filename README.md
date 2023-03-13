
# cito

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/cito)](https://cran.r-project.org/package=cito) 
[![R-CMD-check](https://github.com/citoverse/cito/workflows/R-CMD-check/badge.svg)](https://github.com/citoverse/cito/actions)

<!-- badges: end -->

'cito' aims at helping you build and train Neural Networks with the standard R syntax. It allows the whole model creation process and training to be done with one line of code. Furthermore, all generic R methods such as print or  plot can be used on the created object. It is based on the 'torch' machine learning framework which is  available for R. Since it is native to R, no Python installation or any further API is needed for this package. 

## Installation
Before installing cito make sure torch is installed. See the code chunk below if you are unsure on how to check this 


``` r
#check package 
if(!require('torch',quietly = TRUE)) install.packages('torch')
library('torch') 

#install torch
if(!torch_is_installed()) install_torch()
```

A stable version of cito from CRAN can be installed with: 

```r 
install.packages("cito")

```

The development version from [GitHub](https://github.com/) is available with:

``` r
if(!require('devtools', quietly = TRUE)) install.packages('devtools')
devtools::install_github('citoverse/cito')
```


## Example 
Once installed, the main function dnn() can be used. See the example below. A more in depth explanation can be found in the vignette.

``` r
library(cito)
validation_set <- sample(c(1:nrow(datasets::iris)),25)

# Build and train  Network
nn.fit <- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])

# Analyze training 
analyze_training(nn.fit)

# Print sturcture of Neural Network
print(nn.fit)

# Plot Structure of Neural Network 
plot(nn.fit)

# continue training for another 32 epochs
nn.fit< - continue_training(nn.fit) 

# Use model on validation set
predictions <- predict(nn.fit, iris[validation_set,])

# Scatterplot
plot(iris[validation_set,]$Sepal.Length,predictions)
# MAE
mean(abs(predictions-iris[validation_set,]$Sepal.Length))
``` 
