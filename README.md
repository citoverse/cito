
<!-- README.md is generated from README.Rmd. Please edit that file -->

# cito

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/cito)](https://cran.r-project.org/package=cito)
[![R-CMD-check](https://github.com/citoverse/cito/workflows/R-CMD-check/badge.svg)](https://github.com/citoverse/cito/actions)

<!-- badges: end -->

‘cito’ simplifies the building and training of (deep) neural networks by
relying on standard R syntax and familiar methods from statistical
packages. Model creation and training can be done with a single line of
code. Furthermore, all generic R methods such as print or plot can be
used on the fitted model. At the same time, ‘cito’ is computationally
efficient because it is based on the deep learning framework ‘torch’
(with optional GPU support). The ‘torch’ package is native to R, so no
Python installation or other API is required for this package.

## Installation

Before installing ‘cito’ make sure ‘torch’ is installed. See the code
chunk below if you are unsure on how to check this

``` r
# check package 
if(!require('torch',quietly = TRUE)) install.packages('torch')
library('torch') 

#install torch
if(!torch_is_installed()) install_torch()
```

If you have trouble installing ‘torch’, please [visit their
website](https://torch.mlverse.org/docs/articles/installation.html) or
create an issue on [our github
website](https://github.com/citoverse/cito/issues). We are happy to
help.

A stable version of cito from CRAN can be installed with:

``` r
install.packages("cito")
```

The development version from [GitHub](https://github.com/) is available
with:

``` r
if(!require('devtools', quietly = TRUE)) install.packages('devtools')
devtools::install_github('citoverse/cito')
```

## Example

Once installed, the main function `dnn()` can be used. See the example
below. A more in depth explanation can be found in the vignettes.

1.  Fit model with bootstrapping (for uncertainties)

``` r
library(cito)
nn.fit <- dnn(Sepal.Length~., data = datasets::iris, bootstrap = 30L)
```

2.  Check if models have converged (compare training loss against
    baseline loss (=intercept only model)):

``` r
analyze_training(nn.fit)
# At 1st glance they are converged since the loss is lower than the baseline loss.
```

3.  Plot model architecture

``` r
plot(nn.fit)
```

![](README_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

4.  Return xAI effects and their uncertainties:

``` r
summary(nn.fit)
```

    ## Summary of Deep Neural Network Model
    ## 
    ##  ##########################################################
    ##      Feature Importance 
    ##  ##########################################################
    ##                          Importance Std.Err Z value Pr(>|z|)    
    ## Response_1: Sepal.Width       1.815   0.557    3.26  0.00113 ** 
    ## Response_1: Petal.Length     19.211   5.563    3.45  0.00055 ***
    ## Response_1: Petal.Width       0.485   0.475    1.02  0.30690    
    ## Response_1: Species           0.459   0.277    1.65  0.09826 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## 
    ##  ##########################################################
    ##      Average Conditional Effects 
    ##  ##########################################################
    ##                              ACE Std.Err Z value Pr(>|z|)    
    ## Response_1: Sepal.Width   0.7125  0.0791    9.01   <2e-16 ***
    ## Response_1: Petal.Length  0.6436  0.0664    9.70   <2e-16 ***
    ## Response_1: Petal.Width  -0.1560  0.1360   -1.15     0.25    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## 
    ##  ##########################################################
    ##      Standard Deviation of Conditional Effects 
    ##  ##########################################################
    ##                             ACE Std.Err Z value Pr(>|z|)    
    ## Response_1: Sepal.Width  0.1466  0.0430    3.41  0.00064 ***
    ## Response_1: Petal.Length 0.1271  0.0411    3.09  0.00200 ** 
    ## Response_1: Petal.Width  0.0487  0.0282    1.72  0.08482 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

5.  Predict (with confidence intervals):

``` r
dim(predict(nn.fit, newdata = datasets::iris))
```

    ## [1]  30 150   1

## Advanced

We can also pass custom loss functions to ‘cito’, and we can use
additional parameters within the custom loss function. The only
requirement is that all calculations must be written using the ‘torch’
package (cito automatically converts the initial values for the custom
parameters to ‘torch’ objects).

We use a multivariate normal distribution as the likelihood function and
we want to parameterize/fit the covariance matrix of the multivariate
normal distribution:

1.  We need one helper function, `create_cov()` that builds the
    covariance matrix based on a LU and the diagonals

2.  We need our custom likelihood function which uses the
    `distr_multivariate_normal(…)` function from the torch package:

``` r
create_cov = function(LU, Diag) {
  return(torch::torch_matmul(LU, LU$t()) + torch::torch_diag(Diag+0.01))
}

custom_loss_MVN = function(true, pred) {
  Sigma = create_cov(SigmaPar, SigmaDiag)
  logLik = torch::distr_multivariate_normal(pred,
                                            covariance_matrix = Sigma)$
    log_prob(true)
  return(-logLik$mean())
}
```

3.  We use “SigmaPar” and “SigmaDiag” as parameters that we want to
    optimize along the DNN. We will pass a named list with starting
    values to ‘cito’ and ‘cito’ will infer automatically (based on the R
    shape) the shape of the parameters:

``` r
nn.fit<- dnn(cbind(Sepal.Length, Sepal.Width, Petal.Length)~.,
             data = datasets::iris,
             lr = 0.01,
             epochs = 200L,
             loss = custom_loss_MVN,
             verbose = FALSE,
             plot = FALSE,
             custom_parameters =
               list(SigmaDiag =  rep(1, 3), # Our parameters with starting values
                    SigmaPar = matrix(rnorm(6, sd = 0.001), 3, 2)) # Our parameters with starting values
)
```

Estimated covariance matrix:

``` r
as.matrix(create_cov(nn.fit$loss$parameter$SigmaPar,
                     nn.fit$loss$parameter$SigmaDiag))
```

    ##            [,1]       [,2]       [,3]
    ## [1,] 0.23374167 0.07131696 0.13503927
    ## [2,] 0.07131696 0.09627343 0.03110428
    ## [3,] 0.13503927 0.03110428 0.19582744

Empirical covariance matrix:

``` r
cov(predict(nn.fit) - nn.fit$data$Y)
```

    ##              Sepal.Length Sepal.Width Petal.Length
    ## Sepal.Length   0.22601251  0.05944633   0.12786447
    ## Sepal.Width    0.05944633  0.09067426   0.01754041
    ## Petal.Length   0.12786447  0.01754041   0.14744435
