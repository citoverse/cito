
<!-- README.md is generated from README.Rmd. Please edit that file -->

# cito

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/cito)](https://cran.r-project.org/package=cito)
[![R-CMD-check](https://github.com/citoverse/cito/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/citoverse/cito/actions/workflows/R-CMD-check.yaml)

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

If you have trouble installing ‘torch’, please [visit the website of the
‘torch’
package](https://torch.mlverse.org/docs/articles/installation.html) or
create an issue on [our github
website](https://github.com/citoverse/cito/issues). We are happy to help
you.

A stable version of cito from CRAN can be installed with:

``` r
install.packages("cito")
```

The development version from [GitHub](https://github.com/) can be
installed by:

``` r
if(!require('devtools', quietly = TRUE)) install.packages('devtools')
devtools::install_github('citoverse/cito')
```

## Example

Once installed, the main function `dnn()` can be used. See the example
below. A more in depth explanation can be found in the vignettes or
[here under articles](https://citoverse.github.io/cito/).

1.  Fit model with bootstrapping (to obtain confidence intervals). All
    methods work with and without bootstrapping

``` r
library(cito)
nn.fit <- dnn(Sepal.Length~., data = datasets::iris, bootstrap = 30L)
```

2.  Check if models have converged (compare training loss against
    baseline loss (=intercept only model)):

``` r
analyze_training(nn.fit)
# At 1st glance, the networks converged since the loss is lower than the baseline loss and the training loss is on a plateau at the end of the training.
```

3.  Plot model architecture

``` r
plot(nn.fit)
```

![](README_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

4.  Return xAI effects (feature importances and average conditional
    effects) and their uncertainties:

``` r
summary(nn.fit)
```

    ## Summary of Deep Neural Network Model

    ## 

    ## ── Feature Importance

    ##                             Importance Std.Err Z value Pr(>|z|)   
    ## Sepal.Width → Sepal.Length       1.755   0.570    3.08   0.0021 **
    ## Petal.Length → Sepal.Length     18.027   8.809    2.05   0.0407 * 
    ## Petal.Width → Sepal.Length       0.521   0.716    0.73   0.4666   
    ## Species → Sepal.Length           0.392   0.218    1.80   0.0720 . 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    ## 

    ## ── Average Conditional Effects

    ##                                 ACE Std.Err Z value Pr(>|z|)    
    ## Sepal.Width → Sepal.Length   0.7326  0.0497   14.74   <2e-16 ***
    ## Petal.Length → Sepal.Length  0.6164  0.0901    6.84    8e-12 ***
    ## Petal.Width → Sepal.Length  -0.1181  0.1587   -0.74     0.46    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    ## 

    ## ── Standard Deviation of Conditional Effects

    ##                                ACE Std.Err Z value Pr(>|z|)    
    ## Sepal.Width → Sepal.Length  0.1482  0.0432    3.43   0.0006 ***
    ## Petal.Length → Sepal.Length 0.1167  0.0396    2.94   0.0032 ** 
    ## Petal.Width → Sepal.Length  0.0470  0.0268    1.76   0.0790 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

5.  Predict (with confidence intervals):

``` r
dim(predict(nn.fit, newdata = datasets::iris))
```

    ## [1]  30 150   1

## Advanced

We can pass custom loss functions to ‘cito’, optionally with additional
parameters that should be fitted. The only requirement is that all
calculations must be written using the ‘torch’ package (cito
automatically converts the initial values of the custom parameters to
‘torch’ objects).

We use a multivariate normal distribution as the likelihood function and
we want to parameterize/fit the covariance matrix of the multivariate
normal distribution:

1.  We need one helper function, `create_cov()` that builds the
    covariance matrix based on a lower triangular matrix and the
    diagonals (low-rank approximation of the covariance matrix)

2.  We need our custom likelihood function which uses the
    `distr_multivariate_normal(…)` function from the torch package:

``` r
create_cov = function(L, Diag) {
  return(torch::torch_matmul(L, L$t()) + torch::torch_diag(Diag+0.01))
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
    ## [1,] 0.24824733 0.06395083 0.13815881
    ## [2,] 0.06395083 0.11858207 0.03379684
    ## [3,] 0.13815881 0.03379684 0.24378222

Empirical covariance matrix:

``` r
cov(predict(nn.fit) - nn.fit$data$Y)
```

    ##              Sepal.Length Sepal.Width Petal.Length
    ## Sepal.Length   0.25081444  0.06944643   0.15270816
    ## Sepal.Width    0.06944643  0.09439328   0.02924789
    ## Petal.Length   0.15270816  0.02924789   0.16690478
