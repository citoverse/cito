# Summarize a Neural Network of class citodnn

Calculates feature importance and average conditional effects for a
trained model.

## Usage

``` r
# S3 method for class 'citodnn'
summary(
  object,
  importance = c("ale", "ce", "permutation"),
  n_permute = 10L,
  device = NULL,
  type = c("response", "link"),
  ...
)

# S3 method for class 'citodnnBootstrap'
summary(
  object,
  importance = c("ce", "permutation"),
  n_permute = 10,
  device = NULL,
  adjust_se = FALSE,
  type = c("response", "link"),
  ...
)
```

## Arguments

- object:

  a model of class citodnn created by [`dnn`](dnn.md)

- importance:

  method used to compute feature importance: "ale" (default) or "ce"
  (both derived from the conditional effects, numerical features only),
  or "permutation" (all features, computationally expensive). For
  bootstrapped models "ce" (default) or "permutation".

- n_permute:

  number of permutations used to compute the permutation feature
  importance. Default is 10.

- device:

  device used to calculate the feature importance and conditional
  effects

- type:

  scale on which the average conditional effects are calculated
  ("response" or "link")

- ...:

  additional arguments (currently not used)

- adjust_se:

  if TRUE, the standard errors of the importance are multiplied by
  1/sqrt(3)

## Value

an object of class "summary.citodnn" containing the feature importance
and average conditional effects

## Details

The summary reports the feature importance (following Fisher, Rudin, and
Dominici, 2018) together with the mean and standard deviation of the
average conditional effects (following Pichler & Hartig, 2023).

Feature importances are interpreted similarly to an ANOVA: main and
interaction effects are absorbed into the individual features. They are
also sensitive to collinearity between features, i.e. if two features
are collinear, their importances may be overestimated.

Average conditional effects (ACE) are similar to marginal effects and
approximate linear effects, i.e. their interpretation is similar to the
coefficients in a linear regression model.

The standard deviation of the ACE quantifies the non-linearity of a
feature effect: higher values indicate stronger non-linearities.

For the permutation importance, the loss of the model is evaluated on
the original and on the permuted feature (\\e\_{perm}\\ and
\\e\_{orig}\\), and the importance of feature j is reported as the
relative increase \\FI_j = e\_{perm}/e\_{orig} - 1\\.

The "ale" and "ce" importances are derived from the conditional effects
and are therefore only defined for numerical features; categorical
features are left out of the importance table (a message says which).
Use `importance = "permutation"` to include them. For bootstrapped
models, only "ce" and "permutation" are available.
