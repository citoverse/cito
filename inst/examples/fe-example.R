\donttest{
if(torch::torch_is_installed()){
library(cito)

# linear effects of Temp and Temp^2, a non-linear effect of X1 learned by the DNN
set.seed(42)
n = 1000
dat = data.frame(X1 = runif(n, -2, 2), Temp = rnorm(n), Precp = rnorm(n))
dat$Y = sin(2 * dat$X1) + 1.0 * dat$Temp - 0.5 * dat$Temp^2 + rnorm(n, sd = 0.3)

m = dnn(Y ~ X1 + fe(Temp + I(Temp^2)), data = dat, loss = "mse", epochs = 50L, verbose = FALSE)

# the fixed effects, close to 1.0 and -0.5
coef(m)$fixed_effects

# "." uses every variable that is not in fe(); here X1 and Precp
m2 = dnn(Y ~ . + fe(Temp + I(Temp^2), lambda = 0.001), data = dat, loss = "mse",
         epochs = 50L, verbose = FALSE)
}
}
