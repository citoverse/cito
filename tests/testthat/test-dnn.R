
wrap_dnn = function(pars) {
  testthat::expect_error({model = do.call(dnn, pars)}, NA)
  testthat::expect_error({predict(model, newdata=pars$X)}, NA)
  testthat::expect_error({predict(model)}, NA)
  testthat::expect_error({predict(model, type = "response")}, NA)
  testthat::expect_error({print(model)}, NA)
  testthat::expect_error({coef(model)}, NA)
  testthat::expect_error({plot(model)}, NA)
  testthat::expect_error({summary(model)}, NA)
  testthat::expect_error({plot(PDP(model, variable = "X.1"))}, NA)
  testthat::expect_error({plot(ALE(model, variable = "X.1"))}, NA)
}


X = matrix(runif(3*50), 50, 3)
Y = matrix(rbinom(3*50, 1, 0.5), 50, 3)
data = data.frame(Y = Y, X = X)

#### Test scenarios ####
# Architecture
scenarios =
  list(
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L, hidden = NULL),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L, hidden = c(2L, 5L)),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L, hidden = c(9, 1L, 3L)),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("relu", "tanh")),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh")),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh"), bias=FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh"), bias=c(TRUE, FALSE))
)

testthat::test_that("DNN architecture", {
  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }
})

# Family
X = matrix(runif(3*50), 50, 3)
Y = matrix(as.character(rbinom(50, 3, 0.5)))
data = data.frame(Y = Y, X = X)
scenarios =
  list(
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE,data = data, family = "softmax", epochs = 1L),
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE,data = data, family = stats::binomial(), epochs = 1L)
  )
testthat::test_that("DNN softmax/binomial", {
  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }
})


X = matrix(runif(3*50), 50, 3)
Y = matrix(rpois(50*3,lambda = 2), 50, 3)
data = data.frame(Y = Y, X = X)
scenarios =
  list(
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, family = stats::gaussian(), epochs = 1L),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::gaussian(), epochs = 1L),
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, family = stats::poisson(), epochs = 1L),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE,data = data, family = stats::poisson(), epochs = 1L)
  )
testthat::test_that("DNN rnorm/poisson", {
  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }
})


testthat::test_that("DNN save and reload", {
  testthat::skip_on_cran()
  set.seed(222)
  validation_set<- sample(c(1:nrow(datasets::iris)),25)

  # Build and train  Network
  nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,], epochs = 5L)
  saveRDS(nn.fit, "test_model.RDS")
  nn.fit = readRDS("test_model.RDS")
  testthat::expect_error(predict(nn.fit), NA)
  testthat::expect_error(predict(nn.fit, newdata = datasets::iris[validation_set,]), NA)
  testthat::expect_error(continue_training(nn.fit,epochs = 5), NA)
  file.remove("test_model.RDS")
})
