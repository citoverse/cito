source("utils.R")

wrap_dnn = function(pars) {
  testthat::expect_error({model = do.call(dnn, pars)}, NA)
  testthat::expect_error({predict(model, newdata=pars$X)}, NA)
  testthat::expect_error({predict(model)}, NA)
  testthat::expect_error({predict(model, type = "response")}, NA)
  testthat::expect_error({print(model)}, NA)
  testthat::expect_error({coef(model)}, NA)
  testthat::expect_error({plot(model)}, NA)
  testthat::expect_error({residuals(model)}, NA)
  testthat::expect_error({summary(model)}, NA)
  testthat::expect_error({PDP(model, variable = "X.1")}, NA)
  testthat::expect_error({ALE(model, variable = "X.1")}, NA)
}


X = matrix(runif(3*50), 50, 3)
Y = matrix(rbinom(3*50, 1, 0.5), 50, 3)
data = data.frame(Y = Y, X = X)

#### Test scenarios ####
# Architecture
scenarios =
  list(
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = NULL),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(2L, 5L)),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L, 3L)),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("relu", "tanh")),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh")),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh"), bias=FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh"), bias=c(TRUE, FALSE))
)

testthat::test_that("DNN architecture", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

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
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = "softmax", epochs = 1L),
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L)
  )
testthat::test_that("DNN softmax/binomial", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }
})


X = matrix(runif(3*50), 50, 3)
Y = matrix(rpois(50*3,lambda = 2), 50, 3)
data = data.frame(Y = Y, X = X)
scenarios =
  list(
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = stats::gaussian(), epochs = 1L),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::gaussian(), epochs = 1L),
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = stats::poisson(), epochs = 1L),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::poisson(), epochs = 1L)
  )
testthat::test_that("DNN rnorm/poisson", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }
})


testthat::test_that("DNN save and reload", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

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



testthat::test_that("DNN custom loss and custom parameters", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

custom_loss = function(true, pred) {
  logLik = torch::distr_normal(pred,
                               scale = torch::nnf_relu(scale)+
                                 0.001)$log_prob(true)
  return(-logLik$mean())
}

testthat::expect_error({
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  nn.fit<- dnn(Sepal.Length~.,
               data = datasets::iris[],
               loss = custom_loss,
               epochs = 2L,
               custom_parameters = list(scale = 1.0)
  )
  }, NA)

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

testthat::expect_error({
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

nn.fit<- dnn(cbind(Sepal.Length, Sepal.Width, Petal.Length)~.,
             data = datasets::iris,
             lr = 0.01,
             epochs = 2L,
             loss = custom_loss_MVN,
             custom_parameters =
               list(SigmaDiag =  rep(1., 3),
                    SigmaPar = matrix(rnorm(6, sd = 0.001), 3, 2))
)
}, NA)

testthat::expect_error("DNN coef accuracy check",{


  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  data <- as.data.frame(matrix(rnorm(n=200*10, mean= 0, sd=1),nrow=200,ncol=10))
  coefs <- runif(n=10)
  data$Y <- apply(data,1,function(x) sum(x*coefs))

  nn.fit<- dnn(Y~., data=data, hidden= NULL,epochs=200)
  if(!all(abs((unlist(coef(nn.fit))[-1] - coefs)) < 1e-02)){
    warning("Linear Model not converged")
  }

}, NA)



})






