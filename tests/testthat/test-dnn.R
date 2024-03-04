source("utils.R")

wrap_dnn = function(pars) {
  testthat::expect_error({model <<- do.call(dnn, pars)}, NA)
  testthat::expect_error({.n = predict(model, newdata=pars$X)}, NA)
  testthat::expect_error({.n = continue_training(model, epochs = 2L, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = predict(model, type = "response")}, NA)
  testthat::expect_error({.n = coef(model)}, NA)
  testthat::expect_error({.n = plot(model)}, NA)
  testthat::expect_error({.n = residuals(model)}, NA)
  testthat::expect_error({.n = summary(model)}, NA)
  suppressWarnings(testthat::expect_error({.n = PDP(model, variable = "X.1")}, NA))
  suppressWarnings(testthat::expect_error({.n = ALE(model, variable = "X.1")}, NA))
}


X = matrix(runif(3*50), 50, 3)
Y = matrix(rbinom(3*50, 1, 0.5), 50, 3)
data = data.frame(Y = Y, X, Cat = as.factor(rep(1:5, 10)))

#### Test scenarios ####
# Architecture
scenarios =
  list(
    list(device = "cpu", formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ X1+X2+X3+e(Cat)"), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = NULL),
    list(device = "cpu", formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ X1+X2+X3+e(Cat)"), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(2L, 5L)),
    list(device = "cpu", formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ X1+X2+X3+e(Cat)"), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L, 3L)),
    list(device = "cpu", formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ X1+X2+X3+e(Cat)"), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("relu", "tanh")),
    list(device = "cpu", formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ X1+X2+X3+e(Cat)"), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh")),
    list(device = "cpu", formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ X1+X2+X3+e(Cat)"), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh"), bias=FALSE),
    list(device = "cpu", formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ X1+X2+X3+e(Cat)"), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, hidden = c(9, 1L), activation = c("leaky_relu", "tanh"), bias=c(TRUE, FALSE))
)

testthat::test_that("DNN architecture", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  for(i in 1:length(scenarios)) {
    .n = wrap_dnn(scenarios[[i]])
  }

  if(  torch::cuda_is_available() ) {
    for(i in 1:length(scenarios)) {
      sc = scenarios[[i]]
      sc$device = "cuda"
      .n = wrap_dnn(sc)
    }
  }

  if(  torch::backends_mps_is_available() ) {
    for(i in 1:length(scenarios)) {
      sc = scenarios[[i]]
      sc$device = "mps"
      .n = wrap_dnn(sc)
    }
  }

})

# Family
X = matrix(runif(3*50), 50, 3)
Y = matrix(as.character(rbinom(50, 3, 0.5)))
data = data.frame(Y = Y, X = X)
scenarios =
  list(
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = "softmax", epochs = 1L),
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L),
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = "softmax", epochs = 1L, bootstrap = 2L),
    list(formula = stats::as.formula("Y ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::binomial(), epochs = 1L, bootstrap = 2L)
  )
testthat::test_that("DNN softmax/binomial", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }

  if(  torch::cuda_is_available() ) {
    for(i in 1:length(scenarios)) {
      sc = scenarios[[i]]
      sc$device = "cuda"
      .n = wrap_dnn(sc)
    }
  }

  if(  torch::backends_mps_is_available() ) {
    for(i in 1:length(scenarios)) {
      sc = scenarios[[i]]
      sc$device = "mps"
      .n = wrap_dnn(sc)
    }
  }
})


X = matrix(runif(3*50), 50, 3)
Y = matrix(rpois(50*3,lambda = 2), 50, 3)
data = data.frame(Y = Y, X = X)
scenarios =
  list(
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = stats::gaussian(), epochs = 1L , verbose = FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::gaussian(), epochs = 1L),
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = stats::poisson(), epochs = 1L, verbose = FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::poisson(), epochs = 1L),

    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = stats::gaussian(), epochs = 1L, bootstrap = 2L, verbose = FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::gaussian(), epochs = 1L, bootstrap = 2L),
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = stats::poisson(), epochs = 1L, bootstrap = 2L, verbose = FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = stats::poisson(), epochs = 1L, bootstrap = 2L)
  )
testthat::test_that("DNN rnorm/poisson", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }

  if(  torch::cuda_is_available() ) {
    for(i in 1:length(scenarios)) {
      sc = scenarios[[i]]
      sc$device = "cuda"
      .n = wrap_dnn(sc)
    }
  }

  if(  torch::backends_mps_is_available() ) {
    for(i in 1:length(scenarios)) {

      ## Poisson not yet supported, skip for now
      if(! (i %in% c(3, 4, 7,8))) {
        sc = scenarios[[i]]
        sc$device = "mps"
        .n = wrap_dnn(sc)
      }
    }
  }
})


X = matrix(runif(3*50), 50, 3)
Y = matrix(rpois(50*3,lambda = 2), 50, 3)
data = data.frame(Y = Y, X = X)
scenarios =
  list(
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = "mae", epochs = 1L, verbose = FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = "mae", epochs = 1L),
    list(formula = stats::as.formula("Y.1 ~ ."), plot=FALSE,data = data, loss = "mae", epochs = 1L, bootstrap = 2L, verbose = FALSE),
    list(formula = stats::as.formula("cbind(Y.1, Y.2, Y.3) ~ ."), plot=FALSE, verbose = FALSE, data = data, loss = "mae", epochs = 1L, bootstrap = 2L)
  )
testthat::test_that("DNN mae", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  for(i in 1:length(scenarios)) {
    wrap_dnn(scenarios[[i]])
  }

  if(  torch::cuda_is_available() ) {
    for(i in 1:length(scenarios)) {
      sc = scenarios[[i]]
      sc$device = "cuda"
      .n = wrap_dnn(sc)
    }
  }

  if(  torch::backends_mps_is_available() ) {
    for(i in 1:length(scenarios)) {
      sc = scenarios[[i]]
      sc$device = "mps"
      .n = wrap_dnn(sc)
    }
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
  nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,], epochs = 5L, verbose = FALSE, plot = FALSE)
  saveRDS(nn.fit, "test_model.RDS")
  nn.fit = readRDS("test_model.RDS")
  testthat::expect_error({.n = predict(nn.fit)}, NA)
  testthat::expect_error({.n = predict(nn.fit, newdata = datasets::iris[validation_set,])}, NA)
  testthat::expect_error({.n = continue_training(nn.fit,epochs = 5)}, NA)
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
               verbose = FALSE,
               plot = FALSE,
               custom_parameters = list(scale = 1.0)
  )
  }, NA)

create_cov = function(LU, Diag) {
  return(torch::torch_matmul(LU, LU$t()) + torch::torch_diag(Diag+0.1))
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
             epochs = 200L,
             loss = custom_loss_MVN,
             verbose = FALSE,
             plot = FALSE,
             custom_parameters =
               list(SigmaDiag =  rep(1., 3),
                    SigmaPar = matrix(rnorm(6, sd = 0.001), 3, 2))
)
}, NA)

})



testthat::test_that("DNN coef accuracy check",{

  #testthat::skip_on_cran()
  #testthat::skip_on_ci()
  skip_if_no_torch()

  data <- as.data.frame(matrix(rnorm(n=200*10, mean= 0, sd=1),nrow=200,ncol=10))
  coefs <- runif(n=10)
  data$Y <- apply(data,1,function(x) sum(x*coefs))

  nn.fit<- dnn(Y~., data=data, hidden= NULL,epochs=200, verbose = FALSE, plot = FALSE)

  testthat::expect_lt(max(abs((unlist(coef(nn.fit))[-1] - coefs))), 1e02)

})



testthat::test_that("DNN baseline loss check",{

  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y = rbinom(50, 1, 0.5)
  X = rnorm(50)
  m = dnn(Y~., data = data.frame(Y = Y, X = X), loss = "binomial", epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal( !!m$base_loss, !!(-sum(dbinom(Y, 1, (mean(Y)), log = TRUE)/50)), tolerance = 0.01)

  Y = rpois(50, 5)
  X = rnorm(50)
  m = dnn(Y~., data = data.frame(Y = Y, X = X), loss = "poisson", epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal( !!m$base_loss, !!(-sum(dpois(Y, (mean(Y)), log = TRUE)/50)), tolerance = 0.01)

  Y = rnorm(50, 5)
  X = rnorm(50)
  m = dnn(Y~., data = data.frame(Y = Y, X = X), loss = "mse", epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal( !!m$base_loss, !!mean((Y - mean(Y))**2 ), tolerance = 0.01)

  Y = rbinom(50, 2, 0.5)
  X = rnorm(50)
  m = dnn(Y~., data = data.frame(Y = as.factor(Y+1), X = X), loss = "softmax", epochs = 2L, verbose = FALSE, plot = FALSE)
  pred = log(matrix(table(as.factor(Y+1))/sum(table(as.factor(Y+1))), 50, 3, byrow = TRUE)) + log(3)
  loss = as.numeric(torch::nnf_cross_entropy(pred, torch::torch_tensor(Y+1, dtype = torch::torch_long())))
  testthat::expect_equal( !!m$base_loss, !!loss , tolerance = 0.01)

  })


testthat::test_that("DNN hyperparameter tuning",{
  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(), tuning=config_tuning(steps=2, CV = 2))}, NA)
  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(values=c(0.01, 0.1)), tuning=config_tuning(steps=2, CV = 2))}, NA)
  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(values=c(0.01, 0.1)), bias=tune(), tuning=config_tuning(steps=2, CV = 2))}, NA)
  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(values=c(0.01, 0.1)), bias=tune(),activation=tune(values=c("selu","relu")), tuning=config_tuning(steps=2, CV = 2))}, NA)
  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(values=c(0.01, 0.1)), bias=tune(),activation=tune(),dropout=tune(), tuning=config_tuning(steps=2, CV = 2))}, NA)
  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(values=c(0.01, 0.1)), bias=tune(),activation=tune(),dropout=tune(c(0.2, 0.3)), tuning=config_tuning(steps=2, CV = 2))}, NA)

  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(values=c(0.01, 0.1)), bias=tune(),activation=tune(),dropout=tune(c(0.2, 0.3)), tuning=config_tuning(steps=2, CV = 2))}, NA)

  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",epochs=3,lr=tune(values=c(0.01, 0.1)), bias=tune(),activation=tune(),dropout=tune(c(0.2, 0.3)),
                              tuning=config_tuning(steps=2, CV = 2))}, NA)

  testthat::expect_error({dnn(Species~., data=iris,loss="softmax",lr=tune(values=c(0.01, 0.1)), epochs = tune(1, 4),bias=tune(),activation=tune(),dropout=tune(values=c(0.2, 0.3)),
                              tuning=config_tuning(steps=2, CV = 2))}, NA)
  testthat::expect_error({dnn(Species~., data=iris,loss="softmax", lr=tune(values=c(0.01, 0.1)), epochs=tune(1, 4),batchsize = tune(),bias=tune(),activation=tune(),dropout=tune(0.2, 0.3),
                              tuning=config_tuning(steps=2, CV = 2))}, NA)

  testthat::expect_error({dnn(Species~., data=iris,loss="softmax", lr=tune(values=c(0.01, 0.1)), epochs=tune(1, 4),batchsize = tune(values = c(10, 20)),bias=tune(),activation=tune(),dropout=tune(0.2, 0.3),
                              tuning=config_tuning(steps=2, CV = 2))}, NA)

  })

