source("utils.R")

testthat::test_that("summary", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)

  model = dnn(Sepal.Length~., data = datasets::iris, epoch = 2, verbose = FALSE)
  testthat::expect_error({summary(model)}, NA)

  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "cross-entropy", lr = 0.1, verbose = FALSE)
  testthat::expect_error({summary(model)}, NA)

  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE)
  testthat::expect_error({summary(model)}, NA)

  iris2 = iris
  iris2 = iris2[iris2$Species %in% c("setosa", "versicolor"),]
  #iris2$Species = as.integer(iris2$Species) - 1
  model = dnn(Species~., data = iris2, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE)
  testthat::expect_error({summary(model)}, NA)


  # a few special cases
  ## only one continuous predictor
  model = dnn(Sepal.Length~Petal.Length, data = datasets::iris, epoch = 2, verbose = FALSE)
  testthat::expect_error({summary(model)}, NA)

  model = dnn(Sepal.Length~Petal.Length, data = datasets::iris, epoch = 10, verbose = FALSE, bootstrap = 5)
  testthat::expect_error({summary(model)}, NA)

  ## only one factor
  model = dnn(Sepal.Length~Species, data = datasets::iris, epoch = 2, verbose = FALSE)
  testthat::expect_error({summary(model)}, NA)

  model = dnn(Sepal.Length~Species, data = datasets::iris, epoch = 10, verbose = FALSE, bootstrap = 5)
  testthat::expect_error({summary(model)}, NA)
})




testthat::test_that("PDP", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)


  model = dnn(Sepal.Length~., data = datasets::iris, epoch = 2, verbose = FALSE, plot = FALSE)

  # Build and train  Network
  suppressWarnings({

  testthat::expect_error({.n = PDP(model)}, NA)
  testthat::expect_error({.n = PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = PDP(model, ice = 10)}, NA)
  testthat::expect_error({.n = PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = 20)}, NA)

  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "cross-entropy", lr = 0.1, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n = PDP(model)}, NA)
  testthat::expect_error({.n = PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = PDP(model, ice = TRUE)}, NA)
  testthat::expect_error({.n = PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  30)}, NA)
  testthat::expect_error({.n = PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  3)}, NA)


  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "binomial", lr = 0.1, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n = PDP(model)}, NA)
  testthat::expect_error({.n = PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = PDP(model, ice = TRUE)}, NA)
  testthat::expect_error({.n = PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  30)}, NA)
  testthat::expect_error({.n = PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  3)}, NA)


  iris2 = iris
  iris2 = iris2[iris2$Species %in% c("setosa", "versicolor"),]
  #iris2$Species = as.integer(iris2$Species) - 1
  model = dnn(Species~., data = iris2, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE, plot = FALSE)
  testthat::expect_error({.n = PDP(model)}, NA)
  testthat::expect_error({.n = PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = PDP(model, ice = TRUE)}, NA)
  testthat::expect_error({.n = PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  30)}, NA)
  testthat::expect_error({.n = PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  3)}, NA)

  })
})




testthat::test_that("ALE", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)

  model = dnn(Sepal.Length~., data = datasets::iris, epoch = 2, verbose = FALSE, plot = FALSE)
  suppressWarnings({
  # Build and train  Network
  testthat::expect_error({.n = ALE(model)}, NA)
  testthat::expect_error({.n = ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)


  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "cross-entropy", lr = 0.1, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n = ALE(model)}, NA)
  testthat::expect_error({.n = ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)


  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "binomial", lr = 0.1, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n = ALE(model)}, NA)
  testthat::expect_error({.n = ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)

  iris2 = iris
  iris2 = iris2[iris2$Species %in% c("setosa", "versicolor"),]
  #iris2$Species = as.integer(iris2$Species) - 1
  model = dnn(Species~., data = iris2, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE, plot = FALSE)
  testthat::expect_error({.n = ALE(model)}, NA)
  testthat::expect_error({.n = ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({.n = ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)
  })
})






testthat::test_that("conditionalEffects finite and analytical return the same object structure", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)

  model = dnn(Sepal.Length~., data = datasets::iris, epochs = 5L, verbose = FALSE, plot = FALSE)

  ce_analytical = conditionalEffects(model, method = "analytical")
  ce_finite = conditionalEffects(model, method = "finite")

  # same structure
  testthat::expect_equal(length(ce_finite), length(ce_analytical))
  testthat::expect_equal(names(ce_finite[[1]]), names(ce_analytical[[1]]))
  testthat::expect_equal(dim(ce_finite[[1]]$result), dim(ce_analytical[[1]]$result))
  testthat::expect_equal(dimnames(ce_finite[[1]]$mean), dimnames(ce_analytical[[1]]$mean))

  # both methods should approximate the same effects
  testthat::expect_true(all(abs(diag(ce_finite[[1]]$mean) - diag(ce_analytical[[1]]$mean)) < 0.1))

  testthat::expect_error({.n = print(ce_finite)}, NA)
  testthat::expect_error({.n = print(ce_analytical)}, NA)

  # multi-output + interactions + subsample
  model = dnn(cbind(Sepal.Length, Sepal.Width)~., data = datasets::iris, epochs = 5L, verbose = FALSE, plot = FALSE)
  ce_analytical = conditionalEffects(model, method = "analytical", interactions = TRUE)
  ce_finite = conditionalEffects(model, method = "finite", interactions = TRUE, subsample = 0.5)
  testthat::expect_equal(length(ce_finite), length(ce_analytical))
  testthat::expect_equal(names(ce_finite[[1]]), names(ce_analytical[[1]]))
  testthat::expect_equal(dim(ce_finite[[1]]$result)[2:3], dim(ce_analytical[[1]]$result)[2:3])
})


testthat::test_that("xAI works when the model was fitted with ce = FALSE", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  model = dnn(Sepal.Length~., data = datasets::iris, epochs = 3L, ce = FALSE, verbose = FALSE, plot = FALSE)
  testthat::expect_null(model$conditional_effects)

  testthat::expect_error({.n = summary(model)}, NA)
  testthat::expect_error({.n = summary(model, importance = "ale")}, NA)
  testthat::expect_error({.n = summary(model, importance = "ce")}, NA)
  testthat::expect_error({.n = ALE(model, variable = "Petal.Length", plot = FALSE)}, NA)
  testthat::expect_error({.n = plot(model)}, NA)
})


testthat::test_that("analytical ALE uses the data it is given, not the training data", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  model = dnn(Sepal.Length~Petal.Length+Petal.Width, data = datasets::iris, epochs = 3L, verbose = FALSE, plot = FALSE)

  # fewer rows than the training data must not fail
  testthat::expect_error({.n = ALE(model, variable = "Petal.Length", data = datasets::iris[1:50,], plot = FALSE)}, NA)

  # the curve must follow the range of the supplied data
  shifted = datasets::iris
  shifted$Petal.Length = shifted$Petal.Length + 10
  a_train = ALE(model, variable = "Petal.Length", plot = FALSE)
  a_shift = ALE(model, variable = "Petal.Length", data = shifted, plot = FALSE)
  testthat::expect_true(min(a_shift[[1]]$data$x) > max(a_train[[1]]$data$x))
})


testthat::test_that("continue_training refreshes the stored conditional effects", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  model = dnn(Sepal.Length~., data = datasets::iris, epochs = 3L, verbose = FALSE, plot = FALSE)
  before = model$conditional_effects$response[[1]]$mean
  model = continue_training(model, epochs = 20L, verbose = FALSE, plot = FALSE)

  testthat::expect_false(identical(before, model$conditional_effects$response[[1]]$mean))
  fresh = conditionalEffects(model)[[1]]$mean
  testthat::expect_equal(model$conditional_effects$response[[1]]$mean, fresh)
})


testthat::test_that("continue_training on new data updates model$data", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(1)
  df <- data.frame(Y = rnorm(80), X1 = rnorm(80), X2 = rnorm(80))
  df2 <- data.frame(Y = rnorm(40), X1 = rnorm(40), X2 = rnorm(40))

  m <- dnn(Y ~ X1 + X2, data = df, epochs = 2L, verbose = FALSE, plot = FALSE)

  # continuing on the original data must not touch anything
  m_same <- continue_training(m, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_identical(m$data$data, m_same$data$data)
  testthat::expect_identical(m$data$X, m_same$data$X)

  # continuing on new data must make model$data describe the new data
  m_new <- continue_training(m, data = df2, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(nrow(m_new$data$data), nrow(df2))
  testthat::expect_equal(nrow(m_new$data$X), nrow(df2))
  testthat::expect_equal(as.numeric(m_new$data$Y), df2$Y, tolerance = 1e-5)
  testthat::expect_equal(nrow(predict(m_new)), nrow(df2))
  testthat::expect_equal(length(m_new$conditional_effects$row_indices), nrow(df2))
  testthat::expect_error({.n <- summary(m_new)}, NA)
  testthat::expect_error({.n <- ALE(m_new, variable = "X1", plot = FALSE)}, NA)

  # the contrasts of the original fit have to stay valid for newdata
  testthat::expect_identical(m$data$xlvls, m_new$data$xlvls)
})
