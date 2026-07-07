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
