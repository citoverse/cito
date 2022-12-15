source("utils.R")

testthat::test_that("summary", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)


  model = dnn(Sepal.Length~., data = datasets::iris, epoch = 2, verbose = FALSE)
  testthat::expect_error({summary(model)}, NA)

  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "softmax", lr = 0.1, verbose = FALSE)
  testthat::expect_error({summary(model)}, NA)

  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE)
  testthat::expect_error({summary(model)}, NA)

  iris2 = iris
  iris2 = iris2[iris2$Species %in% c("setosa", "versicolor"),]
  iris2$Species = as.integer(iris2$Species) - 1
  model = dnn(Species~., data = iris2, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE)
  testthat::expect_error({summary(model)}, NA)

})




testthat::test_that("PDP", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)


  model = dnn(Sepal.Length~., data = datasets::iris, epoch = 2, verbose = FALSE)

  # Build and train  Network
  testthat::expect_error({PDP(model)}, NA)
  testthat::expect_error({PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({PDP(model, ice = 10)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = 20)}, NA)

  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "softmax", lr = 0.1, verbose = FALSE)
  testthat::expect_error({PDP(model)}, NA)
  testthat::expect_error({PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({PDP(model, ice = TRUE)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  30)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  3)}, NA)


  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "binomial", lr = 0.1, verbose = FALSE)
  testthat::expect_error({PDP(model)}, NA)
  testthat::expect_error({PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({PDP(model, ice = TRUE)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  30)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  3)}, NA)


  iris2 = iris
  iris2 = iris2[iris2$Species %in% c("setosa", "versicolor"),]
  iris2$Species = as.integer(iris2$Species) - 1
  model = dnn(Species~., data = iris2, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE)
  testthat::expect_error({PDP(model)}, NA)
  testthat::expect_error({PDP(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({PDP(model, ice = TRUE)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  30)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"), ice = TRUE, resolution.ice =  3)}, NA)


})




testthat::test_that("ALE", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)

  model = dnn(Sepal.Length~., data = datasets::iris, epoch = 2, verbose = FALSE)

  # Build and train  Network
  testthat::expect_error({ALE(model)}, NA)
  testthat::expect_error({ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)


  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "softmax", lr = 0.1, verbose = FALSE)
  testthat::expect_error({ALE(model)}, NA)
  testthat::expect_error({ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)


  model = dnn(Species~., data = datasets::iris, epoch = 5, loss = "binomial", lr = 0.1, verbose = FALSE)
  testthat::expect_error({ALE(model)}, NA)
  testthat::expect_error({ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)


  iris2 = iris
  iris2 = iris2[iris2$Species %in% c("setosa", "versicolor"),]
  iris2$Species = as.integer(iris2$Species) - 1
  model = dnn(Species~., data = iris2, epoch = 5, loss = "binomial", lr = 0.1, verbose = TRUE)
  testthat::expect_error({ALE(model)}, NA)
  testthat::expect_error({ALE(model, variable = "Sepal.Width")}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({ALE(model, variable = c("Sepal.Width",  "Petal.Length"), K = 4)}, NA)
})
