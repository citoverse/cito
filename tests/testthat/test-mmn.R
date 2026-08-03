source("utils.R")
library(cito)
set.seed(42)

testthat::test_that("MMN architecture folder inputs png/jpeg and tiffs", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  if(torch::cuda_is_available()) {
    device <- "cuda"
  } else if(torch::backends_mps_is_available()) {
    device <- "mps"
  } else {
    device <- "cpu"
  }

  # create fake data
  dir.create("test_folder_jpeg")
  for(i in 1:100) {
    X = array(0.5, dim = c(50, 70, 3))
    jpeg::writeJPEG(X, paste0("test_folder_jpeg/",i, "_.jpeg"))
  }

  dir.create("test_folder_png")
  for(i in 1:100) {
    X = array(0.5, dim = c(50, 70, 3))
    png::writePNG(X, paste0("test_folder_png/",i, "_.png"))
  }

  dir.create("test_folder_tiff")
  for(i in 1:100) {
    X = array(runif(50*70*20, 0, 5), dim = c(50, 70, 20))
    raster::writeRaster(raster::brick(X), filename = paste0("test_folder_tiff/",i, "_.tiff"), overwrite=TRUE)
  }


  Y = runif(100)
  architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))
  wrap_cnn(list(X = "test_folder_jpeg", Y = Y, loss = "mse", epochs = 1L, architecture = architecture, optim = "adam"))

  m = cnn(X = "test_folder_jpeg", Y = Y, loss = "mse", epochs = 1L, architecture = architecture, optim = "adam")
  .p = predict(m, newdata = "test_folder_jpeg")
  X = array(0.5, dim = c(5, 3, 50, 70))
  .p = predict(m, newdata = X)
  .m = continue_training(m, X = X, Y = runif(5), epochs = 2L)
  .m = continue_training(m, X = "test_folder_jpeg", Y = Y, epochs = 2L)

  wrap_cnn(list(X = "test_folder_png", Y = Y, loss = "mse", epochs = 1L, architecture = architecture, optim = "adam"))
  m = cnn(X = "test_folder_png", Y = Y, loss = "mse", epochs = 1L, architecture = architecture, optim = "adam")
  .p = predict(m, newdata = "test_folder_png")
  X = array(0.5, dim = c(5, 3, 50, 70))
  .p = predict(m, newdata = X)
  .m = continue_training(m, X = X, Y = runif(5), epochs = 2L)
  .m = continue_training(m, X = "test_folder_png", Y = Y, epochs = 2L)

  wrap_cnn(list(X = "test_folder_tiff", Y = Y, loss = "mse", epochs = 1L, architecture = architecture, optim = "adam"))
  m = cnn(X = "test_folder_tiff", Y = Y, loss = "mse", epochs = 1L, architecture = architecture, optim = "adam")
  .p = predict(m, newdata = "test_folder_tiff")
  X = array(0.5, dim = c(5, 20, 50, 70))
  .p = predict(m, newdata = X)
  .m = continue_training(m, X = X, Y = runif(5), epochs = 2L)
  .m = continue_training(m, X = "test_folder_tiff", Y = Y, epochs = 2L)


  unlink("test_folder_jpeg", recursive = TRUE)
  unlink("test_folder_png", recursive = TRUE)
  unlink("test_folder_tiff", recursive = TRUE)
})


testthat::test_that("MMN with tabular and image data", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(42)
  n <- 60
  shapes <- simulate_shapes(n, 20)
  tab <- data.frame(X1 = rnorm(n), X2 = rnorm(n))
  dataList <- list(Y = rnorm(n), tab = tab, img = shapes$data)
  architecture <- create_architecture(conv(3), maxPool(), linear(5))

  wrap_mmn(list(formula = Y ~ dnn(~., data = tab) + cnn(X = img, architecture = architecture),
                dataList = dataList, loss = "mse", epochs = 2L, verbose = FALSE, plot = FALSE))

  m <- mmn(Y ~ dnn(~., data = tab) + cnn(X = img, architecture = architecture),
           dataList = dataList, loss = "mse", epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_s3_class(m, "citommn")
  testthat::expect_equal(nrow(predict(m)), n)

  # saveRDS/readRDS has to rebuild the net (invalid pointers)
  f <- tempfile(fileext = ".RDS")
  saveRDS(m, f)
  testthat::expect_error({.p <- predict(readRDS(f))}, NA)
})


testthat::test_that("MMN accepts custom submodules", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(42)
  n <- 60
  shapes <- simulate_shapes(n, 20)
  tab <- data.frame(X1 = rnorm(n), X2 = rnorm(n))
  dataList <- list(Y = rnorm(n), tab = tab, img = shapes$data)
  architecture <- create_architecture(conv(3), maxPool(), linear(5))

  custom_dnn <- torch::nn_module(initialize = function(input, output) {self$l <- torch::nn_linear(input, output)},
                                 forward = function(x) self$l(x))
  custom_cnn <- torch::nn_module(initialize = function(output) {self$l <- torch::nn_linear(400, output)},
                                 forward = function(x) self$l(torch::torch_flatten(x, start_dim = 2)))

  # custom submodules are supported, so they must not warn
  testthat::expect_no_warning({
    m <- mmn(Y ~ dnn(~., data = tab, custom = list(module = custom_dnn, output_dim = 4L)) +
                 cnn(X = img, architecture = architecture),
             dataList = dataList, loss = "mse", epochs = 1L, verbose = FALSE, plot = FALSE)
  })
  testthat::expect_equal(nrow(predict(m)), n)

  m <- mmn(Y ~ dnn(~., data = tab) + cnn(X = img, custom = list(module = custom_cnn, output_dim = 5L)),
           dataList = dataList, loss = "mse", epochs = 1L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(nrow(predict(m)), n)
})
