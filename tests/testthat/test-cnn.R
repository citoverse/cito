source("utils.R")
library(cito)
set.seed(42)

architecture <- create_architecture(conv(), maxPool(), conv(), avgPool(), linear())

testthat::test_that("CNN dimensions", {
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

  set.seed(42)
  array_dims <- list(c(100,1,10),
                     c(100,1,10,10),
                     c(100,1,10,10,10),
                     c(100,3,10),
                     c(100,3,10,10),
                     c(100,3,10,10,10))
  X <- lapply(array_dims, function(x) array(runif(prod(x)), dim = x))
  Y <- factor(sample(c("a","b","c"), 100, replace=TRUE))

  for (x in X) {
    .n <- wrap_cnn(list(X=x, Y=Y, architecture=architecture, loss="cross-entropy", device=device, epochs=1, plot=FALSE, verbose=FALSE))
  }
})

testthat::test_that("CNN accuracy", {
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

  set.seed(42)
  n <- 1000
  shapes <- cito:::simulate_shapes(n, 50)
  test <- sample.int(n, 0.1*n)
  train <- c(1:n)[-test]
  cnn.fit <- cnn(X=shapes$data[train,,,,drop=F], Y=shapes$labels[train], architecture=architecture, loss="cross-entropy", device=device, validation=0.1, epochs=400, early_stopping = 20, lambda = 0.001, plot=FALSE, verbose=FALSE)
  pred <- predict(cnn.fit, newdata=shapes$data[test,,,,drop=F], type="class")
  true <- shapes$labels[test]
  accuracy <- length(which(pred==true))/length(test)
  testthat::expect_gt(accuracy, 0.95)
})

testthat::test_that("CNN transfer learning", {
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

  set.seed(42)
  shapes <- cito:::simulate_shapes(10, 100, 3)

  models <- list(
    "alexnet",
    "inception_v3",
    "mobilenet_v2",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext101_32x8d",
    "resnext50_32x4d",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "wide_resnet101_2",
    "wide_resnet50_2"
  )

  for (transfer_model in models) {
    architecture <- create_architecture(transfer(transfer_model, pretrained=FALSE))
    wrap_cnn(list(X=shapes$data, Y=shapes$labels, architecture=architecture, epochs=1, batchsize = 5L, loss="cross-entropy", plot=FALSE, verbose=FALSE, device=device))

    architecture <- create_architecture(transfer(transfer_model, pretrained=FALSE), linear(), linear())
    wrap_cnn(list(X=shapes$data, Y=shapes$labels, architecture=architecture, epochs=1, batchsize = 5L, loss="cross-entropy", plot=FALSE, verbose=FALSE, device=device))
  }
})

testthat::test_that("CNN pretrained", {
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

  architecture <- create_architecture(transfer("alexnet", pretrained=TRUE, freeze=TRUE))

  set.seed(42)
  n <- 500
  shapes <- cito:::simulate_shapes(n, 100, 3)
  test <- sample.int(n, 0.1*n)
  train <- c(1:n)[-test]
  cnn.fit <- cnn(X=shapes$data[train,,,,drop=F], Y=shapes$labels[train], architecture=architecture, loss="cross-entropy", device=device, epochs=10)
  pred <- predict(cnn.fit, newdata=shapes$data[test,,,,drop=F], type="class")
  true <- shapes$labels[test]
  accuracy <- length(which(pred==true))/length(test)
  testthat::expect_gt(accuracy, 0.95)
})




testthat::test_that("CNN folder inputs png/jpeg and tiffs", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

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
