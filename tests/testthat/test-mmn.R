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
