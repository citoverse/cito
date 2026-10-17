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

testthat::test_that("modality importance recovers which data stream carries the signal", {
  skip_if_no_torch()
  set.seed(42)
  n <- 160
  shapes <- cito:::simulate_shapes(n = n, size = 16, channels = 3)
  X_dnn <- matrix(runif(n*3), n, 3)
  # Y is driven by the images and only weakly by one tabular column, so the ordering is known
  Y <- 3*(as.integer(shapes$labels) - 1) + 0.3*X_dnn[,1] + rnorm(n, sd = 0.2)
  data <- list(Y = Y, X_cnn = shapes$data, X_dnn = X_dnn)
  arch <- create_architecture(conv(6), maxPool(), linear(8))
  m <- mmn(Y ~ dnn(~., data = X_dnn, hidden = c(8, 8)) + cnn(X = X_cnn, architecture = arch),
           dataList = data, loss = "mse", epochs = 20L, lr = 0.02, verbose = FALSE, plot = FALSE)

  s <- summary(m, n_permute = 4L)
  testthat::expect_s3_class(s, "summary.citommn")
  imp <- s$importance
  testthat::expect_equal(nrow(imp), 2L)
  testthat::expect_true(all(c("modality", "loss_increase", "sd", "share") %in% names(imp)))
  # the image stream must come out ahead of the tabular one
  cnn_row <- grep("cnn", imp$modality)
  testthat::expect_gt(imp$loss_increase[cnn_row], imp$loss_increase[-cnn_row])
  testthat::expect_equal(sum(imp$share), 1, tolerance = 1e-8)
  testthat::expect_output(print(s), "Baseline loss")

  f <- tempfile(fileext = ".pdf")
  pdf(f); on.exit(unlink(f))
  testthat::expect_error({ back <- plot(m, n_permute = 2L) }, NA)
  testthat::expect_error(plot(m, plot_type = "arch"), NA)
  # the architecture plot draws the whole graph, so it must not need a convolutional submodel
  m_dnn <- mmn(Y ~ dnn(~., data = X_dnn, hidden = c(8, 8)), dataList = data, loss = "mse",
               epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_error(plot(m_dnn, plot_type = "arch"), NA)
  dev.off()
  testthat::expect_true(is.data.frame(back))
})

testthat::test_that("mmn() accepts weights and predict() checks the shape of newdata", {
  skip_if_no_torch()
  set.seed(5)
  n <- 60
  sh <- cito:::simulate_shapes(n = n, size = 16, channels = 3)
  Xd <- matrix(runif(n*2), n, 2)
  Y <- matrix(as.numeric(as.integer(sh$labels) == 1), ncol = 1)
  data <- list(Y = Y, X_cnn = sh$data, X_dnn = Xd)
  arch <- create_architecture(conv(4), maxPool(), linear(6))
  wloss <- function(pred, true, weights) weights * (pred - true)^2

  fit <- function(w) mmn(Y ~ dnn(~., data = X_dnn, hidden = c(6, 6)) + cnn(X = X_cnn, architecture = arch),
                         dataList = data, loss = wloss, weights = w, epochs = 5L,
                         verbose = FALSE, plot = FALSE)
  m <- fit(matrix(rep(1, n), ncol = 1))
  testthat::expect_true(m$training_properties$has_weights)

  # a fitted MMN is routinely stripped of its training data before saving, and predict() has to
  # survive that -- the shape check cannot demand a reference it no longer has
  stripped <- m
  stripped$data <- list(ylvls = NULL)
  testthat::expect_error(predict(stripped, newdata = list(X_cnn = sh$data, X_dnn = Xd)), NA)

  # predict() has to carry the sample names through: without them a downstream
  # metadata[rownames(pred), ] yields nothing and an accuracy comes back as NA, not as an error
  named_img <- sh$data
  dimnames(named_img) <- list(paste0("rec", seq_len(n)), NULL, NULL, NULL)
  named_tab <- Xd
  rownames(named_tab) <- paste0("rec", seq_len(n))
  p <- predict(m, newdata = list(X_cnn = named_img, X_dnn = named_tab))
  testthat::expect_equal(rownames(p), paste0("rec", seq_len(n)))
  testthat::expect_null(rownames(predict(m, newdata = list(X_cnn = sh$data, X_dnn = Xd))))
  # zeroing half the weights must change the loss, otherwise they never reach the likelihood
  half <- fit(matrix(rep(c(1, 0), each = n/2), ncol = 1))
  testthat::expect_lt(tail(half$losses$train_l, 1), tail(m$losses$train_l, 1))

  # a wrong image size used to surface as a torch matmul error several frames down
  wrong <- list(X_cnn = cito:::simulate_shapes(n = 5, size = 24, channels = 3)$data,
                X_dnn = Xd[1:5, , drop = FALSE])
  testthat::expect_error(predict(m, newdata = wrong), "3x16x16")
  # streams of different length used to be silently truncated to the shortest
  ragged <- list(X_cnn = sh$data[1:5, , , , drop = FALSE], X_dnn = Xd[1:3, , drop = FALSE])
  testthat::expect_error(predict(m, newdata = ragged), "different numbers of samples")

  ok <- list(X_cnn = sh$data[1:5, , , , drop = FALSE], X_dnn = Xd[1:5, , drop = FALSE])
  testthat::expect_equal(nrow(predict(m, newdata = ok)), 5L)
})
