source("utils.R")
library(cito)
set.seed(42)



testthat::test_that("MMN architecture", {
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
  architecture1 <- create_architecture(conv(), maxPool(), conv(), avgPool(), linear())
  architecture2 <- create_architecture(conv(), conv(), maxPool(), conv(), conv(), avgPool(), linear(), linear())
  X_cnn1 <- array(runif(prod(c(100,3,10,10))), dim = c(100,3,10,10))
  X_cnn2 <- array(runif(prod(c(100,1,50,50))), dim = c(100,1,50,50))
  X_dnn1 <- matrix(runif(100*3), 100, 3)
  X_dnn2 <- factor(sample(c("a","b","c"), 100, replace=TRUE))
  Y <- factor(sample(c("d","e","f"), 100, replace=TRUE))


  .n <- wrap_mmn(list(formula=Y ~ cnn(X=X_cnn1, architecture=architecture1) + dnn(X=X_dnn1, hidden=c(30,30,30)) + cnn(X=X_cnn2, architecture=architecture2) + dnn(~X_dnn2), dataList=list(Y=y, X_cnn1=X_cnn1, X_cnn2=X_cnn2, X_dnn1=X_dnn1, X_dnn2=X_dnn2), loss="cross-entropy", device=device, epochs=1, plot=FALSE, verbose=FALSE))

})
