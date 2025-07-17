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

  #TODO: implement MMNs tests
})
