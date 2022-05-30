library(cito)
testthat::test_that("cito examples", {
  testthat::skip_on_cran()

  path = system.file("examples", package = "cito")
  to_do = list.files(path, full.names = TRUE)
  run_raw = function(rr) suppressWarnings(eval(str2expression(rr[c(-1, -length(rr))])))
  for(i in 1:length(to_do)) {
    raw = readLines(to_do[i])
    testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
  }
})
