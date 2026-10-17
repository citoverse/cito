# Defects that were invisible in ordinary use and are pinned here so they stay fixed.
# They live together rather than in test-dnn/cnn/mmn because each one spans several model types.

source("utils.R")
library(cito)

testthat::test_that("a fitted model does not carry the training data", {
  skip_if_no_torch()
  saved_size = function(n) {
    X = array(runif(n*1*12*12), c(n, 1, 12, 12))
    m = cnn(X = X, Y = rnorm(n), architecture = create_architecture(conv(4), maxPool(), linear(6)),
            loss = "mse", epochs = 1L, verbose = FALSE, plot = FALSE)
    m$data = NULL
    f = tempfile()
    on.exit(unlink(f))
    saveRDS(m, f)
    file.size(f)
  }
  set.seed(1)
  small = saved_size(200)
  large = saved_size(1600)
  # the data grow eightfold; an unforced promise in get_loss() used to drag them into the object,
  # so the file grew with them. The weights are identical in size, so the file must not be
  testthat::expect_lt(large, 2*small)
})

testthat::test_that("a validation split still yields a usable loss curve", {
  skip_if_no_torch()
  set.seed(2)
  n = 300
  df = data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
  m = dnn(y ~ ., data = df, hidden = c(8L, 8L), epochs = 5L, validation = 0.3,
          verbose = FALSE, plot = FALSE)
  # the loop is wrapped in with_no_grad() for memory; the losses themselves must be unchanged
  testthat::expect_equal(sum(!is.na(m$losses$valid_l)), 5L)
  testthat::expect_true(all(is.finite(m$losses$valid_l)))
  testthat::expect_lt(m$losses$valid_l[5], m$losses$valid_l[1])
})

testthat::test_that("predict() on an mmn carries the sample names through", {
  skip_if_no_torch()
  set.seed(3)
  n = 40
  sh = cito:::simulate_shapes(n = n, size = 12, channels = 3)
  img = sh$data
  dimnames(img) = list(paste0("rec", seq_len(n)), NULL, NULL, NULL)
  tab = matrix(runif(n*2), n, 2, dimnames = list(paste0("rec", seq_len(n)), NULL))
  dat = list(Y = rnorm(n), img = img, tab = tab)
  m = mmn(Y ~ cnn(X = img, architecture = create_architecture(conv(4), maxPool(), linear(6))) +
              dnn(~., data = tab), dataList = dat, epochs = 2L, verbose = FALSE, plot = FALSE)
  p = predict(m, newdata = list(img = img, tab = tab))
  testthat::expect_equal(rownames(p), paste0("rec", seq_len(n)))
  # an input without names must not invent any
  testthat::expect_null(rownames(predict(m, newdata = list(img = unname(img), tab = unname(tab)))))
})
