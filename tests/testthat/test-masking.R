source("utils.R")

sim_mask = function(n = 300, S = 5) {
  set.seed(4)
  env = data.frame(X.1 = rnorm(n), X.2 = rnorm(n), X.3 = rnorm(n))
  lat = rnorm(n)
  Y = sapply(seq_len(S), function(j) rbinom(n, 1, plogis(env$X.1 * (j - 3) / 2 + (if(j <= 2) 1.5 else -1.5) * lat)))
  colnames(Y) = paste0("sp", seq_len(S))
  list(env = env, Y = Y, y = env$X.1 + env$X.2 + rnorm(n, sd = 0.2))
}

test_that("dnn() with masked predictors keeps NA rows and works through the S3 surface", {
  skip_if_no_torch()
  d = sim_mask()
  dat = data.frame(y = d$y, d$env)
  dat$X.2[1:30] = NA
  wrap_dnn(list(formula = y ~ ., data = dat, masking = ~ X.2, epochs = 3L, plot = FALSE, verbose = FALSE))
  fit = dnn(y ~ ., data = dat, masking = ~ X.2 + X.3, epochs = 3L, plot = FALSE, verbose = FALSE)
  expect_equal(nrow(fit$data$X), nrow(dat))
  expect_equal(sum(is.nan(fit$data$X)), 30)
  expect_length(predict(fit, newdata = dat), nrow(dat))
  a = dat
  a$X.2 = NA
  expect_false(anyNA(predict(fit, newdata = a)))
  expect_false(isTRUE(all.equal(predict(fit, newdata = a), predict(fit, newdata = dat))))
  # the validation rows are masked too, so both losses measure the same quantity
  fit = dnn(y ~ ., data = dat, masking = ~ X.2 + X.3, epochs = 2L, validation = 0.2, plot = FALSE, verbose = FALSE)
  expect_equal(ncol(fit$data$val_hidden), 2)
  expect_true(any(fit$data$val_hidden[fit$data$validation, ]))
  fit = continue_training(fit, epochs = 1L, verbose = FALSE)
  expect_equal(sum(is.nan(fit$data$X)), 30)
  # an unknown row must not dilute the conditional effects
  r = fit$conditional_effects$link[[1]]$result
  expect_equal(dim(r)[1], 270)
  expect_true(all(is.finite(sapply(seq_len(dim(r)[2]), function(j) r[, j, j]))))
})

test_that("NA of masked columns stays with its row whatever the row names, and NA responses are not scored", {
  skip_if_no_torch()
  d = sim_mask()
  Y = d$Y
  rownames(Y) = paste0("site", seq_len(nrow(Y)))
  Y[5, 4] = NA
  env = d$env
  env$X.2[11:30] = NA
  env$X.3[1] = NA
  fit = dnn(X = env, Y = Y, loss = "bernoulli", masking = ~ . + X.2, epochs = 2L, validation = 0.2, plot = FALSE, verbose = FALSE)
  expect_equal(nrow(fit$data$X), 299)
  expect_equal(sum(is.nan(fit$data$X[, "X.2"])), 20)
  expect_true(all(is.nan(fit$data$X[10:29, "X.2"])))
  expect_true(is.finite(fit$losses$train_l[2]) && is.finite(fit$losses$valid_l[2]))
  dd = data.frame(d$Y, d$env)
  dd$sp1[1:10] = NA
  fit = dnn(cbind(sp1, sp2, sp3) ~ X.1 + X.2, data = dd, loss = "bernoulli", masking = ~ sp1 + sp2, epochs = 2L, plot = FALSE, verbose = FALSE)
  expect_equal(nrow(fit$data$X), 300)
  expect_true(is.finite(fit$losses$train_l[2]))
  expect_equal(nrow(predict(fit, newdata = dd)), 300)
})

test_that("a response given as input is scored only where hidden, and hidden values do not reach the prediction", {
  skip_if_no_torch()
  d = sim_mask()
  fit = dnn(X = d$env, Y = d$Y, loss = "bernoulli", masking = ~ ., epochs = 3L, validation = 0.2, plot = FALSE, verbose = FALSE)
  expect_equal(fit$model_properties$masking$resp, 1:5)
  x = torch::torch_tensor(cbind(as.matrix(d$env), d$Y)[1:50, ])
  fit$net$train()
  torch::torch_manual_seed(7)
  o1 = fit$net(x, scored = TRUE)
  h = as.matrix(o1$hidden)
  flipped = x$clone()
  idx = which(h, arr.ind = TRUE)
  for(k in seq_len(nrow(idx))) flipped[idx[k, 1], 3 + idx[k, 2]] = 1 - d$Y[idx[k, 1], idx[k, 2]]
  torch::torch_manual_seed(7)
  o2 = fit$net(flipped, scored = TRUE)
  fit$net$eval()
  expect_true(any(h) && !all(h))
  expect_equal(as.matrix(o1$hidden), as.matrix(o2$hidden))
  expect_equal(as.matrix(o1$pred), as.matrix(o2$pred), tolerance = 1e-6)
  # the fixed validation mask hides something in every validation row
  v = fit$data$validation
  expect_true(all(rowSums(fit$data$val_hidden[v, , drop = FALSE]) > 0))
  expect_false(any(fit$data$val_hidden[-v, ]))
})

test_that("predict() of a dnn() with the response as input conditions on Y", {
  skip_if_no_torch()
  d = sim_mask()
  fit = dnn(X = d$env, Y = d$Y, loss = "bernoulli", masking = ~ sp1 + sp2 + sp3, epochs = 3L, plot = FALSE, verbose = FALSE)
  expect_equal(fit$model_properties$masking$resp, 1:3)
  p0 = predict(fit)
  expect_equal(dim(p0), dim(d$Y))
  expect_equal(p0, predict(fit, Y = d$Y * NA))
  Yc = d$Y
  Yc[, 1] = NA
  p1 = predict(fit, Y = Yc[, c("sp2", "sp1")])
  expect_true(all(is.na(p1[, 2])))
  expect_false(anyNA(p1[, c(1, 3:5)]))
  expect_false(isTRUE(all.equal(p0[, 1], p1[, 1])))
  # responses not given as input are always predicted, given ones never
  p2 = predict(fit, Y = d$Y)
  expect_false(anyNA(p2[, 4:5]))
  expect_true(all(is.na(p2[, 1:3])))
  f = tempfile(fileext = ".rds")
  saveRDS(fit, f)
  expect_equal(predict(readRDS(f), Y = Yc), predict(fit, Y = Yc))
  expect_error(continue_training(fit, epochs = 1L, verbose = FALSE), NA)
  expect_error(summary(fit), NA)
  expect_error(predict(fit, Y = `colnames<-`(d$Y, paste0("x", 1:5))), "no response")
  expect_error(predict(fit, Y = d$Y[, 4:5]), "none of the responses")
})

test_that("dnn() masking refuses what it cannot do", {
  skip_if_no_torch()
  d = sim_mask()
  dat = data.frame(d$env, f = factor(sample(letters[1:3], 300, TRUE)), y = d$y)
  expect_error(dnn(y ~ ., data = dat, masking = ~ f, epochs = 1L, plot = FALSE, verbose = FALSE), "numeric")
  expect_error(dnn(y ~ ., data = dat, masking = ~ nope, epochs = 1L, plot = FALSE, verbose = FALSE), "neither")
  expect_error(dnn(y ~ ., data = dat, masking = ~ log(X.1), epochs = 1L, plot = FALSE, verbose = FALSE), "Only names")
  expect_error(dnn(y ~ X.1 + I(X.1^2) + X.2, data = dat, masking = ~ X.1, epochs = 1L, plot = FALSE, verbose = FALSE), "plain terms")
  expect_error(dnn(y ~ X.1 * X.2, data = dat, masking = ~ X.1, epochs = 1L, plot = FALSE, verbose = FALSE), "plain terms")
  expect_error(dnn(y ~ ., data = dat, masking = ~ ., epochs = 1L, plot = FALSE, verbose = FALSE), "several response")
  expect_error(dnn(cbind(sp1, sp2) ~ X.1, data = data.frame(d$Y, d$env), loss = "bernoulli", masking = ~ ., weights = rep(1, 300), epochs = 1L, plot = FALSE, verbose = FALSE), "weights")
  expect_error(dnn(X = d$env, Y = d$Y[, 1:2], loss = "cross-entropy", masking = ~ ., epochs = 1L, plot = FALSE, verbose = FALSE), "elementwise")
  expect_error(dnn(y ~ ., data = dat, masking = ~ X.1, bootstrap = 2L, epochs = 1L, plot = FALSE, verbose = FALSE), "bootstrap")
  dd = data.frame(d$Y, d$env)
  expect_error(dnn(cbind(sp1, sp2) ~ X.1 + sp1, data = dd, loss = "bernoulli", masking = ~ X.1, epochs = 1L, plot = FALSE, verbose = FALSE), "also predictors")
  fit = dnn(y ~ ., data = dat, epochs = 1L, plot = FALSE, verbose = FALSE)
  expect_error(predict(fit, Y = d$Y), "masking")
  expect_error(suppressWarnings(mmn(y ~ dnn(~ X.1, data = env, masking = ~ X.1), dataList = list(y = matrix(d$y), env = d$env), epochs = 1L, plot = FALSE, verbose = FALSE)), "inside mmn")
})

test_that("mmn() hides whole modalities, and NA rows of a masked modality are unknown", {
  skip_if_no_torch()
  d = sim_mask()
  img = array(rnorm(300 * 4 * 4), c(300, 1, 4, 4))
  b = d$env[, "X.3", drop = FALSE]
  b$X.3[1:20] = NA
  dl = list(y = matrix(d$y), env = d$env[, 1:2], b = b, img = img)
  arch = create_architecture(conv(2), linear(4))
  pars = list(formula = y ~ dnn(~ ., data = env) + dnn(~ X.3, data = b) + cnn(X = img, architecture = arch),
              dataList = dl, masking = ~ b + img, epochs = 2L, plot = FALSE, verbose = FALSE)
  fit = do.call(mmn, pars)
  expect_equal(fit$model_properties$masking$masked, c(FALSE, TRUE, TRUE))
  expect_equal(nrow(predict(fit)), 300)
  nd = dl
  nd$img[] = NA
  nd2 = nd
  nd2$b$X.3 = NA
  p1 = predict(fit, newdata = nd)
  p2 = predict(fit, newdata = nd2)
  expect_false(anyNA(p2))
  nd3 = nd2
  nd3$b$X.3 = 0
  nd3$b$X.3[1] = NA
  expect_equal(p2[1, ], predict(fit, newdata = nd3)[1, ])
  expect_false(isTRUE(all.equal(p1, p2)))
  expect_error(summary(fit, n_permute = 2L), NA)
  f = tempfile(fileext = ".rds")
  saveRDS(fit, f)
  expect_equal(predict(readRDS(f), newdata = nd2), p2)
  expect_error(continue_training(fit, epochs = 1L, verbose = FALSE), NA)
  pars$masking = ~ nope
  expect_error(do.call(mmn, pars), "names no modality")
})

test_that("summary() of a tnn() with tokens(Y) reports the conditional gain", {
  skip_if_no_torch()
  d = sim_mask(S = 10)
  fit = tnn(Y ~ dnn(~ ., data = env) + tokens(Y), dataList = list(Y = d$Y, env = d$env), loss = "bernoulli",
            epochs = 2L, validation = 0.2, plot = FALSE, verbose = FALSE)
  s = summary(fit, n_permute = 2L, n_masks = 2L, fractions = c(0.3, 0.7))
  expect_equal(s$gain$columns, c("all", "rarest 10%", "rarest 20%", "rarest 30%"))
  expect_equal(s$gain$n_columns, c(10, 1, 2, 3))
  expect_equal(nrow(s$curve), 2)
  expect_true(all(is.finite(s$gain$gain)))
  expect_match(s$gain_rows, "60 validation rows")
  expect_output(print(s), "revealed")
})
