source("utils.R")

sim_tnn = function(n = 200, S = 6, na = 40) {
  set.seed(3)
  env = data.frame(a = rnorm(n), b = rnorm(n))
  Y = sapply(seq_len(S), function(j) rbinom(n, 1, plogis(env$a * (j - S / 2) / 2)))
  colnames(Y) = paste0("sp", seq_len(S))
  Y[sample(length(Y), na)] = NA
  list(Y = Y, env = env, y = matrix(env$a + rnorm(n, sd = 0.3)), tr = matrix(rnorm(n * 3), n))
}

fit_tokens = function(d, ...) tnn(Y ~ dnn(~ a + b, data = env) + tokens(Y), dataList = d, loss = "bernoulli",
                                  plot = FALSE, verbose = FALSE, ...)

# forward in evaluation mode with an injected hidden mask
forward_eval = function(fit, d, hidden, values = d$Y) {
  d$Y = values
  inp = cito:::tnn_inputs(fit, d, hidden)
  fit$net$eval()
  torch::with_no_grad(fit$net(inp$inputs))
}

test_that("the attention compiler builds the documented masks", {
  tok = data.frame(group = c("env", "Y", "Y", "Y", "register"), name = c("env", "sp1", "sp2", "sp3", "register"),
                   kind = c("modality", "column", "column", "column", "register"))
  k = cito:::compile_attention(NULL, tok, NULL, response = "Y")
  expect_equal(unname(k["env", ]), c(TRUE, FALSE, FALSE, FALSE, TRUE))
  expect_equal(unname(diag(k)), c(TRUE, FALSE, FALSE, FALSE, TRUE))
  expect_true(all(k["register", ]))
  k = cito:::compile_attention(list(env ~ ., birds ~ . - sp3, sp1 ~ env), tok, list(birds = c("sp1", "sp2")), response = "Y")
  expect_true(all(k["env", ]))
  expect_equal(unname(k["sp1", ]), c(TRUE, FALSE, FALSE, FALSE, FALSE))
  expect_equal(unname(k["sp2", ]), c(TRUE, TRUE, FALSE, FALSE, TRUE))
  expect_error(cito:::compile_attention(list(env ~ 0), tok, NULL, "Y"), "never unknown")
  expect_error(cito:::compile_attention(list(sp1 ~ Y), tok, NULL, "Y"), "never unknown")
  expect_error(cito:::compile_attention(list(frogs ~ .), tok, NULL, "Y"), "Unknown token")
  m = cito:::compile_masking(sp1 ~ sp2, tok[tok$group == "Y", ], NULL)
  expect_equal(m$always, c(TRUE, FALSE, FALSE))
  expect_equal(m$random, c(FALSE, FALSE, FALSE))
  expect_equal(m$never, c(FALSE, FALSE, TRUE))
  m = cito:::compile_masking(sp1 ~ . + sp1, tok[tok$group == "Y", ], NULL)
  expect_equal(m$random, c(TRUE, FALSE, FALSE))
  expect_equal(m$never, c(FALSE, FALSE, FALSE))
})

test_that("a hidden value never reaches any prediction, a revealed one does", {
  skip_if_no_torch()
  d = sim_tnn()
  for(att in list(NULL, list(. ~ .))) {
    fit = fit_tokens(d, epochs = 2L, attention = att)
    hidden = matrix(FALSE, nrow(d$Y), ncol(d$Y))
    hidden[, 2] = TRUE
    obs = !is.na(d$Y[, 2])
    flipped = d$Y
    flipped[, 2] = 1 - flipped[, 2]
    a = forward_eval(fit, d, hidden)
    b = forward_eval(fit, d, hidden, flipped)
    expect_equal(as.matrix(a$pred), as.matrix(b$pred))
    expect_equal(unname(as.matrix(a$hidden)), unname(hidden & !is.na(d$Y)))
    # positive control: the same flip with column 2 revealed moves the other columns
    c0 = forward_eval(fit, d, hidden & FALSE)
    c1 = forward_eval(fit, d, hidden & FALSE, flipped)
    expect_gt(max(abs(as.matrix(c0$pred)[obs, -2] - as.matrix(c1$pred)[obs, -2])), 1e-4)
  }
})

test_that("the masked loss scores exactly the hidden cells", {
  skip_if_no_torch()
  d = sim_tnn()
  fit = fit_tokens(d, epochs = 1L)
  hidden = matrix(runif(length(d$Y)) < 0.3, nrow(d$Y))
  out = forward_eval(fit, d, hidden)
  fit$loss$eval_scale = NULL
  target = d$Y
  target[is.na(target)] = 0
  l = fit$loss(out, torch::torch_tensor(target))
  expect_equal(l$size(1), sum(hidden & !is.na(d$Y)))
  fit$net$train()
  inp = cito:::tnn_inputs(fit, d)
  o = fit$net(inp$inputs)
  expect_false(any(as.matrix(o$hidden) & is.na(d$Y)))
})

test_that("gradients stay finite with NA in the response and in tokens()", {
  skip_if_no_torch()
  d = sim_tnn()
  d$tr[sample(length(d$tr), 30)] = NA
  fit = tnn(Y ~ dnn(~ a, data = env) + tokens(tr) + tokens(Y), dataList = d, loss = "bernoulli", epochs = 1L,
            plot = FALSE, verbose = FALSE)
  fit$net$train()
  inp = cito:::tnn_inputs(fit, d)
  target = d$Y
  target[is.na(target)] = 0
  fit$net$zero_grad()
  fit$loss$eval_scale = NULL
  fit$loss(fit$net(inp$inputs), torch::torch_tensor(target))$mean()$backward()
  g = unlist(lapply(fit$net$parameters, function(p) if(is.null(p$grad) || !p$grad$numel()) numeric(0) else as.numeric(p$grad$cpu())))
  expect_true(all(is.finite(g)))
})

test_that("the validation loss is the pooled loss of the hidden cells, for any batch size", {
  skip_if_no_torch()
  d = sim_tnn()
  for(bs in c(7L, 50L)) {
    fit = fit_tokens(d, epochs = 1L, validation = 0.3, batchsize = bs, dropout = 0)
    v = fit$data$validation
    hidden = fit$data$val_hidden
    out = forward_eval(fit, d, hidden)
    target = d$Y
    target[is.na(target)] = 0
    fit$loss$eval_scale = NULL
    li = as.matrix(fit$loss$inner(out$pred, torch::torch_tensor(target)))
    expect_equal(fit$losses$valid_l[1], mean(li[hidden]), tolerance = 1e-5)
  }
})

test_that("keyless tokens and misuse are refused", {
  skip_if_no_torch()
  d = sim_tnn()
  expect_error(tnn(Y ~ tokens(Y), dataList = d, loss = "bernoulli", register = FALSE, epochs = 1L, plot = FALSE, verbose = FALSE), "never unknown")
  expect_error(tnn(Y ~ tokens(Y), dataList = d, loss = "multinomial", epochs = 1L, plot = FALSE, verbose = FALSE), "elementwise")
  expect_error(tnn(y ~ dnn(~ a, data = env), dataList = d, masking = . ~ ., epochs = 1L, plot = FALSE, verbose = FALSE), "masking")
  expect_error(mmn(y ~ dnn(~ a, data = env) + tokens(tr), dataList = d, epochs = 1L, plot = FALSE, verbose = FALSE))
})

test_that("token mode predicts, queries, saves and continues", {
  skip_if_no_torch()
  d = sim_tnn()
  fit = fit_tokens(d, epochs = 3L, validation = 0.2, groups = list(left = c("sp1", "sp2", "sp3")))
  p = predict(fit, type = "response")
  expect_equal(dim(p), dim(d$Y))
  expect_true(all(p > 0 & p < 1))
  q = predict(fit, Y = d$Y, query = left ~ .)
  expect_true(all(is.na(q[, 4:6])))
  expect_false(anyNA(q[, 1:3]))
  expect_error(predict(fit, Y = d$Y, query = left ~ left), "both sides")
  l = predict(fit, Y = d$Y[1:5, ], newdata = list(env = d$env[1:5, ]), loo = TRUE)
  expect_false(anyNA(l[, colSums(!is.na(d$Y[1:5, ])) > 0]))
  c = predict(fit, Y = d$Y[1:5, ], newdata = list(env = d$env[1:5, ]))
  expect_equal(unname(is.na(c)), unname(!is.na(d$Y[1:5, ])))
  f = tempfile(fileext = ".rds")
  saveRDS(fit, f)
  expect_equal(predict(readRDS(f), type = "response"), p, tolerance = 1e-6)
  fit2 = continue_training(fit, epochs = 2L, verbose = FALSE, plot = FALSE)
  expect_equal(nrow(fit2$losses), 5L)

  g = tnn(Y ~ dnn(~ a + b, data = env) + tokens(Y), dataList = list(Y = d$tr, env = d$env), loss = "gaussian",
          epochs = 2L, plot = FALSE, verbose = FALSE)
  saveRDS(g, f)
  expect_equal(as.numeric(readRDS(f) |> cito:::check_model() |> (\(m) m$loss$inner$parameter)()),
               as.numeric(g$loss$inner$parameter))
  cnt = matrix(rpois(600, 50), 200)
  h = tnn(Y ~ dnn(~ a, data = env) + tokens(Y, transform = log1p), dataList = list(Y = cnt, env = d$env), loss = "poisson",
          epochs = 2L, plot = FALSE, verbose = FALSE, masking = . ~ 0)
  expect_true(all(is.finite(predict(h, type = "response"))))
})

test_that("fusion mode works with the losses and the S3 surface of mmn", {
  skip_if_no_torch()
  d = sim_tnn()
  arch = create_architecture(conv(2), maxPool(), linear(4))
  d$img = array(rnorm(200 * 64), c(200, 1, 8, 8))
  fit = tnn(y ~ dnn(~ a + b, data = env) + cnn(X = img, architecture = arch) + tokens(tr), dataList = d,
            epochs = 3L, validation = 0.2, plot = FALSE, verbose = FALSE)
  expect_equal(dim(predict(fit)), c(200L, 1L))
  expect_equal(dim(predict(fit, newdata = list(env = d$env[1:3, ], img = d$img[1:3, , , , drop = FALSE], tr = d$tr[1:3, ]))), c(3L, 1L))
  expect_equal(ncol(predict(fit, return_embeddings = TRUE)), 32L)
  cls = factor(sample(c("a", "b", "c"), 200, TRUE))
  fit = tnn(cls ~ dnn(~ a, data = env) + dnn(~ b, data = env), dataList = list(cls = cls, env = d$env),
            loss = "cross-entropy", epochs = 2L, plot = FALSE, verbose = FALSE)
  expect_s3_class(predict(fit, type = "class"), "factor")
  wl = function(pred, true, weights) (pred - true)^2 * weights
  fit = tnn(y ~ dnn(~ a, data = env), dataList = d, loss = wl, weights = runif(200), epochs = 2L, plot = FALSE, verbose = FALSE)
  f = tempfile(fileext = ".rds")
  saveRDS(fit, f)
  expect_equal(predict(readRDS(f)), predict(fit), tolerance = 1e-6)
  expect_error(continue_training(fit, epochs = 1L, verbose = FALSE, plot = FALSE), NA)
})

