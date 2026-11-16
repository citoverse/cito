# Input masking for dnn() and mmn(): masked inputs are hidden at random in training, with one rate
# p ~ U(0,1) drawn per row and minibatch, and enter as unknown (0 plus an indicator), as do NA values.

masking_names = function(f) {
  if(!inherits(f, "formula") || length(f) != 2) stop("'masking' is a one-sided formula, e.g. ~ temp + prec.", call. = FALSE)
  vapply(flatten_sum(f[[2]]), function(t) {
    if(!is.name(t)) stop("Only names, '.' and '+' are allowed in 'masking', not '", deparse1(t), "'.", call. = FALSE)
    as.character(t)
  }, character(1))
}

# Masked columns before get_X_Y(): predictors, and response columns given as input ('.' = all of them).
masked_vars = function(masking, responses) {
  nm = masking_names(masking)
  list(pred = setdiff(nm, c(".", responses)), resp = if("." %in% nm) responses else intersect(nm, responses))
}

# NA of a masked column would make model.frame() drop the row, so it passes get_X_Y() as Inf and becomes
# NaN (unknown) in the model matrix and NA again in the stored data, independent of row names.
to_sentinel = function(d, vars) {
  vars = intersect(vars, colnames(d))
  if(length(vars) == 0) return(d)
  if(!is.matrix(d)) d = data.frame(d, check.names = FALSE)
  for(v in vars) {
    if(all(is.na(d[, v]))) d[, v] = NA_real_
    if(!is.numeric(d[, v])) stop("Masked columns must be numeric: ", v, call. = FALSE)
    d[is.na(d[, v]), v] = Inf
  }
  d
}

from_sentinel = function(d, vars, value = NA) {
  for(v in intersect(vars, colnames(d))) d[is.infinite(d[, v]), v] = value
  d
}

response_names = function(formula, Y) {
  if(!is.null(Y)) return(if(!is.null(dim(Y)) && !is.null(colnames(Y))) colnames(Y) else if(!is.null(dim(Y))) paste0("Y", seq_len(NCOL(Y))) else "Y")
  if(is.null(formula) || length(formula) < 3) return(character(0))
  all.vars(formula[[2]])
}

get_X_Y_masked = function(formula, X, Y, data, vars) {
  if(is.null(vars)) return(get_X_Y(formula, X, Y, data))
  if(!is.null(X)) {
    X = to_sentinel(X, vars$pred)
    if(!is.null(Y) && length(vars$resp) > 0) {
      if(is.null(dim(Y))) Y = matrix(Y, ncol = 1L, dimnames = list(NULL, "Y"))
      if(is.null(colnames(Y))) colnames(Y) = paste0("Y", seq_len(ncol(Y)))
      Y = to_sentinel(as.matrix(Y), vars$resp)
    }
  } else {
    data = to_sentinel(data, c(vars$pred, vars$resp))
  }
  out = get_X_Y(formula, X, Y, data)
  out$X = from_sentinel(out$X, vars$pred, NaN)
  if(any(is.infinite(out$X)))
    stop("A masked predictor also enters a transformed term or an interaction; mask only predictors that enter as plain terms.", call. = FALSE)
  if(!is.null(out$Y) && !is.null(dim(out$Y))) out$Y = from_sentinel(out$Y, vars$resp)
  if(!is.null(out$data)) out$data = from_sentinel(out$data, c(vars$pred, vars$resp))
  out
}

dnn_masking = function(vars, masking, X, Y, formula) {
  resp = if(!is.null(dim(Y)) && !is.null(colnames(Y))) colnames(Y) else all.vars(formula[[2]])
  overlap = intersect(colnames(X), resp)
  if(length(overlap) > 0) stop("Response columns are also predictors: ", paste(overlap, collapse = ", "),
                               ". To give the response as input, name it in 'masking' only.", call. = FALSE)
  r = match(vars$resp, resp)
  if(length(r) > 0 && length(resp) < 2)
    stop("The response can be given as input only with several response columns; '.' in 'masking' names the response columns.", call. = FALSE)
  bad = setdiff(vars$pred, colnames(X))
  if(length(bad) > 0) stop("'masking' names neither a numeric predictor nor a response column: ", paste(bad, collapse = ", "),
                           ". Factors and transformed terms cannot be masked.", call. = FALSE)
  fac = attr(stats::terms(formula), "factors")
  for(v in vars$pred) {
    other = setdiff(rownames(fac), v)
    inside = other[vapply(other, function(e) v %in% all.vars(str2lang(e)), logical(1))]
    if(length(inside) > 0) stop("The masked predictor '", v, "' also enters ", paste(inside, collapse = ", "),
                                "; mask only predictors that enter as plain terms.", call. = FALSE)
    used = if(v %in% rownames(fac)) colnames(fac)[fac[v, ] > 0] else v
    if(!identical(used, v)) stop("The masked predictor '", v, "' also enters ", paste(setdiff(used, v), collapse = ", "),
                                 "; mask only predictors that enter as plain terms.", call. = FALSE)
  }
  list(vars = vars, names = vars$pred, responses = resp, pred = match(vars$pred, colnames(X)), resp = sort(r), n_x = ncol(X), n_y = length(resp),
       rule = deparse1(masking))
}

# The loader input: predictors, then the response columns given as input, NaN where the fixed validation
# mask hides them (columns of val_hidden: masked predictors, then response inputs).
masked_dnn_X = function(X, Y, masking, val_hidden) {
  k = length(masking$pred)
  if(k > 0) {
    P = X[, masking$pred, drop = FALSE]
    P[val_hidden[, seq_len(k), drop = FALSE]] = NaN
    X[, masking$pred] = P
  }
  if(length(masking$resp) == 0) return(X)
  R = as.matrix(Y)[, masking$resp, drop = FALSE]
  R[val_hidden[, k + seq_along(masking$resp), drop = FALSE]] = NaN
  colnames(R) = NULL
  cbind(X, R)
}

# conditioning values for the response inputs, NaN where not given
response_input = function(Y, columns, responses, n) {
  Y = as.matrix(Y)
  if(nrow(Y) != n) stop("'Y' has ", nrow(Y), " rows, the data ", n, ".", call. = FALSE)
  if(is.null(colnames(Y))) {
    if(ncol(Y) != length(columns)) stop("'Y' without column names needs one column per response given as input.", call. = FALSE)
    colnames(Y) = columns
  }
  unknown = setdiff(colnames(Y), responses)
  if(length(unknown) > 0) stop("'Y' has columns that are no response of the model: ", paste(utils::head(unknown, 5), collapse = ", "), call. = FALSE)
  have = intersect(columns, colnames(Y))
  if(length(have) == 0) stop("'Y' holds none of the responses given as input: ", paste(utils::head(columns, 5), collapse = ", "), call. = FALSE)
  out = matrix(NaN, n, length(columns))
  out[, match(have, columns)] = Y[, have]
  out[is.na(out)] = NaN
  storage.mode(out) = "double"
  out
}

# NA remain only in response columns given as input (rows with other NA are dropped): NaN there,
# unknown in the input and never scored
format_Y_na = function(loss, Y) {
  if(!anyNA(Y)) return(loss$format_Y(Y))
  Yt = loss$format_Y(replace(Y, is.na(Y), 0))
  Yt[torch::torch_tensor(is.na(as.matrix(Y)))] = NaN
  Yt
}

dnn_validation_mask = function(X, Y, valid, masking) {
  C = cbind(X[, masking$pred, drop = FALSE], as.matrix(Y)[, masking$resp, drop = FALSE])
  k = ncol(C)
  if(k == 0 || is.null(valid)) return(list(mask = matrix(FALSE, nrow(C), k), valid = valid))
  draw_validation_mask(C, valid, list(always = rep(FALSE, k), random = rep(TRUE, k)))
}

set_dnn_eval_scale = function(object) {
  v = object$data$validation
  m = object$model_properties$masking
  vh = object$data$val_hidden[, length(m$pred) + seq_along(m$resp), drop = FALSE]
  object$loss$eval_scale = if(is.null(v)) NULL else
    c(batches = ceiling(length(v) / object$training_properties$batchsize),
      cells = sum(vh[v, ]) + length(v) * (m$n_y - length(m$resp)))
  invisible(object)
}

# Hides masked inputs at random in training; hidden and NaN inputs enter as 0 plus an unknown
# indicator. Response inputs absent from x (predict without Y, xAI) are unknown.
MaskedDNN = torch::nn_module(
  "DNN",
  initialize = function(inner, masking) {
    self$inner = inner
    self$has_embeddings = isTRUE(inner$has_embeddings)
    self$n_x = masking$n_x
    self$n_resp = length(masking$resp)
    self$k_pred = length(masking$pred)
    self$cols = c(masking$pred, masking$n_x + seq_along(masking$resp))
    self$keep = setdiff(seq_len(masking$n_x), masking$pred)
    map = rep(1L, masking$n_y)
    map[masking$resp] = 1L + self$k_pred + seq_along(masking$resp)
    self$resp_map = map
  },
  forward = function(x, z = NULL, return_embeddings = FALSE, scored = FALSE) {
    B = x$size(1)
    idx = function(i) torch::torch_tensor(i, dtype = torch::torch_long(), device = x$device)
    if(self$n_resp > 0 && x$size(2) == self$n_x)
      x = torch::torch_cat(list(x, torch::torch_full(c(B, self$n_resp), NaN, dtype = x$dtype, device = x$device)), 2L)
    v = x$index_select(2L, idx(self$cols))
    unknown = torch::torch_isnan(v)
    if(self$training) {
      p = torch::torch_rand(c(B, 1L), device = x$device)
      unknown = unknown$logical_or(torch::torch_rand(v$shape, device = x$device) < p)
    }
    v = v$nan_to_num(0)$masked_fill(unknown, 0)
    h = torch::torch_cat(list(x$index_select(2L, idx(self$keep)), v, unknown$to(dtype = x$dtype)), 2L)
    out = if(self$has_embeddings) self$inner(h, z, return_embeddings = return_embeddings) else self$inner(h, return_embeddings = return_embeddings)
    if(!scored || self$n_resp == 0) return(out)
    s = torch::torch_cat(list(torch::torch_ones(c(B, 1L), dtype = torch::torch_bool(), device = x$device), unknown), 2L)
    list(pred = out, hidden = s$index_select(2L, idx(self$resp_map)))
  }
)

# mmn(): a row of a masked modality is unknown when it contains NA, or when hidden in training.
mmn_masking = function(masking, rhs) {
  if(is.null(masking)) return(NULL)
  names = dedupe_names(vapply(flatten_sum(rhs), modality_name, character(1)))
  nm = masking_names(masking)
  bad = setdiff(nm, names)
  if(length(bad) > 0) stop("'masking' names no modality: ", paste(bad, collapse = ", "), ". Modalities are named by their data: ",
                           paste(names, collapse = ", "), call. = FALSE)
  list(masked = names %in% nm, names = names, rule = deparse1(masking))
}

mmn_masked = function(object) {
  m = if(inherits(object, "citommn")) object$model_properties$masking else object
  if(is.null(m)) NULL else m$masked
}

# One hidden/shown draw per validation row and masked modality, from the training scheme. Applied to
# tensor inputs only; lazily loaded modalities are not masked in validation.
mmn_validation_mask = function(n, valid, masked) {
  m = matrix(FALSE, n, length(masked))
  if(is.null(valid) || !any(masked)) return(m)
  p = stats::runif(length(valid))
  m[valid, masked] = matrix(stats::runif(length(valid) * sum(masked)), length(valid)) < p
  m
}

mmn_apply_mask = function(X, hidden) {
  if(is.null(hidden) || !any(hidden)) return(X)
  slots = which(!vapply(X, function(x) inherits(x, "torch_tensor") && as.character(x$dtype) == "Long", logical(1)))
  for(j in which(colSums(hidden) > 0)) {
    x = X[[slots[j]]]
    if(!inherits(x, "torch_tensor") || !x$is_floating_point()) next
    X[[slots[j]]] = x$clone()$index_fill_(1L, torch::torch_tensor(which(hidden[, j]), dtype = torch::torch_long()), NaN)
  }
  X
}

print_masking = function(m) {
  if(is.null(m)) return(invisible(NULL))
  cat("Masking ", m$rule, ": hidden at random in training", if(!is.null(m$vars) && length(m$vars$resp) > 0)
      paste0("; responses given as input: ", paste(m$vars$resp, collapse = ", ")), "\n", sep = "")
}

# PDP and ALE build their grid from the stored data; rows with an unknown masked predictor have no value there
xai_data = function(model) {
  d = model$data$data
  v = intersect(model$model_properties$masking$names, colnames(d))
  if(length(v) > 0) d = d[stats::complete.cases(d[, v, drop = FALSE]), , drop = FALSE]
  d
}
