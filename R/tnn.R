#' Train a transformer neural network (TNN)
#'
#' The transformer counterpart of \code{\link{mmn}}. Every \code{dnn()} and \code{cnn()} term of the
#' formula is encoded to one token, every column of a matrix written as \code{tokens(M)} becomes one
#' token, and the tokens are fused by a transformer encoder instead of the fusion DNN of \code{mmn()}.
#' Which token may read which is set with formulas (\code{attention}). When the response itself enters
#' as \code{tokens(Y)}, the model is trained with a masked objective and can be asked conditional
#' questions, e.g. "the birds given the plants", see \code{\link{predict.citotnn}}.
#'
#' @param formula A formula. Left of \code{~} the response, a name in \code{dataList}. Right of \code{~}
#'   a sum of \code{dnn()} and \code{cnn()} terms, written as in \code{\link{mmn}}, and of
#'   \code{tokens(M)} terms, where \code{M} names a numeric matrix or data.frame in \code{dataList} whose
#'   columns become one token each (\code{NA} = unknown). \code{tokens(M, transform = log1p)} transforms
#'   the values before they enter the tokens; the response used as a target is never transformed.
#' @param dataList A named list holding every object named in the formula.
#' @param emb_dim Width of the tokens.
#' @param heads Number of attention heads; \code{emb_dim} must be a multiple of it.
#' @param layers Number of transformer layers.
#' @param ff_dim Width of the feed-forward block of each layer.
#' @param dropout Dropout in the transformer layers.
#' @param register Logical, add a learned register token. \code{NULL} resolves to \code{TRUE} when the
#'   response is tokenized and to \code{FALSE} otherwise.
#' @param attention \code{NULL} or a list of formulas \code{readers ~ readable}, see Details.
#' @param masking \code{NULL} or one formula \code{hidden ~ given} over the response columns that sets
#'   which cells are hidden in training, see Details. \code{NULL} is \code{. ~ .}.
#' @param groups A named list of character vectors; each name can be used in \code{attention},
#'   \code{masking} and in the queries of \code{predict()} for the tokens (or token groups) it lists.
#' @param loss,custom_parameters,optimizer,lr,lr_scheduler,alpha,lambda,validation,batchsize,shuffle,data_augmentation,epochs,weights,early_stopping,burnin,baseloss,device,plot,verbose
#'   As in \code{\link{mmn}}. With a tokenized response only elementwise losses are possible
#'   (\code{"mse"}, \code{"mae"}, \code{"gaussian"}, \code{"bernoulli"}, \code{"poisson"},
#'   \code{"nbinom"} or a function), and \code{weights} are not supported.
#'
#' @details
#' \strong{Tokens.} Token order is the \code{dnn()}/\code{cnn()} terms, the \code{tokens()} columns, the
#' register and, if the response is not tokenized, a \code{cls} token. A column token is a learned column
#' embedding plus \eqn{v w + b} for an observed value \eqn{v} or a learned \emph{unknown} vector, with
#' \eqn{w, b} and the unknown vector shared within a \code{tokens()} term. Tokens form a set: there are
#' no positional encodings. Every token is named: a \code{dnn()}/\code{cnn()} token by its data
#' (\code{data =} or \code{X =}; repeated names get \code{.2}, \code{.3}), a \code{tokens(M)} group by
#' \code{M} and each of its tokens by its column name, plus \code{register} and \code{cls}.
#'
#' \strong{Readout.} With \code{tokens(Y)} for the response \code{Y}, column \eqn{j} is predicted by a
#' column-specific linear readout of token \eqn{j}. Otherwise the response is read out from the
#' \code{cls} token by a linear layer, and every loss of \code{mmn()} is available.
#'
#' \strong{Attention.} By default every token reads every token, except that with a tokenized response
#' each \code{dnn()}/\code{cnn()} token reads only itself and the register (the configuration of the
#' community transformer). A rule \code{readers ~ readable} replaces what the readers may read; rules are
#' applied in order, so a later rule wins. Sets are written with the token names above, names of
#' \code{groups}, \code{.} (all tokens), \code{0} (none), \code{+} (union) and \code{-} (removal):
#' \code{list(env ~ env + register, birds ~ . - plants)}. Two rules are fixed: a response token never
#' reads itself, and a token whose value is unknown or hidden is never read. Every token has to be able
#' to read at least one token that is never unknown (a \code{dnn()}/\code{cnn()} token, the register or
#' \code{cls}), otherwise a row whose other keys are all unknown would have nothing to attend to.
#'
#' \strong{Masked objective.} With a tokenized response, the cells hidden in a minibatch enter as
#' unknown and only they are scored; \code{NA} cells are never observed and never scored. \code{masking
#' = hidden ~ given} sorts the response columns: columns only on the left are always hidden, columns
#' only on the right are never hidden, columns on both sides are hidden at random with a rate
#' \eqn{p \sim U(0,1)} drawn per row and minibatch, and columns on neither side are always unknown and
#' never scored. As in a model formula, \code{.} means all columns not on the other side, and \code{.}
#' on both sides means all columns on both sides. \code{. ~ .} (default) trains every conditional
#' query, \code{. ~ 0} the unconditional prediction only, \code{birds ~ plants} exactly the query
#' "birds given plants". The validation rows get one mask drawn once from the same scheme, and the
#' validation loss is the pooled mean over their hidden cells.
#'
#' @return An S3 object of class \code{"citotnn"}, structured like a \code{\link{mmn}} fit.
#' @example /inst/examples/tnn-example.R
#' @seealso \code{\link{predict.citotnn}}, \code{\link{mmn}}, \code{\link{continue_training}}
#' @export
tnn = function(formula,
               dataList = NULL,
               emb_dim = 32L,
               heads = 4L,
               layers = 2L,
               ff_dim = 128L,
               dropout = 0.1,
               register = NULL,
               attention = NULL,
               masking = NULL,
               groups = NULL,
               loss = c("mse", "mae", "cross-entropy", "bernoulli", "gaussian", "binomial", "poisson", "mvp", "nbinom", "multinomial", "clogit"),
               custom_parameters = NULL,
               optimizer = c("adam", "sgd", "adadelta", "adagrad", "rmsprop", "rprop", "ignite_adam"),
               lr = 0.001,
               lr_scheduler = NULL,
               alpha = 0.5,
               lambda = 0.0,
               validation = 0.0,
               batchsize = NULL,
               shuffle = TRUE,
               data_augmentation = NULL,
               epochs = 100,
               weights = NULL,
               early_stopping = Inf,
               burnin = Inf,
               baseloss = NULL,
               device = c("cpu", "cuda", "mps"),
               plot = TRUE,
               verbose = TRUE) {

  caller_env = environment(formula)
  if(is.null(caller_env)) caller_env = parent.frame()
  checkmate::assert_list(dataList, names = "named")
  checkmate::qassert(emb_dim, "X1[1,)")
  checkmate::qassert(heads, "X1[1,)")
  checkmate::qassert(layers, "X1[1,)")
  checkmate::qassert(ff_dim, "X1[1,)")
  checkmate::qassert(dropout, "N1[0,1)")
  checkmate::qassert(register, c("0", "B1"))
  checkmate::assert(checkmate::checkNull(attention), checkmate::checkList(attention, types = "formula"))
  checkmate::assert(checkmate::checkNull(masking), checkmate::checkFormula(masking))
  checkmate::assert(checkmate::checkNull(groups), checkmate::checkList(groups, types = "character", names = "unique"))
  checkmate::qassert(lr, "N1(0,)")
  checkmate::qassert(validation, c("N1[0,1)", "X>1[1,)"))
  checkmate::qassert(batchsize, c("0", "X1[1,)"))
  checkmate::qassert(epochs, "X1[0,)")
  checkmate::qassert(early_stopping, "N1[1,]")
  checkmate::qassert(burnin, "N1[1,]")
  checkmate::qassert(baseloss, c("0", "N1"))
  checkmate::qassert(plot, "B1")
  checkmate::qassert(verbose, "B1")
  if(emb_dim %% heads != 0) stop("emb_dim must be a multiple of heads.", call. = FALSE)
  device = match.arg(device)
  device_old = device
  device = check_device(device)
  if(is.character(loss)) loss = match.arg(loss)
  optimizer = if(is.character(optimizer)) match.arg(optimizer) else optimizer

  spec = tnn_spec(formula, dataList, caller_env)
  Y = dataList[[spec$response]]
  n = NROW(Y)
  token_mode = spec$response %in% vapply(spec$tokens, function(g) g$name, character(1))
  if(is.null(register)) register = token_mode
  if(is.null(batchsize)) batchsize = max(1L, round(0.1 * n))

  props = list(subModules = if(is.null(spec$rhs)) list() else get_model_properties(spec$rhs, dataList, caller_env),
               groups = lapply(spec$tokens, function(g) list(name = g$name, columns = colnames(token_matrix(dataList[[g$name]], g$name)))),
               modalities = spec$modalities,
               emb_dim = as.integer(emb_dim), heads = as.integer(heads), layers = as.integer(layers),
               ff_dim = as.integer(ff_dim), dropout = dropout, register = register, cls = !token_mode,
               response = NULL)
  tok = token_table(props)
  props$attention = compile_attention(attention, tok, groups, if(token_mode) spec$response)
  props$attention_rules = vapply(attention, deparse1, character(1))
  props$groups_named = groups

  if(token_mode) {
    if(!is.null(weights)) stop("weights are not supported with a tokenized response.", call. = FALSE)
    if(!(is.function(loss) || loss %in% c("mse", "mae", "gaussian", "bernoulli", "poisson", "nbinom")))
      stop("With a tokenized response the loss must be elementwise: mse, mae, gaussian, bernoulli, poisson, nbinom or a function.", call. = FALSE)
    props$response = which(vapply(spec$tokens, function(g) g$name, character(1)) == spec$response)
    Ym = token_matrix(Y, spec$response)
    props$masking = compile_masking(masking, tok[tok$group == spec$response, ], groups)
    props$masking_rule = if(is.null(masking)) ". ~ ." else deparse1(masking)
    loss_obj = tnn_masked_loss(tnn_inner_loss(loss, Ym, custom_parameters, baseloss))
  } else {
    if(!is.null(masking)) stop("'masking' needs the response as tokens(): ", spec$response, " ~ ... + tokens(", spec$response, ").", call. = FALSE)
    if(!is.null(weights)) weights = matrix(weights, nrow = n)
    loss_obj = get_loss(loss, Y, custom_parameters, baseloss, weights)
    assert_weights_supported(loss, weights)
    props$y_dim = loss_obj$y_dim
  }
  class(props) = "citotnn_properties"

  valid = if(length(validation) == 1 && validation == 0) NULL
          else if(length(validation) > 1) validation
          else sort(sample.int(n, round(validation * n)))
  val_hidden = NULL
  if(token_mode) {
    vm = draw_validation_mask(Ym, valid, props$masking)
    valid = vm$valid
    val_hidden = vm$mask
  }
  if(!is.null(data_augmentation)) data_augmentation = check_data_augmentation(data_augmentation)

  out = list(net = build_tnn(props),
             version = "1.2",
             call = match.call(),
             loss = loss_obj,
             data = list(dataList = dataList, spec = spec, validation = valid, val_hidden = val_hidden, weights = weights),
             model_properties = props,
             training_properties = list(optimizer = optimizer, lr = lr, lr_scheduler = lr_scheduler,
                                        alpha = alpha, lambda = lambda, validation = validation,
                                        batchsize = batchsize, shuffle = shuffle,
                                        data_augmentation = data_augmentation, epochs = epochs,
                                        early_stopping = early_stopping, burnin = burnin,
                                        baseloss = loss_obj$baseloss, device = device_old,
                                        plot = plot, verbose = verbose, has_weights = !is.null(weights)))
  out$call$formula = formula
  class(out) = "citotnn"
  if(token_mode) set_eval_scale(out)
  dls = tnn_loaders(out)
  train_model(model = out, epochs = epochs, device = device, train_dl = dls$train_dl, valid_dl = dls$valid_dl)
}

# Splits the formula into the dnn()/cnn() part, which goes through mmn's machinery unchanged, and
# the tokens() terms, which mmn() must keep rejecting.
tnn_spec = function(formula, dataList, env) {
  if(length(formula) != 3 || !is.name(formula[[2]])) stop("The left side of ~ must be the name of the response in dataList.", call. = FALSE)
  response = as.character(formula[[2]])
  if(!response %in% names(dataList)) stop("Couldn't find '", response, "' in names of dataList.", call. = FALSE)
  terms = flatten_sum(formula[[3]])
  is_tok = vapply(terms, function(t) is.call(t) && identical(t[[1]], as.name("tokens")), logical(1))
  modal = terms[!is_tok]
  rhs = if(length(modal) > 0) Reduce(function(a, b) call("+", a, b), modal) else NULL
  if(!is.null(rhs)) check_mmn_formula(call("~", formula[[2]], rhs))
  tok = lapply(terms[is_tok], function(t) {
    cl = match.call(function(M, transform = NULL) NULL, t)
    if(!is.name(cl$M) || !as.character(cl$M) %in% names(dataList))
      stop("In '", deparse1(t), "': the matrix must be a name in dataList.", call. = FALSE)
    list(name = as.character(cl$M), transform = if(is.null(cl$transform)) NULL else eval(cl$transform, env))
  })
  names_tok = vapply(tok, function(g) g$name, character(1))
  if(anyDuplicated(names_tok)) stop("Each matrix may enter tokens() once.", call. = FALSE)
  if(length(modal) == 0 && length(tok) == 0) stop("The formula has no terms on the right of ~.", call. = FALSE)
  list(response = response, rhs = rhs, tokens = tok, modalities = dedupe_names(vapply(modal, modality_name, character(1))))
}

flatten_sum = function(term) {
  if(is.call(term) && identical(term[[1]], as.name("+")) && length(term) == 3) c(flatten_sum(term[[2]]), flatten_sum(term[[3]]))
  else list(term)
}

modality_name = function(term) {
  cl = match.call(if(identical(term[[1]], as.name("cnn"))) cnn else dnn, term)
  deparse1(if(!is.null(cl$X)) cl$X else cl$data)
}

dedupe_names = function(x) {
  for(nm in unique(x[duplicated(x)])) {
    i = which(x == nm)
    x[i[-1]] = paste0(nm, ".", seq_along(i)[-1])
  }
  x
}

token_matrix = function(M, name) {
  if(is.data.frame(M)) M = as.matrix(M)
  if(is.null(dim(M))) M = matrix(M, ncol = 1L)
  if(!is.numeric(M) && !is.logical(M)) stop("tokens(", name, ") needs a numeric matrix or data.frame.", call. = FALSE)
  storage.mode(M) = "double"
  if(is.null(colnames(M))) colnames(M) = paste0(name, "_", seq_len(ncol(M)))
  M
}

token_table = function(props) {
  M = length(props$modalities)
  grp = c(props$modalities, vapply(props$groups, function(g) g$name, character(1)), "register", "cls")
  if(anyDuplicated(grp)) stop("Token group names must be unique ('register' and 'cls' are reserved): ",
                              paste(unique(grp[duplicated(grp)]), collapse = ", "), call. = FALSE)
  cols = lapply(props$groups, function(g) data.frame(group = g$name, name = g$columns, kind = "column"))
  tok = do.call(rbind, c(list(data.frame(group = props$modalities, name = props$modalities, kind = rep("modality", M))),
                         cols,
                         if(props$register) list(data.frame(group = "register", name = "register", kind = "register")),
                         if(props$cls) list(data.frame(group = "cls", name = "cls", kind = "cls"))))
  clash = intersect(unique(tok$group[tok$kind == "column"]), tok$name[tok$kind == "column"])
  if(anyDuplicated(tok$name) || length(clash) > 0)
    stop("Token names must be unique across modalities, tokens() columns, 'register' and 'cls': ",
         paste(unique(c(tok$name[duplicated(tok$name)], clash)), collapse = ", "), call. = FALSE)
  tok
}

# A set of tokens from the query grammar: names, '.', '0', '+', '-' and '('.
token_set = function(term, tok, groups, dot) {
  if(is.numeric(term) && length(term) == 1 && term == 0) return(rep(FALSE, nrow(tok)))
  if(is.name(term)) {
    nm = as.character(term)
    if(nm == ".") return(if(is.null(dot)) rep(FALSE, nrow(tok)) else dot)
    if(nm %in% tok$group) return(tok$group == nm)
    if(nm %in% tok$name) return(tok$name == nm)
    if(nm %in% names(groups)) return(Reduce(`|`, lapply(groups[[nm]], function(g) token_set(as.name(g), tok, NULL, NULL))))
    stop("Unknown token or group: ", nm, call. = FALSE)
  }
  if(is.call(term)) {
    op = as.character(term[[1]])
    if(op == "(") return(token_set(term[[2]], tok, groups, dot))
    if(op == "+" && length(term) == 3) return(token_set(term[[2]], tok, groups, dot) | token_set(term[[3]], tok, groups, dot))
    if(op == "-" && length(term) == 3) return(token_set(term[[2]], tok, groups, dot) & !token_set(term[[3]], tok, groups, dot))
    stop("Only names, '.', '0', '+' and '-' are allowed, not '", op, "'.", call. = FALSE)
  }
  stop("Cannot read: ", deparse1(term), call. = FALSE)
}

uses_dot = function(term) {
  if(is.name(term)) return(identical(term, as.name(".")))
  if(is.call(term)) return(any(vapply(as.list(term)[-1], uses_dot, logical(1))))
  FALSE
}

# Both sides of targets ~ given; '.' is everything not on the other side, '.' on both sides everything.
two_sided_sets = function(f, tok, groups) {
  if(!inherits(f, "formula") || length(f) != 3) stop("Expected a two-sided formula: ", deparse1(f), call. = FALSE)
  all = rep(TRUE, nrow(tok))
  ld = uses_dot(f[[2]])
  rd = uses_dot(f[[3]])
  if(ld && rd) return(list(left = token_set(f[[2]], tok, groups, all), right = token_set(f[[3]], tok, groups, all)))
  left = token_set(f[[2]], tok, groups, NULL)
  right = token_set(f[[3]], tok, groups, NULL)
  if(ld) left = token_set(f[[2]], tok, groups, !right)
  if(rd) right = token_set(f[[3]], tok, groups, !left)
  list(left = left, right = right)
}

compile_attention = function(rules, tok, groups, response = NULL) {
  token_mode = !is.null(response)
  n = nrow(tok)
  keep = matrix(TRUE, n, n, dimnames = list(tok$name, tok$name))
  if(token_mode) for(m in which(tok$kind == "modality")) keep[m, ] = tok$kind == "register" | seq_len(n) == m
  for(r in rules) {
    if(!inherits(r, "formula") || length(r) != 3) stop("Attention rules are two-sided formulas, readers ~ readable.", call. = FALSE)
    readers = token_set(r[[2]], tok, groups, rep(TRUE, n))
    if(!any(readers)) stop("The rule '", deparse1(r), "' has no readers.", call. = FALSE)
    readable = token_set(r[[3]], tok, groups, rep(TRUE, n))
    keep[readers, ] = matrix(readable, sum(readers), n, byrow = TRUE)
  }
  resp = which(tok$group == response & tok$kind == "column")
  keep[cbind(resp, resp)] = FALSE
  anchor = tok$kind != "column"
  orphan = rowSums(keep[, anchor, drop = FALSE]) == 0
  if(any(orphan)) stop("These tokens cannot read any token that is never unknown (a dnn()/cnn() token, register or cls), ",
                       "so a row in which their other keys are unknown leaves them nothing to read: ",
                       paste(utils::head(tok$name[orphan], 5), collapse = ", "), call. = FALSE)
  keep
}

compile_masking = function(f, tok, groups) {
  if(is.null(f)) f = . ~ .
  s = two_sided_sets(f, tok, groups)
  if(!any(s$left)) stop("The masking rule '", deparse1(f), "' hides no column, so nothing would be scored.", call. = FALSE)
  list(always = s$left & !s$right, random = s$left & s$right, never = !s$left & !s$right)
}

# The cito loss on cells, with an NA-free target and a baseline computed on the observed cells only.
tnn_inner_loss = function(loss, Y, custom_parameters, baseloss) {
  obs = !is.na(Y)
  Yf = Y
  Yf[!obs] = 0
  inner = get_loss(loss, Yf, custom_parameters, baseloss = if(is.null(baseloss)) 0 else baseloss)
  if(is.null(baseloss)) {
    mu = colMeans(Y, na.rm = TRUE)
    mu[!is.finite(mu)] = 0
    if(identical(loss, "bernoulli")) mu = pmin(pmax(mu, 1e-4), 1 - 1e-4)
    if(identical(loss, "poisson") || identical(loss, "nbinom")) mu = pmax(mu, 1e-4)
    pred = inner$link(torch::torch_tensor(matrix(mu, nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32()))
    li = torch::with_no_grad(inner(pred, torch::torch_tensor(Yf, dtype = torch::torch_float32())))
    inner$baseloss = mean(as.matrix(li)[obs])
  }
  inner
}

tnn_masked_loss = torch::nn_module(
  "tnn_masked_loss",
  initialize = function(inner) {
    self$inner = inner
    self$y_dim = inner$y_dim
    self$responses = inner$responses
    self$baseloss = inner$baseloss
    self$eval_scale = NULL
  },
  # unreduced: one value per hidden cell, so cito's $mean() is the pooled loss of a minibatch
  forward = function(output, target) {
    hidden = output$hidden
    if(hidden$sum()$item() == 0) return((output$pred * 0)$sum()$view(1L))
    v = self$inner(output$pred, target)[hidden]
    # cito averages the means of the validation batches; scaled by (batches x cells in batch / all
    # cells), that average is the pooled mean over the whole validation set
    if(!self$training && !is.null(self$eval_scale))
      v = v * (self$eval_scale[["batches"]] * v$size(1) / self$eval_scale[["cells"]])
    v
  },
  link = function(x) self$inner$link(x),
  invlink = function(x) self$inner$invlink(x)
)

set_eval_scale = function(object) {
  v = object$data$validation
  object$loss$eval_scale = if(is.null(v)) NULL else
    c(batches = ceiling(length(v) / object$training_properties$batchsize), cells = sum(object$data$val_hidden[v, ]))
  invisible(object)
}

# One mask per validation row, drawn once from the training scheme; a row that drew nothing to score
# is redrawn, and a row that can never have a hidden cell leaves the validation set.
draw_validation_mask = function(Y, valid, masking) {
  m = matrix(FALSE, nrow(Y), ncol(Y))
  if(is.null(valid)) return(list(mask = m, valid = NULL))
  obs = !is.na(Y)
  can = obs & matrix(masking$always | masking$random, nrow(Y), ncol(Y), byrow = TRUE)
  empty = rowSums(can[valid, , drop = FALSE]) == 0
  if(any(empty)) message(sum(empty), " validation rows without any cell that could be hidden are left out of the validation set.")
  valid = valid[!empty]
  todo = valid
  while(length(todo) > 0) {
    k = length(todo)
    rnd = matrix(stats::runif(k * ncol(Y)), k) < stats::runif(k)
    m[todo, ] = can[todo, , drop = FALSE] & (matrix(masking$always, k, ncol(Y), byrow = TRUE) | (matrix(masking$random, k, ncol(Y), byrow = TRUE) & rnd))
    todo = todo[rowSums(m[todo, , drop = FALSE]) == 0]
  }
  list(mask = m, valid = valid)
}

# The input streams in the batch layout of TNN: modality streams, value and known per tokens() group,
# and with a tokenized response the fixed validation mask. `values` replaces matrices of dataList.
tnn_inputs = function(object, dataList, hidden = NULL) {
  spec = object$data$spec
  props = object$model_properties
  X = if(is.null(spec$rhs)) list() else format_input_data(spec$rhs, dataList)
  from_folder = any(vapply(X, lazy_input, logical(1)))
  X = lapply(X, expand_input)
  for(g in seq_along(spec$tokens)) {
    name = spec$tokens[[g]]$name
    if(!name %in% names(dataList)) stop("The data lacks '", name, "'.", call. = FALSE)
    M = align_columns(token_matrix(dataList[[name]], name), props$groups[[g]]$columns, name)
    obs = !is.na(M)
    v = M
    tf = spec$tokens[[g]]$transform
    if(!is.null(tf)) v[obs] = tf(v[obs])
    if(any(!is.finite(v[obs]))) stop("tokens(", name, ") has non-finite values", if(!is.null(tf)) " after the transform", ".", call. = FALSE)
    v[!obs] = 0
    X = c(X, list(torch::torch_tensor(v, dtype = torch::torch_float32()), torch::torch_tensor(obs, dtype = torch::torch_bool())))
  }
  rows = vapply(X, function(x) if(inherits(x, "torch_tensor")) x$size(1) else length(x), numeric(1))
  if(length(unique(rows)) > 1) stop("The inputs have different numbers of rows: ", paste(unique(rows), collapse = ", "),
                                    ". Rows with NA predictors are dropped by model.frame(); remove or impute them.", call. = FALSE)
  if(!is.null(props$response)) {
    if(is.null(hidden)) hidden = matrix(FALSE, rows[1], length(props$groups[[props$response]]$columns))
    X = c(X, list(torch::torch_tensor(hidden, dtype = torch::torch_bool())))
  }
  list(inputs = X, from_folder = from_folder, n = rows[1])
}

align_columns = function(M, columns, name) {
  if(identical(colnames(M), columns)) return(M)
  if(all(columns %in% colnames(M))) return(M[, columns, drop = FALSE])
  if(ncol(M) == length(columns) && all(grepl(paste0("^", name, "_[0-9]+$"), colnames(M)))) {
    colnames(M) = columns
    return(M)
  }
  stop(name, " lacks the columns the model was fitted with: ", paste(utils::head(setdiff(columns, colnames(M)), 5), collapse = ", "), call. = FALSE)
}

tnn_loaders = function(object) {
  weights = object$data$weights
  inp = tnn_inputs(object, object$data$dataList, object$data$val_hidden)
  props = object$model_properties
  Y = object$data$dataList[[object$data$spec$response]]
  target = if(is.null(props$response)) object$loss$format_Y(Y) else {
    Ym = token_matrix(Y, object$data$spec$response)
    Ym[is.na(Ym)] = 0
    torch::torch_tensor(Ym, dtype = torch::torch_float32())
  }
  build_loaders(inp$inputs, target, weights = if(is.null(weights)) NULL else torch::torch_tensor(weights),
                valid = object$data$validation, batch_size = object$training_properties$batchsize,
                shuffle = object$training_properties$shuffle, from_folder = inp$from_folder,
                data_augmentation = object$training_properties$data_augmentation)
}

#' Predict with a fitted TNN
#'
#' Without a tokenized response this works as \code{\link{predict.citommn}}. With \code{tokens(Y)} for
#' the response there are three kinds of query, all from the same fitted model:
#' \itemize{
#'   \item \strong{unconditional} (\code{Y = NULL}): every response column from the other tokens alone.
#'   \item \strong{conditional} (\code{Y} given): the observed cells of \code{Y} condition the prediction,
#'     the \code{NA} cells are predicted. Cells whose value was given are returned as \code{NA}.
#'   \item \strong{leave-one-out} (\code{loo = TRUE}): each observed column is predicted from all other
#'     observed columns, by hiding it alone; one forward pass per column.
#' }
#' Only what the \code{masking} of the fit trained is answered: columns it never hid are not
#' predicted, and values of columns it never gave do not condition (they are ignored with a warning).
#'
#' @param object A model fitted with \code{\link{tnn}}.
#' @param newdata A list holding every data object the formula names except a tokenized response.
#'   \code{NULL} uses the training data.
#' @param Y Optional matrix of conditioning values for a tokenized response: observed cells condition,
#'   \code{NA} cells are predicted. Columns are matched by name.
#' @param query Optional formula \code{targets ~ given} over the response columns, in the grammar of
#'   the \code{attention} rules of \code{\link{tnn}}: \code{birds ~ plants} predicts the birds from the
#'   values of the plants in \code{Y}, all other columns enter as unknown. A column may not be on both
#'   sides; use \code{loo = TRUE} to predict each column from all others.
#' @param groups Named list of column names usable in \code{query}; defaults to the \code{groups} of the fit.
#' @param loo Logical, leave-one-out prediction for the columns observed in \code{Y}.
#' @param type \code{"link"}, \code{"response"} or, without a tokenized response, \code{"class"}.
#' @param device,batchsize As in \code{\link{predict.citommn}}.
#' @param return_embeddings Without a tokenized response, return the output of the \code{cls} token.
#' @param ... Not used.
#' @return A matrix of predictions; with a tokenized response, cells that are not predicted are \code{NA}.
#' @export
predict.citotnn = function(object, newdata = NULL, Y = NULL, query = NULL, groups = NULL, loo = FALSE,
                           type = c("link", "response", "class"), device = NULL, batchsize = NULL,
                           return_embeddings = FALSE, ...) {
  checkmate::assert(checkmate::checkNull(newdata), checkmate::checkList(newdata, names = "named"))
  checkmate::qassert(loo, "B1")
  object = check_model(object)
  type = match.arg(type)
  if(is.null(device)) device = object$training_properties$device
  device = check_device(device)
  if(is.null(batchsize)) batchsize = object$training_properties$batchsize
  props = object$model_properties
  data = object$data$dataList
  if(!is.null(newdata)) {
    needed = setdiff(c(all.vars(object$data$spec$rhs), vapply(object$data$spec$tokens, function(g) g$name, character(1))),
                     if(is.null(props$response)) character(0) else object$data$spec$response)
    needed = intersect(needed, names(data))
    absent = setdiff(needed, names(newdata))
    if(length(absent) > 0) stop("newdata lacks: ", paste(absent, collapse = ", "), call. = FALSE)
    data[names(newdata)] = newdata
  }
  object$net$to(device = device)
  on.exit(object$net$to(device = "cpu"), add = TRUE)
  if(is.null(props$response)) return(tnn_predict_head(object, data, type, device, batchsize, return_embeddings))
  if(type == "class") stop("type = 'class' needs an untokenized response.", call. = FALSE)
  tnn_predict_tokens(object, data, newdata, Y, query, if(is.null(groups)) props$groups_named else groups, loo, type, device, batchsize)
}

tnn_predict_head = function(object, data, type, device, batchsize, return_embeddings) {
  inp = tnn_inputs(object, data)
  dl = do.call(get_data_loader, c(inp$inputs, list(batch_size = batchsize, shuffle = FALSE, from_folder = inp$from_folder)))
  link = if(type %in% c("response", "class")) object$loss$invlink else function(a) a
  pred = list()
  torch::with_no_grad(coro::loop(for(b in dl) {
    b = lapply(b, function(x) x$to(device = device))
    out = object$net(b, return_embeddings = return_embeddings)
    pred[[length(pred) + 1L]] = as.matrix(if(return_embeddings) out$cpu() else link(out)$cpu())
  }))
  pred = do.call(rbind, pred)
  if(!return_embeddings && !is.null(object$loss$responses)) {
    colnames(pred) = object$loss$responses
    if(type == "class") pred = factor(object$loss$responses[apply(pred, 1, which.max)], levels = object$loss$responses)
  }
  pred
}

tnn_predict_tokens = function(object, data, newdata, Y, query, groups, loo, type, device, batchsize) {
  props = object$model_properties
  r = props$response
  rname = object$data$spec$response
  cols = props$groups[[r]]$columns
  if(!is.null(newdata) && rname %in% names(newdata)) message("The response in newdata is not used; conditioning values go into 'Y'.")
  n = tnn_rows(object, data, if(is.null(Y)) 0L else if(is.null(dim(Y))) 1L else nrow(Y))
  if(!is.null(Y)) Y = align_columns(token_matrix(if(is.null(dim(Y)) && n == 1) matrix(Y, 1, dimnames = list(NULL, names(Y))) else Y, rname), cols, rname)
  if(!is.null(Y) && nrow(Y) != n) stop(sprintf("Y has %d rows, the other data %d.", nrow(Y), n), call. = FALSE)
  # the masking decides what the model was trained to do: only columns it hid can be predicted, and
  # only columns it gave can condition
  givable = !props$masking$always & !props$masking$never
  hideable = props$masking$always | props$masking$random
  targets = hideable
  if(loo && !is.null(query)) stop("Use either 'query' or 'loo = TRUE', not both.", call. = FALSE)
  if(!is.null(query)) {
    tok = data.frame(group = rname, name = cols, kind = "column")
    s = two_sided_sets(query, tok, groups)
    if(any(s$left & s$right)) stop("Columns on both sides of the query: ", paste(utils::head(cols[s$left & s$right], 5), collapse = ", "),
                                   ". Use loo = TRUE to predict each column from the others.", call. = FALSE)
    if(!any(s$left)) stop("The query predicts no column.", call. = FALSE)
    if(any(s$left & !hideable)) stop("The masking never hid these columns in training, so they cannot be predicted: ",
                                     paste(utils::head(cols[s$left & !hideable], 5), collapse = ", "), call. = FALSE)
    if(any(s$right & !givable)) stop("The masking never gave these columns in training, so they cannot condition: ",
                                     paste(utils::head(cols[s$right & !givable], 5), collapse = ", "), call. = FALSE)
    if(any(s$right) && is.null(Y)) stop("The query conditions on columns, so their values must be given in 'Y'.", call. = FALSE)
    Yq = matrix(NA_real_, n, length(cols), dimnames = list(NULL, cols))
    if(any(s$right)) Yq[, s$right] = Y[, s$right]
    Y = Yq
    targets = s$left
  }
  if(!is.null(Y) && any(!is.na(Y[, !givable]))) {
    warning("Values of columns that the masking never gave in training are ignored: ",
            paste(utils::head(cols[!givable & colSums(!is.na(Y)) > 0], 5), collapse = ", "), call. = FALSE)
    Y[, !givable] = NA
  }
  if(is.null(Y)) Y = matrix(NA_real_, n, length(cols), dimnames = list(NULL, cols))
  loo_cols = if(loo) which(colSums(!is.na(Y)) > 0 & targets & givable) else integer(0)
  if(loo && length(loo_cols) == 0) stop("loo = TRUE needs observed values in 'Y'.", call. = FALSE)
  if(!loo && !anyNA(Y[, targets, drop = FALSE])) stop("Every cell of Y is observed, so there is nothing to predict. Use loo = TRUE to predict each column from the others.", call. = FALSE)
  data[[rname]] = Y
  inp = tnn_inputs(object, data)
  inputs = inp$inputs[-length(inp$inputs)]
  dl = do.call(get_data_loader, c(inputs, list(batch_size = batchsize, shuffle = FALSE, from_folder = inp$from_folder)))
  net = object$net
  pred = matrix(NA_real_, n, length(cols))
  out_loo = matrix(NA_real_, n, length(cols))
  start = 0L
  torch::with_no_grad(coro::loop(for(b in dl) {
    b = lapply(b, function(x) x$to(device = device))
    s = net$split_input(b)
    mtok = net$modality_tokens(s$modal)
    k = s$known[[r]]$size(1)
    idx = start + seq_len(k)
    start = start + k
    pred[idx, ] = as.matrix(net$response_logits(net$encode(mtok, s$values, s$known))$cpu())
    if(length(loo_cols) == 0) next
    m = length(loo_cols)
    rep_ = function(x) if(is.null(x)) NULL else x$repeat_interleave(m, dim = 1L)
    known = lapply(s$known, rep_)
    rows = torch::torch_arange(1L, k * m, dtype = torch::torch_long(), device = device)
    hide = torch::torch_tensor(rep(loo_cols, times = k), dtype = torch::torch_long(), device = device)
    known[[r]] = known[[r]]$index_put(list(rows, hide), torch::torch_tensor(FALSE, device = device))
    lg = as.matrix(net$response_logits(net$encode(rep_(mtok), lapply(s$values, rep_), known))$cpu())
    out_loo[idx, loo_cols] = matrix(lg[cbind(seq_len(k * m), rep(loo_cols, times = k))], ncol = m, byrow = TRUE)
  }))
  pred[!is.na(Y)] = NA
  if(length(loo_cols) > 0) pred[, loo_cols] = out_loo[, loo_cols]
  pred[, !targets] = NA
  if(type == "response") {
    done = !is.na(pred)
    pred[done] = as.numeric(object$loss$invlink(torch::torch_tensor(pred[done], dtype = torch::torch_float32())))
  }
  dimnames(pred) = list(tnn_row_names(data, object, n), cols)
  pred
}

tnn_rows = function(object, data, n_Y) tnn_inputs(object, replace_response(object, data, n_Y))$n

replace_response = function(object, data, n_Y = 0) {
  rname = object$data$spec$response
  cols = object$model_properties$groups[[object$model_properties$response]]$columns
  others = setdiff(names(data), rname)
  n = NULL
  for(nm in c(all.vars(object$data$spec$rhs), vapply(object$data$spec$tokens, function(g) g$name, character(1)))) {
    if(nm == rname || !nm %in% others) next
    x = data[[nm]]
    n = if(is.character(x) && length(x) == 1 && dir.exists(x)) length(list.files(x)) else if(inherits(x, "citoraster")) length(x) else NROW(x)
    break
  }
  if(is.null(n)) n = if(n_Y > 0) n_Y else NROW(data[[rname]])
  data[[rname]] = matrix(NA_real_, n, length(cols), dimnames = list(NULL, cols))
  data
}

tnn_row_names = function(data, object, n) {
  for(nm in c(all.vars(object$data$spec$rhs), vapply(object$data$spec$tokens, function(g) g$name, character(1)))) {
    x = data[[nm]]
    if(inherits(x, "citoraster")) x = x$coords
    if(!is.null(rownames(x)) && NROW(x) == n) return(rownames(x))
  }
  NULL
}

#' Print a fitted TNN
#' @param x A model created by \code{\link{tnn}}.
#' @param ... Not used.
#' @return \code{x}, invisibly.
#' @export
print.citotnn = function(x, ...) {
  p = x$model_properties
  K = vapply(p$groups, function(g) length(g$columns), integer(1))
  cat(sprintf("Transformer neural network: %d modality token(s), %d column token(s)%s%s\n", length(p$modalities), sum(K),
              if(p$register) " + register" else "", if(p$cls) " + cls" else ""))
  cat(sprintf("  tokens of width %d, %d layer(s), %d head(s)\n", p$emb_dim, p$layers, p$heads))
  if(!is.null(p$response)) cat(sprintf("  response %s as tokens, masking %s\n", x$data$spec$response, p$masking_rule))
  if(length(p$attention_rules) > 0) cat("  attention: ", paste(p$attention_rules, collapse = ", "), "\n", sep = "")
  cat("Call: ", deparse1(x$call), "\n", sep = "")
  invisible(x)
}

#' @rdname continue_training
#' @export
continue_training.citotnn = function(model, epochs = 32, dataList = NULL, device = NULL, changed_params = NULL, init_optimizer = TRUE, ...) {
  checkmate::qassert(epochs, "X1[1,)")
  if(!is.null(dataList)) stop("continue_training() of a tnn() model continues on its training data; new data is not supported yet.", call. = FALSE)
  if(is.null(device)) device = model$training_properties$device
  device = check_device(device)
  model$use_model_epoch = "last"
  model = check_model(model)
  for(name in names(changed_params)) model$training_properties[[name]] = changed_params[[name]]
  if(!is.null(model$model_properties$response)) set_eval_scale(model)
  dls = tnn_loaders(model)
  train_model(model = model, epochs = epochs, device = device, train_dl = dls$train_dl, valid_dl = dls$valid_dl,
              plot_new = TRUE, init_optimizer = init_optimizer)
}

#' Summarize a fitted TNN
#'
#' Permutation importance of every \code{dnn()}/\code{cnn()} token and every \code{tokens()} group:
#' one is shuffled across rows while the others are left intact, and the increase in loss is what it
#' contributed. With a tokenized response the loss is the loss of the cells hidden by one mask, drawn
#' from the training scheme and fixed across permutations, and the table gains a row for the given
#' values of the response: the increase in loss when every response cell is unknown, i.e. the
#' conditional gain.
#'
#' @param object A model created by \code{\link{tnn}}.
#' @param n_permute Number of permutations per stream.
#' @param device Device on which to evaluate. Defaults to the training device.
#' @param ... Not used.
#'
#' @details
#' Evaluated on the training data. Permutation breaks the dependence between correlated streams and
#' evaluates the network away from the data it was trained on, and the standard deviation is across
#' permutations of one fitted model, not across fits. The conditional gain is not a measure of
#' interactions between the response columns: shared responses to unmeasured predictors produce it too.
#'
#' @return An object of class \code{"summary.citotnn"} with the importance table.
#' @export
summary.citotnn = function(object, n_permute = 10L, device = NULL, ...) {
  out = list(call = object$call, importance = tnn_importance(object, n_permute, device),
             response_tokens = !is.null(object$model_properties$response))
  class(out) = "summary.citotnn"
  out
}

#' @rdname summary.citotnn
#' @param x A summary of a fitted TNN.
#' @export
print.summary.citotnn = function(x, ...) {
  cat("Summary of Transformer Neural Network\n\n")
  cli::cli_h3(cli::style_bold("Permutation importance of the token streams"))
  m = as.matrix(x$importance[, c("loss_increase", "sd", "share"), drop = FALSE])
  rownames(m) = x$importance$modality
  colnames(m) = c("Loss increase", "SD", "Share")
  print(round(m, 4))
  cat(sprintf("\nBaseline loss %.4f%s, %d permutations per stream. Compare an estimate against its SD before reading it.\n",
              x$importance$baseline_loss[1], if(x$response_tokens) " (hidden cells of one fixed mask)" else "",
              x$importance$n_permute[1]))
  invisible(x)
}

tnn_importance = function(object, n_permute = 10L, device = NULL) {
  checkmate::qassert(n_permute, "X1[1,)")
  object = check_model(object)
  if(is.null(device)) device = object$training_properties$device
  device = check_device(device)
  props = object$model_properties
  token_mode = !is.null(props$response)
  data = object$data$dataList
  Y = data[[object$data$spec$response]]
  hidden = NULL
  if(token_mode) {
    Ym = token_matrix(Y, object$data$spec$response)
    hidden = suppressMessages(draw_validation_mask(Ym, seq_len(nrow(Ym)), props$masking))$mask
    Ym[is.na(Ym)] = 0
    target = torch::torch_tensor(Ym, dtype = torch::torch_float32())
  } else {
    target = object$loss$format_Y(Y)
  }
  inp = tnn_inputs(object, data, hidden)
  n = inp$n
  # positions of each stream in the batch layout: modality inputs (two with embeddings), then value
  # and known of each tokens() group
  m_len = vapply(seq_along(props$subModules), function(j) 1L + isTRUE(object$net$has_embeddings[j]), integer(1))
  ends = cumsum(m_len)
  streams = lapply(seq_along(m_len), function(j) (ends[j] - m_len[j] + 1L):ends[j])
  labels = props$modalities
  off = sum(m_len)
  for(g in seq_along(props$groups)) {
    if(token_mode && g == props$response) next
    streams = c(streams, list(off + c(2L * g - 1L, 2L * g)))
    labels = c(labels, props$groups[[g]]$name)
  }
  object$net$to(device = device)
  object$loss$to(device = device)
  on.exit(object$net$to(device = "cpu"), add = TRUE)
  on.exit(object$loss$to(device = "cpu"), add = TRUE)
  scale = object$loss$eval_scale
  object$loss$eval_scale = NULL
  on.exit(object$loss$eval_scale <- scale, add = TRUE)
  w = if(is.null(object$data$weights)) NULL else torch::torch_tensor(object$data$weights)
  r = props$response
  run_loss = function(ins, unconditional = FALSE) {
    dl = do.call(get_data_loader, c(ins, if(is.null(w)) list(target) else list(w, target),
                                    list(batch_size = object$training_properties$batchsize,
                                         shuffle = FALSE, from_folder = inp$from_folder)))
    total = 0
    count = 0
    torch::with_no_grad(coro::loop(for(b in dl) {
      b = lapply(b, function(z) z$to(device = device))
      x = b[seq_along(ins)]
      tgt = b[[length(b)]]
      v = if(!token_mode) {
        if(is.null(w)) object$loss(object$net(x), tgt) else object$loss(object$net(x), tgt, weights = b[[length(b) - 1L]])
      } else if(!unconditional) {
        out = object$net(x)
        object$loss$inner(out$pred, tgt)[out$hidden]
      } else {
        s = object$net$split_input(x)
        hid = x[[length(x)]]$logical_and(s$known[[r]])
        s$known[[r]] = torch::torch_zeros_like(s$known[[r]])
        object$loss$inner(object$net$response_logits(object$net$encode(object$net$modality_tokens(s$modal), s$values, s$known)), tgt)[hid]
      }
      total = total + as.numeric(v$sum()$cpu())
      count = count + v$numel()
    }))
    total / count
  }
  base = run_loss(inp$inputs)
  draws = lapply(streams, function(idx) vapply(seq_len(n_permute), function(k) {
    perm = sample.int(n)
    ins = inp$inputs
    for(i in idx) ins[[i]] = permute_rows(ins[[i]], perm)
    run_loss(ins)
  }, numeric(1)) - base)
  if(token_mode) {
    draws = c(draws, list(run_loss(inp$inputs, unconditional = TRUE) - base))
    labels = c(labels, paste0(object$data$spec$response, " (given values)"))
  }
  inc = vapply(draws, mean, numeric(1))
  sd_inc = vapply(draws, function(d) if(length(d) > 1) stats::sd(d) else NA_real_, numeric(1))
  share = if(all(inc <= 0)) rep(NA_real_, length(inc)) else pmax(inc, 0) / sum(pmax(inc, 0))
  data.frame(modality = labels, loss_increase = inc, sd = sd_inc, share = share,
             baseline_loss = base, n_permute = n_permute, stringsAsFactors = FALSE)
}

#' Plot a fitted TNN
#'
#' @param x A model created by \code{\link{tnn}}.
#' @param plot_type \code{"importance"} (the table of \code{\link{summary.citotnn}} as a bar chart) or
#'   \code{"attention"} (which token may read which, as compiled from the \code{attention} rules).
#' @param n_permute,device See \code{\link{summary.citotnn}}.
#' @param ... Passed to \code{\link[graphics]{barplot}} or \code{\link[graphics]{image}}.
#' @return Invisibly, the importance table or the attention matrix (rows read columns).
#' @export
plot.citotnn = function(x, plot_type = c("importance", "attention"), n_permute = 10L, device = NULL, ...) {
  plot_type = match.arg(plot_type)
  if(plot_type == "importance") {
    imp = tnn_importance(x, n_permute, device)
    return(plot_importance(imp, imp$modality, ...))
  }
  k = x$model_properties$attention
  n = nrow(k)
  op = graphics::par(mar = c(2.1, max(4.1, 0.6 * max(nchar(rownames(k))) + 1.5), 4.1, 2.1))
  on.exit(graphics::par(op))
  graphics::image(seq_len(n), seq_len(n), t(k[n:1, , drop = FALSE]), col = c("white", "grey30"), axes = FALSE,
                  xlab = "", ylab = "", main = "Attention: rows read columns", ...)
  graphics::axis(3, seq_len(n), colnames(k), las = 2, cex.axis = 0.7)
  graphics::axis(2, seq_len(n), rev(rownames(k)), las = 1, cex.axis = 0.7)
  invisible(k)
}

#' @rdname coef.citommn
#' @export
coef.citotnn = function(object, ...) coef.citommn(object, ...)
