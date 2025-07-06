format_targets <- function(Y, loss_obj, ylvls=NULL) {

  if(is.vector(Y)) y_dim = 1
  else y_dim = ncol(Y)

  if(is.null(ylvls) && is.factor(Y)) ylvls <- levels(Y)
  if(!inherits(Y, "matrix")) Y = as.matrix(Y)
  responses <- colnames(Y)

  if(inherits(loss_obj$call, "family") && loss_obj$call$family == "binomial") {
    if(all(Y %in% c(0,1))) {
      Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())

    } else if(is.character(Y)) {
      if (is.null(ylvls)) {
        Y <- factor(Y[,1])
        ylvls <- levels(Y)
      } else {
        Y <- factor(Y[,1], levels = ylvls)
      }
      Y <- torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32())

    } else {
      Y <- as.integer(Y[,1])
      Y <- torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32())
    }
    y_dim <- ncol(Y)
    Y_base <- torch::torch_tensor(matrix(apply(as.matrix(Y), 2, mean), nrow(Y), ncol(Y), byrow = TRUE))


    ##### TODO move Y preparation to loss objects!!!!

  } else if(!is.function(loss_obj$call) && any(loss_obj$call %in% c("softmax", "cross-entropy"))) {
    if (is.character(Y)) {
      if (is.null(ylvls)) {
        Y <- factor(Y[,1])
        ylvls <- levels(Y)
      } else {
        Y <- factor(Y[,1], levels = ylvls)
      }
      prop <- as.vector(table(Y)/sum(table(Y)))
      y_dim <- length(ylvls)
      Y <- as.matrix(as.integer(Y), ncol=1L)
      if(length(ylvls) != length(unique(Y))) {
        warning("There exist labels without any corresponding samples. Make sure this is intended:\n
                1) dnn, cnn: The provided factor containing the labels has levels with zero occurences. For each level a node in the output layer will be created.\n
                2) continue_training: The new data provided has labels with zero corresponding samples.")
      }
    } else {
      Y <- as.matrix(as.integer(Y[,1]), ncol=1L)
      y_dim <- length(unique(Y))
      prop <- as.vector(table(Y)/sum(table(Y)))
    }
    Y_base <- torch::torch_tensor( matrix(prop, nrow = nrow(Y), ncol = length(prop), byrow = TRUE), dtype = torch::torch_float32() )
    Y <- torch::torch_tensor(Y, dtype = torch::torch_long())

  }  else if(!is.function(loss_obj$call) && any(loss_obj$call %in% c("multinomial", "clogit" ))) {

    if(ncol(Y) > 1.5) {
      Y_base = torch::torch_tensor(matrix(colMeans(Y), nrow = nrow(Y), ncol = ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
      Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
    } else {

      if(is.character(Y)) {
        if (is.null(ylvls)) {
          Y <- factor(Y[,1])
          ylvls <- levels(Y)
        } else {
          Y <- factor(Y[,1], levels = ylvls)
        }
        Y <- torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32())

      } else {
        Y <- as.integer(Y[,1])
        Y <- torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32())
      }
      y_dim <- ncol(Y)
      YY = apply(as.matrix(Y), 1, which.max)

      prop <- as.vector(table(YY)/sum(table(YY)))
      Y_base <- torch::torch_tensor( matrix(prop, nrow = nrow(Y), ncol = length(prop), byrow = TRUE), dtype = torch::torch_float32() )
    }

  } else {
    y_dim <- ncol(Y)
    Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
    Y_base = torch::torch_tensor(matrix(apply(as.matrix(Y), 2, mean), nrow(Y), ncol(Y), byrow = TRUE))
  }

  if(!is.null(ylvls)) responses <- ylvls

  return(list(Y=Y, Y_base=Y_base, y_dim=y_dim, ylvls=ylvls, responses=responses))
}


get_data_loader = function(..., batch_size=25L, shuffle=TRUE, from_folder = FALSE) {

  if(from_folder) ds = dataset_folder(...)
  else ds <- torch::tensor_dataset(...)

  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = TRUE)

  return(dl)
}

dataset_folder = torch::dataset(
  initialize = function(...) {
    self$inputs = list(...)
  },
  .getbatch = function(index) {

    batch = lapply(self$inputs, function(x) {
      if(inherits(x, "torch_tensor")) {
        return(x[index])
      } else {
        if(any(grepl(".png", x)) | any(grepl(".jpeg", x))) {
          X = lapply(x[index], function(p) torch::torch_tensor(torchvision::base_loader(p), torch::torch_float32())$unsqueeze(1L))
        }
        if(any(grepl(".tiff", x))) {
          X = lapply(x[index], function(p) torch::torch_tensor(tiff::readTIFF(x), torch::torch_float32())$unsqueeze(1L))
        }
        X = torch::torch_cat(X, dim = 1L)/255.
        X = X$permute(c(1, 4, 2, 3))
        return(X)
      }
    })
    return(batch)
  },
  .length = function() {
    # this class is only used when there is at least one folder/paths...search for it an use it to infer the length
    for(x in self$inputs) {
      if(any(is.character(x))) return(length(x))
    }
  }
)

# ds = do.call(dataset_folder, list(list.files(path = "test_folder/", full.names = T), torch_rand(1000), torch_rand(1000, 20, 10)))
# DL = torch::dataloader(ds, batch_size = 20, shuffle = TRUE)
# k = coro::collect(DL, 1)
# k

#' Multinomial log likelihood
#'
#' @param probs probabilities
#' @param value observed values
#'
#' Multinomial log likelihood
#'
#' @export
multinomial_log_prob = function(probs, value) {
  logits = probs$log()
  log_factorial_n = torch::torch_lgamma(value$sum(-1) + 1)
  log_factorial_xs = torch::torch_lgamma(value + 1)$sum(-1)
  logits[(value == 0) & (logits == -Inf)] = 0
  log_powers = (logits * value)$sum(-1)
  return(log_factorial_n - log_factorial_xs + log_powers)
}

# binomial_log_prob = function(probs, value, total_count) {
#   log_factorial_n = torch_lgamma(total_count + 1)
#   log_factorial_k = torch_lgamma(value + 1)
#   log_factorial_nmk = torch_lgamma(total_count - value + 1)
#
#   normalize_term = (
#     self.total_count * _clamp_by_zero(self.logits)
#     + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
#     - log_factorial_n
#   )
#   return (
#     value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
#   )
#
# }

get_loss_new <- function(loss, Y, custom_parameters) {
  out <- list()
  if(is.character(loss)) loss <- tolower(loss)
  if(!inherits(loss, "family") & is.character(loss)) {
    loss <- switch(loss,
                   "gaussian" = stats::gaussian(),
                   "binomial" = stats::binomial(),
                   "poisson" = stats::poisson(),
                   loss)
  }

  if(is.character(loss) && loss == "softmax") {
    warning("loss = 'softmax' is deprecated and will be removed in a future version of 'cito'. Please use loss = 'cross-entropy' instead.")
    loss <- "cross-entropy"
  }

  if(is.function(loss)) {
    create_loss <- torch::nn_module(
      classname = "custom loss",
      initialize = function() {
        if(is.null(formals(loss)$pred) | is.null(formals(loss)$true)) stop("The custom loss function has to take two arguments: \"pred\" and \"true\"")

        Y <- as.matrix(Y)
        self$y_dim = ncol(Y)
        self$responses = colnames(Y)

        self$loss.fkt = loss

        if(!is.null(custom_parameters)) {
          for (name in names(custom_parameters)) {
            self[[name]] <- torch::nn_parameter(torch::torch_tensor(custom_parameters[[name]], dtype = torch::torch_float32(), requires_grad = TRUE))
          }
        }

        list2env(self$parameters, envir = environment(fun=self$loss.fkt))
      },
      forward = function(pred, true) {return(self$loss.fkt(pred=pred, true=true))},
      link = function(x) {x},
      invlink = function(x) {x},
      format_Y = function(Y) {
        return(torch::torch_tensor(as.matrix(Y), dtype = torch::torch_float32()))
      }
    )
  } else if(inherits(loss, "family")) {
    if(loss$family == "gaussian") {
      create_loss <- torch::nn_module(
        classname = "gaussian loss",
        initialize = function() {
          checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                            checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                            checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

          if(is.vector(Y)) {
            self$y_dim <- 1
            self$responses <- NULL
          } else {
            self$y_dim <- ncol(Y)
            self$responses <- colnames(Y)
          }

          self$parameter <- torch::nn_parameter(torch::torch_ones(1, self$y_dim, requires_grad = TRUE))

          if(loss$link != "identity") warning(paste0("Link '", loss$link, "' is not implemented for gaussian loss, yet. Using 'identity' link instead."))
          self$link = function(x) {x}
          self$invlink = function(x) {x}

        },
        forward = function(pred, true) {
          return(torch::distr_normal(self$invlink(pred), torch::torch_clamp(self$parameter, 0.0001, 20))$log_prob(true)$negative())
        },
        format_Y = function(Y) {
          checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                            checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                            checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

          Y <- as.matrix(Y)
          if(self$y_dim != ncol(Y)) {
            if (self$y_dim == 1) stop("Wrong dimension of provided target data. Model expects target data to be provided as a numerical vector or numerical matrix/data.frame with 1 column.")
            else stop(paste0("Wrong dimension of provided target data. Model expects target data to be provided as a numerical matrix/data.frame with ",self$y_dim," columns."))
          }

          return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
        }
      )
    } else if(loss$family == "poisson") {
      create_loss <- torch::nn_module(
        classname = "poisson loss",
        initialize = function() {
          checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                            checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                            checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

          if(is.vector(Y)) {
            self$y_dim <- 1
            self$responses <- NULL
          } else {
            self$y_dim <- ncol(Y)
            self$responses <- colnames(Y)
          }

          if(loss$link == "identity") {
            self$link = function(x) {x}
            self$invlink = function(x) {x}
          } else {
            if(loss$link != "log") warning(paste0("Link '", loss$link, "' is not implemented for poisson loss, yet. Using 'log' link instead."))
            self$link = function(x) {log(x)}
            self$invlink = function(x) {torch::torch_exp(x)}
          }
        },
        forward = function(pred, true) {
          return(torch::distr_poisson(self$invlink(pred))$log_prob(true)$negative())
        },
        format_Y = function(Y) {
          checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                            checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                            checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

          Y <- as.matrix(Y)
          if(self$y_dim != ncol(Y)) {
            if (self$y_dim == 1) stop("Wrong dimension of provided target data. Model expects target data to be provided as a numerical vector or numerical matrix/data.frame with 1 column.")
            else stop(paste0("Wrong dimension of provided target data. Model expects target data to be provided as a numerical matrix/data.frame with ",self$y_dim," columns."))
          }

          return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
        }
      )
    } else if(loss$family == "binomial") {
      create_loss <- torch::nn_module(
        classname = "binomial loss",
        initialize = function() {
          checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                            checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                            checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                            checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                            checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, ncols = 2),
                            checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, ncols = 2))

          if(is.factor(Y)) {
            self$control_level <- levels(Y)[1]
          } else if(is.vector(Y)) {
            self$control_level <- levels(factor(Y))[1]
          } else if(is.matrix(Y)) {
            if(is.character(Y)) self$control_level <- levels(factor(Y))[1]
            else self$control_level <- NULL
          } else if(is.data.frame(Y)) {
            if(is.character(Y[,1])) self$control_level <- levels(factor(Y[,1]))[1]
            else self$control_level <- NULL
          }

          self$y_dim = 1
          if(is.null(self$control_level)) self$responses <- "p(success)"
          else self$responses <- paste0("p(NOT ", self$control_level, ")")

          if(loss$link == "probit")  {
            self$link = function(x) {torch::torch_tensor(stats::binomial("probit")$linkfun(as.matrix(x$cpu())), dtype = torch::torch_float32())}
            self$invlink = function(x) {torch::torch_sigmoid(x*1.7012)}
          } else {
            if(loss$link != "logit") warning(paste0("Link '", loss$link, "' is not implemented for binomial loss, yet. Using 'logit' link instead."))
            self$link = function(x) {torch::torch_tensor(stats::binomial("logit")$linkfun(as.matrix(x$cpu())), dtype = torch::torch_float32())}
            self$invlink = function(x) {torch::torch_sigmoid(x)}
          }
        },
        forward = function(pred, true) {
          n = torch::torch_sum(true, 2)
          s = true[, 1]$squeeze(dim=2)
          f = true[, 2]$squeeze(dim=2)
          p = self$invlink(pred)

          nll = - torch::torch_lgamma(n + 1) + torch::torch_lgamma(s + 1) + torch::torch_lgamma(f + 1) - s * torch::torch_log(p) - f * torch::torch_log(1 - p)
          # nll = nll/n # normalize
          return(nll)
        },
        format_Y = function(Y) {
          checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                            checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                            checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                            checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                            checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, ncols = 2),
                            checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, ncols = 2))

          if(is.factor(Y) || is.vector(y) || ncol(Y)==1) {
            if(is.null(self$control_level)) stop("Model expects target data to be provided as integerish matrix/data.frame with 2 columns (first column: #successes, second column: #failures).")
            Y <- as.integer(Y != self$control_level)
            Y <- cbind(Y,Y-1)
          }

          return(torch::torch_tensor(as.matrix(Y), dtype = torch::torch_float32()))
        }
      )
    } else {
      stop(paste0("Family '", loss$family,"' not supported."))
    }
  } else if(loss=="cross-entropy") {
    create_loss <- torch::nn_module(
      classname = "cross-entropy loss",
      initialize = function() {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1))

        if(is.factor(Y)) {
          if(length(levels(Y)) != length(unique(Y))) warning("The provided factor containing the labels has levels with zero occurences. Make sure this is intended, as for each level a node in the output layer will be created.")
        } else {
          Y <- factor(Y)
        }

        self$responses <- levels(Y)
        self$y_dim <- length(levels(Y))

        # prop <- as.vector(table(Y)/sum(table(Y)))
        # Y_base <- torch::torch_tensor(matrix(prop, nrow = length(Y), ncol = length(levels(Y)), byrow = TRUE), dtype = torch::torch_float32())
        # Y_torch <- torch::torch_tensor(as.matrix(as.integer(Y), ncol=1L), dtype = torch::torch_long())
        # self$baseloss <- self$forward(Y_base, Y_torch)
      },
      forward = function(pred, true) {
        return(torch::nnf_cross_entropy(input = pred, target = true, reduction = "none"))
      },
      link = function(x) {log(x) + log(ncol(x))},
      invlink = function(x) {torch::nnf_softmax(x, dim = 2)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1))
        Y <- factor(Y, self$responses)
        if(anyNA(Y)) stop("Unknown class labels. This probably means that new provided target data (e.g. for continued training) includes class labels that did not exist in the data used for the initial training.\n
                          If this was intended, make sure to provide the target data for the initial training as factor that includes all required class labels as level, even those with zero occurences in the initial data.")
        return(torch::torch_tensor(as.integer(Y), dtype = torch::torch_long()))
      }
    )
  } else if(loss=="mse") {
    create_loss <- torch::nn_module(
      classname = "mean squared error loss",
      initialize = function() {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

        if(is.vector(Y)) {
          self$y_dim <- 1
          self$responses <- NULL
        } else {
          self$y_dim <- ncol(Y)
          self$responses <- colnames(Y)
        }
      },
      forward = function(pred, true) {
        return(torch::nnf_mse_loss(input = pred, target = true, reduction = "none"))
      },
      link = function(x) {x},
      invlink = function(x) {x},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

        Y <- as.matrix(Y)
        if(self$y_dim != ncol(Y)) {
          if (self$y_dim == 1) stop("Wrong dimension of provided target data. Model expects target data to be provided as a numerical vector or numerical matrix/data.frame with 1 column.")
          else stop(paste0("Wrong dimension of provided target data. Model expects target data to be provided as a numerical matrix/data.frame with ",self$y_dim," columns."))
        }

        return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
      }
    )
  } else if(loss=="mae") {
    create_loss <- torch::nn_module(
      classname = "mean absolute error loss",
      initialize = function() {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

        if(is.vector(Y)) {
          self$y_dim <- 1
          self$responses <- NULL
        } else {
          self$y_dim <- ncol(Y)
          self$responses <- colnames(Y)
        }
      },
      forward = function(pred, true) {
        return(torch::nnf_l1_loss(input = pred, target = true, reduction = "none"))
      },
      link = function(x) {x},
      invlink = function(x) {x},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

        Y <- as.matrix(Y)
        if(self$y_dim != ncol(Y)) {
          if (self$y_dim == 1) stop("Wrong dimension of provided target data. Model expects target data to be provided as a numerical vector or numerical matrix/data.frame with 1 column.")
          else stop(paste0("Wrong dimension of provided target data. Model expects target data to be provided as a numerical matrix/data.frame with ",self$y_dim," columns."))
        }

        return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
      }
    )
  } else if(loss == "multinomial") {
    create_loss <- torch::nn_module(
      classname = "multinomial loss",
      initialize = function() {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))

        if(is.factor(Y)) {
          self$responses <- levels(Y)
        } else if(is.vector(Y)) {
          self$responses <- levels(factor(Y))
        } else if(is.matrix(Y)) {
          if(is.character(Y)) self$responses <- levels(factor(Y))
          else self$responses <- colnames(Y)
        } else if(is.data.frame(Y)) {
          if(is.character(Y[,1])) self$responses <- levels(factor(Y[,1]))
          else self$responses <- colnames(Y)
        }

        if(is.null(self$responses)) self$y_dim <- ncol(Y)
        else self$y_dim <- length(responses)
      },
      forward = function(pred, true) {
        n = torch::torch_sum(true, 2)
        p = torch::nnf_softmax(pred, dim = 2)
        log_p = torch::torch_log(p)
        log_p[true == 0 & log_p == -Inf] = 0

        nll = - torch::torch_lgamma(n + 1) + torch::torch_lgamma(true + 1)$sum(2) - (log_p * true)$sum(2)
        #nll = nll/n # normalize
        return(nll)
      },
      link = function(x) {log(x) + log(ncol(x))},
      invlink = function(x) {torch::nnf_softmax(x, dim = 2)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))

        if(is.factor(Y) || is.vector(y) || ncol(Y)==1) {
          if(is.null(self$responses)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns."))
          if(!all(Y %in% self$responses)) stop("Unknown class labels. This probably means that new provided target data (e.g. for continued training) includes class labels that did not exist in the data used for the initial training.\n
                                                If this was intended, make sure to provide the target data for the initial training as factor that includes all required class labels as level, even those with zero occurences in the initial data.")
          Y <- factor(Y, levels = self$responses)
          return(torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32()))

        } else {
          if(!is.null(self$responses) && all(self$responses %in% colnames(Y))) {
            return(torch::torch_tensor(Y[, self$responses], dtype = torch::torch_float32()))
          } else{
            if(self$y_dim != ncol(Y)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns."))
            return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
          }
        }
      }
    )
  } else if(loss == "mvp") {
    create_loss <- torch::nn_module(
      classname = "multivariate probit loss",
      initialize = function() {
        checkmate::assert(checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))
        if(!all(Y %in% c(0,1))) stop("The matrix/data.frame must only contain zeroes and ones.")

        self$y_dim <- ncol(Y)
        self$responses <- colnames(Y)

        df = floor(ncol(Y)/2)
        self$parameter <- torch::nn_parameter(torch::torch_tensor(matrix(stats::runif(ncol(Y)*df, -0.001, 0.001), ncol(Y), df), requires_grad = TRUE))
      },
      forward = function(pred, true) {
        noise = torch::torch_randn(list(100L, nrow(pred), ncol(self$parameter)), device = self$parameter$device)
        E = torch::torch_sigmoid((torch::torch_einsum("ijk, lk -> ijl", list(noise, self$parameter))+pred)*1.7012)*0.999999+0.0000005
        logprob = log(E)*true + log(1.0-E)*(1.0-true)
        logprob = logprob$sum(3)
        maxlogprob = torch::torch_amax(logprob, dim = 1)
        Eprob = (exp(logprob-maxlogprob))$mean(dim = 1)
        return((-log(Eprob) - maxlogprob)$mean())
      },
      link = function(x) {torch::torch_tensor(stats::binomial("probit")$linkfun(as.matrix(x$cpu())), dtype = torch::torch_float32())},
      invlink = function(x) {torch::torch_sigmoid(x*1.7012)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))
        if(!all(Y %in% c(0,1))) stop("The matrix/data.frame must only contain zeroes and ones.")
        if(!is.null(self$responses) && all(self$responses %in% colnames(Y))) {
          return(torch::torch_tensor(Y[, self$responses], dtype = torch::torch_float32()))
        } else{
          if(self$y_dim != ncol(Y)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns."))
          return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
        }
      }
    )
  } else if (loss == "clogit") {
    create_loss <- torch::nn_module(
      classname = "conditional binomial loss",
      initialize = function() {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))

        if(is.factor(Y)) {
          self$responses <- levels(Y)
        } else if(is.vector(Y)) {
          self$responses <- levels(factor(Y))
        } else if(is.matrix(Y)) {
          if(is.character(Y)) self$responses <- levels(factor(Y))
          else {
            if(!all(Y %in% c(0,1))) stop("The matrix/data.frame must only contain zeroes and ones.")
            self$responses <- colnames(Y)
          }
        } else if(is.data.frame(Y)) {
          if(is.character(Y[,1])) self$responses <- levels(factor(Y[,1]))
          else {
            if(!all(Y %in% c(0,1))) stop("The matrix/data.frame must only contain zeroes and ones.")
            self$responses <- colnames(Y)
          }
        }

        if(is.null(self$responses)) self$y_dim <- ncol(Y)
        else self$y_dim <- length(responses)
      },
      forward = function(pred, true) {
        return(torch::distr_bernoulli(probs = torch::nnf_softmax(pred, dim = 2))$log_prob(true)$negative())
      },
      link <- function(x) {log(x) + log(ncol(x))},
      invlink <- function(x) {torch::nnf_softmax(x, dim = 2)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F, empty.levels.ok = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))

        if(is.factor(Y) || is.vector(y) || ncol(Y)==1) {
          if(is.null(self$responses)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns."))
          if(!all(Y %in% self$responses)) stop("Unknown class labels. This probably means that new provided target data (e.g. for continued training) includes class labels that did not exist in the data used for the initial training.\n
                                                If this was intended, make sure to provide the target data for the initial training as factor that includes all required class labels as level, even those with zero occurences in the initial data.")
          Y <- factor(Y, levels = self$responses)
          return(torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32()))
        } else {
          if(!all(Y %in% c(0,1))) stop("The matrix/data.frame must only contain zeroes and ones.")
          if(!is.null(self$responses) && all(self$responses %in% colnames(Y))) {
            return(torch::torch_tensor(Y[, self$responses], dtype = torch::torch_float32()))
          } else{
            if(self$y_dim != ncol(Y)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns."))
            return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
          }
        }
      }
    )
  } else if(loss == "nbinom") {
    create_loss <- torch::nn_module(
      classname = "negative binomial loss",
      initialize = function() {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

        if(is.vector(Y)) {
          self$y_dim <- 1
          self$responses <- NULL
          self$parameter = torch::nn_parameter(torch::torch_tensor(0.5, requires_grad=TRUE))
        } else {
          self$y_dim <- ncol(Y)
          self$responses <- colnames(Y)
          self$parameter = torch::nn_parameter(torch::torch_tensor(rep(0.5, ncol(Y)), requires_grad=TRUE))
        }
      },
      forward = function(pred, true) {
        eps = 0.0001
        pred = torch::torch_exp(pred)
        theta_tmp = 1.0/(torch::nnf_softplus(self$parameter)+eps)
        probs = torch::torch_clamp(1.0 - theta_tmp/(theta_tmp+pred), 0.0+eps, 1.0-eps)
        total_count = theta_tmp
        value = true
        logits = torch::torch_log(probs) - torch::torch_log1p(-probs)
        log_unnormalized_prob <- total_count * torch::torch_log(torch::torch_sigmoid(-logits)) + value * torch::torch_log(torch::torch_sigmoid(logits))
        log_normalization <- -torch::torch_lgamma(total_count + value) + torch::torch_lgamma(1 + value) + torch::torch_lgamma(total_count)
        log_normalization <- torch::torch_where(total_count + value == 0, torch::torch_tensor(0, dtype = log_normalization$dtype, device = self$parameter$device), log_normalization)
        return(-(log_unnormalized_prob - log_normalization))
      },
      link = function(x) {log(x)},
      invlink = function(x) {torch::torch_exp(x)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

        Y <- as.matrix(Y)
        if(self$y_dim != ncol(Y)) {
          if (self$y_dim == 1) stop("Wrong dimension of provided target data. Model expects target data to be provided as a numerical vector or numerical matrix/data.frame with 1 column.")
          else stop(paste0("Wrong dimension of provided target data. Model expects target data to be provided as a numerical matrix/data.frame with ",self$y_dim," columns."))
        }

        return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
      },
      simulate = function(pred) { #Not used atm
        theta_tmp = as.numeric((1.0/(torch::nnf_softplus(self$parameter)+0.0001))$cpu)
        probs = 1.0 - theta_tmp/(theta_tmp + pred)
        total_count = theta_tmp

        if(is.matrix(pred)) {
          sim = sapply(1:ncol(pred), function(i) {
            logits = log(probs[,i]) - log1p(-probs[,i])
            stats::rpois(length(logits), exp(-logits))
            return( stats::rpois(length(logits), stats::rgamma(length(logits),total_count[i], exp(- logits ))) )
          })
        } else {
          logits = log(probs) - log1p(-probs)
          stats::rpois(length(pred), exp(-logits))
          sim = stats::rpois(length(pred), stats::rgamma(length(pred),total_count, exp(- logits )))
        }
        return(sim)
      }
    )
  } else {
    stop("Loss not implemented.")
  }

  return(create_loss())
}


get_loss <- function(loss, device = "cpu", X = NULL, Y = NULL) {

  out <- list()
  out$parameter <- NULL

  if(is.character(loss)) loss <- tolower(loss)
  if(!inherits(loss, "family")& is.character(loss)) {
    loss <- switch(loss,
                   "gaussian" = stats::gaussian(),
                   "binomial" = stats::binomial(),
                   "poisson" = stats::poisson(),
                   loss)
  }

  if(inherits(loss, "family")){
    if(loss$family == "gaussian") {
      out$parameter <- torch::torch_tensor(1.0, requires_grad = TRUE, device = device)
      out$parameter_r = as.numeric(out$parameter$cpu())
      out$invlink <- function(a) a
      out$link <- function(a) a
      out$loss <- function(pred, true) {
        return(torch::distr_normal(pred, torch::torch_clamp(out$parameter, 0.0001, 20))$log_prob(true)$negative())
      }
    } else if(loss$family == "binomial") {
      if(loss$link == "logit") {
        out$invlink <- function(a) torch::torch_sigmoid(a)
        out$link <- function(a) stats::binomial("logit")$linkfun(as.matrix(a))
      } else if(loss$link == "probit")  {
        out$invlink <- function(a) torch::torch_sigmoid(a*1.7012)
        out$link <- function(a) stats::binomial("probit")$linkfun(as.matrix(a))
      } else {
        out$invlink <- function(a) a
        out$link <- function(a) a
      }
      out$loss <- function(pred, true) {
        return(torch::distr_bernoulli(probs = out$invlink(pred))$log_prob(true)$negative())
      }
    } else if(loss$family == "poisson") {
      if(loss$link == "log") {
        out$invlink <- function(a) torch::torch_exp(a)
        out$link <- function(a) log(a)
      } else {
        out$invlink <- function(a) a
        out$link <- function(a) a
      }
      out$loss <- function(pred, true) {
        return(torch::distr_poisson( out$invlink(pred) )$log_prob(true)$negative())
      }
    } else { stop("family not supported")}
  } else  if (is.function(loss)){
    if(is.null(formals(loss)$pred) | is.null(formals(loss)$true)){
      stop("loss function has to take two arguments, \"pred\" and \"true\"")
    }
    out$loss <- loss
    out$invlink <- function(a) a
    out$link <- function(a) a
  } else {
    if(loss == "mae"){
      out$invlink <- function(a) a
      out$link <- function(a) a
      out$loss <- function(pred, true) return(torch::nnf_l1_loss(input = pred, target = true))
    }else if(loss == "mse"){
      out$invlink <- function(a) a
      out$link <- function(a) a
      out$loss <- function(pred,true) return(torch::nnf_mse_loss(input= pred, target = true))
    }else if(loss == "softmax" | loss == "cross-entropy") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$link <- function(a) log(a) + log(ncol(a))
      out$loss <- function(pred, true) {
        return(torch::nnf_cross_entropy(pred, true$squeeze(dim = 2), reduction = "none"))
      }
    } else if(loss == "mvp") {

      if(!exists("Y")) Y = matrix(1., 1,1)

      df = floor(ncol(Y)/2)
      out$parameter <- torch::torch_tensor(matrix(stats::runif(ncol(Y)*df, -0.001, 0.001), ncol(Y), df), requires_grad = TRUE, device = device)
      out$invlink <- function(a) torch::torch_sigmoid(a*1.7012)
      out$link <- function(a) stats::binomial("probit")$linkfun(as.matrix(a$cpu()))
      out$loss <- function(pred, true) {
        sigma = out$parameter
        Ys = true
        df = ncol(sigma)
        noise = torch::torch_randn(list(100L, nrow(pred), df), device = device)
        E = torch::torch_sigmoid((torch::torch_einsum("ijk, lk -> ijl", list(noise, sigma))+pred)*1.702)*0.999999+0.0000005
        logprob = -(log(E)*Ys + log(1.0-E)*(1.0-Ys))
        logprob = - logprob$sum(3)
        maxlogprob = torch::torch_amax(logprob, dim = 1)
        Eprob = (exp(logprob-maxlogprob))$mean(dim = 1)
        return((-log(Eprob) - maxlogprob)$mean())
      }
    } else if(loss == "nbinom") {

      if(is.matrix(Y)) out$parameter = torch::torch_tensor(rep(0.5, ncol(Y)), requires_grad=TRUE, device = device)
      else out$parameter = torch::torch_tensor(0.5, requires_grad=TRUE, device = device)
      out$parameter_r = as.numeric(out$parameter$cpu())
      out$invlink <- function(a) torch::torch_exp(a)
      out$link <- function(a) log(as.matrix(a$cpu()))
      out$parameter_link = function() {
        out$parameter = re_init(out$parameter, out$parameter_r)
        as.numeric((1.0/(torch::nnf_softplus(out$parameter)+0.0001))$cpu())
      }
      out$simulate = function(pred) {
        theta_tmp = out$parameter_link()
        probs = 1.0 - theta_tmp/(theta_tmp + pred)
        total_count = theta_tmp

        if(is.matrix(pred)) {
          sim = sapply(1:ncol(pred), function(i) {
            logits = log(probs[,i]) - log1p(-probs[,i])
            stats::rpois(length(logits), exp(-logits))
            return( stats::rpois(length(logits), stats::rgamma(length(logits),total_count[i], exp(- logits ))) )
          })
        } else {
          logits = log(probs) - log1p(-probs)
          stats::rpois(length(pred), exp(-logits))
          sim = stats::rpois(length(pred), stats::rgamma(length(pred),total_count, exp(- logits )))
        }
        return(sim)
      }

      out$loss = function(pred, true) {
        eps = 0.0001
        pred = torch::torch_exp(pred)
        if(pred$device$type != out$parameter$device$type) pred = pred$to(device = out$parameter$device)
        theta_tmp = 1.0/(torch::nnf_softplus(out$parameter)+eps)
        probs = torch::torch_clamp(1.0 - theta_tmp/(theta_tmp+pred), 0.0+eps, 1.0-eps)
        total_count = theta_tmp
        value = true
        logits = torch::torch_log(probs) - torch::torch_log1p(-probs)
        log_unnormalized_prob <- total_count * torch::torch_log(torch::torch_sigmoid(-logits)) + value * torch::torch_log(torch::torch_sigmoid(logits))
        log_normalization <- -torch::torch_lgamma(total_count + value) + torch::torch_lgamma(1 + value) + torch::torch_lgamma(total_count)
        log_normalization <- torch::torch_where(total_count + value == 0, torch::torch_tensor(0, dtype = log_normalization$dtype, device = out$parameter$device), log_normalization)
        return( - (log_unnormalized_prob - log_normalization))
      }

    } else if(loss == "multinomial") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$link <- function(a) log(a) + log(ncol(a))
      out$loss <- function(pred, true) {
        return(multinomial_log_prob(torch::nnf_softmax(pred, dim = 2), true)$negative())
      }
    } else if(loss == "clogit") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$link <- function(a) log(a) + log(ncol(a))
      out$loss <- function(pred, true) {
        # return(binomial_log_prob(torch::nnf_softmax(pred, dim = 2), true))
        return(torch::distr_bernoulli(probs = torch::nnf_softmax(pred, dim = 2))$log_prob(true)$negative())
      }
    }else{
      cat( "unidentified loss \n")
    }

  }
  out$call <- loss

  return(out)
}

get_activation_layer <- function(activation) {
  return(switch(tolower(activation),
                "relu" = torch::nn_relu(),
                "leaky_relu" = torch::nn_leaky_relu(),
                "tanh" = torch::nn_tanh(),
                "elu" = torch::nn_elu(),
                "rrelu" = torch::nn_rrelu(),
                "prelu" = torch::nn_prelu(),
                "softplus" = torch::nn_softplus(),
                "celu" = torch::nn_celu(),
                "selu" = torch::nn_selu(),
                "gelu" = torch::nn_gelu(),
                "relu6" = torch:: nn_relu6(),
                "sigmoid" = torch::nn_sigmoid(),
                "softsign" = torch::nn_softsign(),
                "hardtanh" = torch::nn_hardtanh(),
                "tanhshrink" = torch::nn_tanhshrink(),
                "softshrink" = torch::nn_softshrink(),
                "hardshrink" = torch::nn_hardshrink(),
                "log_sigmoid" = torch::nn_log_sigmoid(),
                stop(paste0(activation, " as an activation function is not supported"))
  ))
}

get_var_names <- function(formula, data){
  X_helper <- stats::model.matrix(formula,data[1,])
  var_names <- c()
  for(i in seq_len(ncol(data))){
    if(colnames(data)[i]%in%colnames(X_helper)){
      var_names<- append(var_names, colnames(data)[i])

    }else if (is.factor(data[,i])){
      count <- startsWith(colnames(X_helper),colnames(data)[i])
      count <- sum(count, na.rm = TRUE) + 1
      if(count >= nlevels(data[,i])){
        var_names<- append(var_names, colnames(data)[i])

      }
    }
  }
  return(var_names)
}

get_output_shape <- function(input_shape, n_kernels, kernel_size, stride, padding, dilation) {
  input_shape[1] <- n_kernels
  for(i in 2:length(input_shape)) {
    l <- input_shape[i] + 2*padding[i-1]
    k <- kernel_size[i-1] + (kernel_size[i-1]-1)*(dilation[i-1]-1)
    s <- stride[i-1]
    input_shape[i] <- floor((l-k)/s)+1
  }
  return(input_shape)
}


adjust_architecture <- function(architecture, input_dim) {

  adjusted_architecture <- list()
  for(layer in architecture) {
    if(class(layer)[1] %in% c("avgPool", "maxPool")) {
      if(is.null(layer$stride)) layer$stride <- layer$kernel_size
    }

    if(input_dim != 1) {
      if(class(layer)[1] %in% c("conv", "avgPool", "maxPool")) {
        if(length(layer$kernel_size) == 1) layer$kernel_size <- rep(layer$kernel_size, input_dim)
        if(length(layer$stride) == 1) layer$stride <- rep(layer$stride, input_dim)
        if(length(layer$padding) == 1) layer$padding <- rep(layer$padding, input_dim)
      }

      if(class(layer)[1] %in% c("conv", "maxPool")) {
        if(length(layer$dilation) == 1) layer$dilation <- rep(layer$dilation, input_dim)
      }
    }
    adjusted_architecture <- append(adjusted_architecture, list(layer))
  }
  class(adjusted_architecture) <- "citoarchitecture"
  return(adjusted_architecture)
}

#Output shapes of the avgpool layers right before the classifier
get_transfer_output_shape <- function(name) {
  return(switch(name,
                "alexnet" = c(256, 6, 6),
                "inception_v3" = c(2048, 1, 1),
                "mobilenet_v2" = c(1280, 1, 1),
                "resnet101" = c(2048, 1, 1),
                "resnet152" = c(2048, 1, 1),
                "resnet18" = c(512, 1, 1),
                "resnet34" = c(512, 1, 1),
                "resnet50" = c(2048, 1, 1),
                "resnext101_32x8d" = c(2048, 1, 1),
                "resnext50_32x4d" = c(2048, 1, 1),
                "vgg11" = c(512, 7, 7),
                "vgg11_bn" = c(512, 7, 7),
                "vgg13" = c(512, 7, 7),
                "vgg13_bn" = c(512, 7, 7),
                "vgg16" = c(512, 7, 7),
                "vgg16_bn" = c(512, 7, 7),
                "vgg19" = c(512, 7, 7),
                "vgg19_bn" = c(512, 7, 7),
                "wide_resnet101_2" = c(2048, 1, 1),
                "wide_resnet50_2" = c(2048, 1, 1),
                stop(paste0(name, " not supported."))))
}

# Load the pretrained models.
# In inception_v3 the auxiliary part is omitted since we don't use it and the input transformation is moved to the forward function (if pretrained=TRUE)
# In mobilenet_v2 the global average pool is moved from the forward function to a module, so the last 2 modules of all models are avgpool and classifier, respectively.
get_pretrained_model <- function(transfer, pretrained, rgb) {
  if(transfer == "inception_v3") {
    inception_v3 <- torchvision::model_inception_v3(pretrained = pretrained)


    forward <- deparse(inception_v3$.forward)[1:2]
    if(pretrained & rgb) {
      forward <- c(forward, deparse(inception_v3$.transform_input)[c(4:9)], "        x <- torch::torch_cat(list(x_ch0, x_ch1, x_ch2), 2)")
    }

    forward <- c(forward, deparse(inception_v3$.forward)[c(3:17, 24:30)], "    x", "}")

    torch_model <- torch::nn_module(
      classname = "inception_v3",
      initialize = function(inception_v3) {
        for (child in names(inception_v3$children)) {
          if(child != "AuxLogits") {
            eval(parse(text=paste0("self$", child, " <- inception_v3$", child)))
          }
        }
      },
      forward = eval(parse(text=forward)),
      transform_input = pretrained
    )(inception_v3)

  } else if(transfer == "mobilenet_v2") {
    mobilenet_v2 <- torchvision::model_mobilenet_v2(pretrained = pretrained)

    forward <- deparse(mobilenet_v2$forward)
    forward[4] <- "    x <- self$avgpool(x)"

    torch_model <- torch::nn_module(
      classname = "mobilenet_v2",
      initialize = function(mobilenet_v2) {
        self$features <- mobilenet_v2$features
        self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
        self$classifier <- mobilenet_v2$classifier
      },
      forward = eval(parse(text=forward))
    )(mobilenet_v2)

  } else {
    eval(parse(text = paste0("torch_model <- torchvision::model_", transfer, "(pretrained = pretrained)")))
  }
  return(torch_model)
}

replace_first_conv_layer <- function(torch_model, in_channels) {
  tmp <- "torch_model"
  while(TRUE) {
    eval(parse(text = paste0("children <- names(", tmp, "$children)")))
    if(is.null(children)) stop("Could not find first convolutional layer of pretrained model. Pls report this to the developers of 'cito'.")
    tmp <- paste0(tmp, "$'", children[1], "'")
    module <- eval(parse(text = tmp))
    if(inherits(module, "nn_conv2d")) {
      out_channels <- module$out_channels
      kernel_size <- module$kernel_size
      stride <- module$stride
      padding <- module$padding
      dilation <- module$dilation
      bias <- ifelse(is.null(module$bias), FALSE, TRUE)
      groups <- module$groups
      padding_mode = module$padding_mode

      new_layer <- torch::nn_conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding,
                                    dilation = dilation,
                                    groups = groups,
                                    bias = bias,
                                    padding_mode = padding_mode)

      # Initialize bias
      if(bias) new_layer$bias$set_data(module$bias)
      # Initialize weights
      avg_weights <- torch::torch_mean(module$weight, dim = 2L, keepdim = TRUE)
      new_layer$weight$set_data(avg_weights$'repeat'(c(1,in_channels,1,1)))

      eval(parse(text = paste0(tmp, "<- new_layer")))

      break
    }
  }
}

replace_classifier <- function(transfer_model, cito_model) {

  forward <- deparse(transfer_model$forward)
  forward <- c(forward[1:(which(grepl("flatten", forward))-1)], "    x <- self$classifier(x)", "    x", "}")

  net <- torch::nn_module(
    initialize = function(transfer_model, cito_model) {
      for (child in names(transfer_model$children)[-length(transfer_model$children)]) {
        eval(parse(text=paste0("self$", child, " <- transfer_model$", child)))
      }
      self$classifier <- cito_model
    },
    forward = eval(parse(text=forward))
  )

  return(net(transfer_model, cito_model))
}

freeze_weights <- function(transfer_model) {
  for(parameter in transfer_model$parameters) {
    parameter$requires_grad_(FALSE)
  }
  return(transfer_model)
}




re_init = function(param, param_r) {
  pointer_check <- tryCatch(torch::as_array(param), error = function(e) e)

  if(inherits(pointer_check,"error")){
    param = torch::torch_tensor(param_r)
  }
  return(param)
}


# check if model is loaded and if current parameters are the desired ones
check_model <- function(object) {

  if(!inherits(object, c("citodnn", "citocnn", "citommn"))) stop("model not of class citodnn, citocnn or citommn")

  pointer_check <- tryCatch(torch::as_array(object$net$parameters[[1]]), error = function(e) e)
  if(inherits(pointer_check,"error")){
    object$net <- build_model(object)
    object$loaded_model_epoch <- torch::torch_tensor(0)
    object$loss<- get_loss(object$loss$call)
  }


  if(as.numeric(object$loaded_model_epoch)!= object$use_model_epoch){

    set_data <- function(params_buffers_names, mode) {
      module_name <- sapply(params_buffers_names, function(x) {
        period_indices <- which(strsplit(x,"")[[1]]==".")
        last_period_index <- period_indices[length(period_indices)]
        substr(x,1,last_period_index-1)
      })

      module_type <- sapply(params_buffers_names, function(x) {
        period_indices <- which(strsplit(x,"")[[1]]==".")
        last_period_index <- period_indices[length(period_indices)]
        substring(x,last_period_index+1)
      })

      if(mode == 1) {
        text1 <- "parameters"
        text2 <- "weights"
      } else {
        text1 <- "buffers"
        text2 <- "buffers"
      }

      for ( i in names(object$net$modules)){
        if(i %in% module_name){
          k<- which(i == module_name)
          sapply(k, function(x) eval(parse(text=paste0("object$net$modules$`",i,"`$",text1,"$",module_type[k],"$set_data(object$",text2,"[[object$use_model_epoch]]$`",params_buffers_names[k],"`)"))))
        }
      }
    }

    module_params <- names(object$weights[[object$use_model_epoch]])
    if(!is.null(module_params)) set_data(module_params, mode = 1)
    module_buffers <- names(object$buffers[[object$use_model_epoch]])
    if(!is.null(module_buffers)) set_data(module_buffers, mode = 2)

    object$loaded_model_epoch$set_data(object$use_model_epoch)
  }

  if(!is.null(object$parameter)) object$loss$parameter <- lapply(object$parameter, torch::torch_tensor)

  object$net$eval()

  return(object)
}

check_call_config <- function(mc, variable ,standards, dim = 1, check_var = FALSE, verbose = FALSE){
  value <- NULL
  if(variable %in% names(mc)){
    if(dim ==1){
      eval(parse(text = paste0("value  <- mc$",variable)))
    }else{
      eval(parse(text= paste0("value <- tryCatch(as.numeric(eval(mc$",variable,")), error = function(err)
              print(\"must be numeric input\")) ")))
    }

    if(!isFALSE(check_var)) checkmate::qassert(value,check_var)

  } else{
    value <- unlist(standards[which(names(standards) == variable)])
  }

  if(verbose) cat( paste0(variable,": [", paste(value, collapse = ", "),"] \n"))
  return(value)
}


check_listable_parameter <- function(parameter, check, vname = checkmate::vname(parameter)) {
  checkmate::qassert(parameter, c(check, "l+"), vname)
  if(inherits(parameter, "list")) {
    for (i in names(parameter)) {
      checkmate::qassert(parameter[[i]], check, paste0(vname, "$", i))
    }
  }
}

check_device = function(device) {
  if(device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")}
    else{
      warning("No cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  } else if(device == "mps") {
    if (torch::backends_mps_is_available()) {
      device <- torch::torch_device("mps")}
    else{
      warning("No mps device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }
  } else {
    if(device != "cpu") warning(paste0("device ",device," not known, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }
  return(device)
}


# taken and adopted from lme4:::RHSForm
LHSForm = function (form, as.form = FALSE)
{
  rhsf <- form[[2]]
  if (as.form)
    stats::reformulate(deparse(rhsf))
  else rhsf
}


cast_to_r_keep_dim = function(x) {
  d = dim(x)
  if(length(d) == 1) return(as.numeric(x$cpu()))
  else return(as.matrix(x$cpu()))
}


get_X_Y = function(formula, X, Y, data) {

  if(!is.null(formula)) old_formula = formula

  if(!is.null(X)) {
    if(!is.null(Y)) {
      if(!is.matrix(Y)) Y <- data.frame(Y)
      if(ncol(Y) == 1) {
        if(is.null(colnames(Y))) colnames(Y) <- "Y"
        formula <- stats::formula(paste0(colnames(Y), " ~ ."))
      } else {
        if(is.null(colnames(Y))) colnames(Y) <- paste0("Y", 1:ncol(Y))
        formula <- stats::formula(paste0("cbind(", paste(colnames(Y), collapse=","), ") ~ ."))
      }
      data <- cbind(data.frame(Y), data.frame(X))
    } else {
      formula <- stats::formula("~ .")
      data <- data.frame(X)
    }
    formula <- formula(stats::terms.formula(formula, data = data))
    old_formula = formula

    Specials = NULL
  } else if(!is.null(formula)) {
    if(!is.null(data)) {
      data <- data.frame(data)
    }
    old_formula = formula
    parsed_formula = splitForm(formula)
    formula = parsed_formula$fixedFormula
    Specials = list(terms = parsed_formula$reTrmFormulas, types = parsed_formula$reTrmClasses, args = parsed_formula$reTrmAddArgs)
    formula <- formula(stats::terms.formula(formula, data = data))
    formula <- stats::update.formula(formula, ~ . + 1)
  } else {
    stop("Either formula (and data) or X (and Y) have to be specified.")
  }

  if(!is.null(data)) {
    char_cols <- sapply(data, is.character)
    data[,char_cols] <- lapply(data[,char_cols,drop=F], as.factor)
  }

  tmp <- stats::model.matrix(formula, data)
  X <- tmp[, -1, drop=FALSE]
  attr(X, "assign") <- attr(tmp, "assign")[-1]
  Y <- stats::model.response(stats::model.frame(formula, data))

  if(is.null(Specials$terms)) {
    out = list(X = X, Y = Y, formula = formula, data = data, Z = NULL, Z_terms = NULL)
  } else {
    terms = sapply(Specials$terms, as.character)

    Zlvls =
      sapply(terms, function(i) {
        if(!is.factor(data[,i])) stop("Embeddings must be passed as factor/categorical feature.")
        return(nlevels(data[,i]))
      })
    Z =
      lapply(terms, function(i) {
        return(as.integer(data[,i]))
      })
    Z = do.call(cbind, Z)
    colnames(Z) = terms
    out = list(X = X, Y = Y, formula = formula, data = data, Z = Z, Z_terms = terms, Z_args = Specials$args, Zlvls = Zlvls)
  }
  out$old_formula = old_formula
  return(out)
}


