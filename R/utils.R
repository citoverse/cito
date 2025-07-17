get_data_loader = function(..., batch_size=25L, shuffle=TRUE, from_folder = FALSE, data_augmentation = NULL) {

  if(from_folder | !is.null(data_augmentation)) ds = cito_dataset(..., data_augmentation = data_augmentation)
  else ds <- torch::tensor_dataset(...)

  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = TRUE)

  return(dl)
}

cito_dataset = torch::dataset(
  initialize = function(..., data_augmentation) {
    self$inputs = list(...)
    self$data_augmentation = data_augmentation
  },
  .getbatch = function(index) {

    batch = lapply(self$inputs, function(x) {
      if(inherits(x, "torch_tensor")) {
        return(x[index, drop=FALSE])
      } else {
        X <- lapply(x[index], function(path) {
          if(grepl(".png", path) | grepl(".jpeg", path)) return(self$load_image(path))
          else if(grepl(".tiff", path)) return(self$load_tiff(path))
          else stop(paste0("File format not supported: ", path))
        })
        return(torch::torch_stack(X))
      }
    })

    if(!is.null(self$data_augmentation)) {
      batch <- lapply(batch, function(x) {
        if(x$ndim < 3) return(x)
        x <- torch::torch_cat(lapply(1:(dim(x)[1]), function(i) {
          sample <- x[i, drop=FALSE]
          for(fn in self$data_augmentation) {
            sample <- fn(sample)
          }
          return(sample)
        }))
        return(x)
      })
    }

    return(batch)
  },
  .length = function() {
    if(inherits(self$inputs[[1]], "torch_tensor")) return(dim(self$inputs[[1]])[1])
    else return(length(self$inputs[[1]]))
  },
  load_image = function(path) {
    img <- torch::torch_tensor(torchvision::base_loader(path), torch::torch_float32())
    if(img$ndim == 2) img <- img$unsqueeze(3)
    img <- img$permute(c(3, 1, 2))
    img <- img/255
    return(img)
  },
  load_tiff = function(path) {
    img_list <- tiff::readTIFF(path, all = TRUE)
    img_list <- lapply(img_list, function(img) {
      img <- torch::torch_tensor(img, dtype = torch::torch_float32())
      if(img$ndim == 2) img <- img$unsqueeze(3)
      img <- img$permute(c(3, 1, 2))
    })

    if(length(img_list) == 1) img <- img_list[[1]]
    else img <- torch::torch_stack(img_list, 4) #3D object

    return(img)
  }
)

check_data_augmentation <- function(data_augmentation) {
  for(i in 1:length(data_augmentation)) {
    if(is.character(data_augmentation[[i]])) {
      data_augmentation[[i]] <- switch(data_augmentation[[i]],
                                       "rotate90" = augment_rotate90,
                                       "flip" = augment_flip,
                                       "noise" = augment_noise,
                                       stop(paste0("Data augmentation function '", data_augmentation[[i]], "' not found.")))
    } else if(is.function(data_augmentation[[i]])) {
      args <- formals(data_augmentation[[i]])
      if(length(args) == 0) stop("Data augmentation function must have at least one argument.")
      has_default <- sapply(args, function(x) !is.symbol(x) || deparse(x) != "")
      required_args <- which(!has_default)
      if (length(required_args) > 1) stop("Data augmentation function must have at most one argument without a default.")
      if (length(required_args) == 1 && required_args != 1) stop("The one argument without a default must be the first argument.")
    } else {
      stop("Elements of data_augmentation must be either a function or astring corresponding to one of cito's inbuilt data augmentation functions ('rotate90', 'flip' or 'noise').")
    }
  }
  return(data_augmentation)
}

get_loss <- function(loss, Y, custom_parameters) {
  out <- list()
  if(is.character(loss)) loss <- tolower(loss)
  if(is.character(loss) && loss == "softmax") {
    warning("loss = 'softmax' is deprecated and will be removed in a future version of 'cito'. Please use loss = 'cross-entropy' instead.")
    loss <- "cross-entropy"
  }
  if(!inherits(loss, "family") & is.character(loss)) {
    loss <- switch(loss,
                   "gaussian" = stats::gaussian(),
                   "binomial" = stats::binomial(),
                   "poisson" = stats::poisson(),
                   loss)
  }

  if(is.function(loss)) {
    create_loss <- torch::nn_module(
      classname = "custom loss",
      initialize = function() {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

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

        Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
      },
      forward = function(pred, true) {return(self$loss.fkt(pred=pred, true=true))},
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

          Y <- as.matrix(Y)
          self$y_dim <- ncol(Y)
          self$responses <- colnames(Y)

          self$parameter <- torch::nn_parameter(torch::torch_ones(1, self$y_dim, requires_grad = TRUE)) # might want to rename self$parameter

          if(loss$link != "identity") warning(paste0("Link '", loss$link, "' is not implemented for gaussian loss, yet. Using 'identity' link instead."))
          self$link = function(x) {x}
          self$invlink = function(x) {x}

          Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
          self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
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

          Y <- as.matrix(Y)
          self$y_dim <- ncol(Y)
          self$responses <- colnames(Y)

          if(loss$link == "identity") {
            self$link = function(x) {x}
            self$invlink = function(x) {x}
          } else {
            if(loss$link != "log") warning(paste0("Link '", loss$link, "' is not implemented for poisson loss, yet. Using 'log' link instead."))
            self$link = function(x) {log(x)}
            self$invlink = function(x) {torch::torch_exp(x)}
          }

          Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
          self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
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
                            checkmate::checkFactor(Y, any.missing = F, all.missing = F),
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

          Y_base <- self$format_Y(Y)
          prob <- Y_base[,1]$sum()/Y_base$sum()
          Y_base <- prob$expand(c(dim(Y_base)[1],1))
          self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
        },
        forward = function(pred, true) {
          n = torch::torch_sum(true, 2)
          s = true[, 1]
          f = true[, 2]
          p = self$invlink(pred$squeeze())
          nll = - torch::torch_lgamma(n + 1) + torch::torch_lgamma(s + 1) + torch::torch_lgamma(f + 1) - s * torch::torch_log(p) - f * torch::torch_log(1 - p)
          # nll = nll/n # normalize
          return(nll)
        },
        format_Y = function(Y) {
          checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                            checkmate::checkFactor(Y, any.missing = F, all.missing = F),
                            checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                            checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                            checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, ncols = 2),
                            checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, ncols = 2))

          if(is.factor(Y) || is.vector(Y) || ncol(Y)==1) {
            if(is.null(self$control_level)) stop("Model expects target data to be provided as integerish matrix/data.frame with 2 columns (first column: #successes, second column: #failures).")
            Y <- as.integer(Y != self$control_level)
            Y <- cbind(Y,1-Y)
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
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1))

        if(is.factor(Y)) {
          if(length(levels(Y)) != length(unique(Y))) warning("The provided factor containing the labels has levels with zero occurences. Make sure this is intended, as for each level a node in the output layer will be created.")
        } else if(is.data.frame(Y)) {
          Y <- factor(Y[,1])
        } else {
          Y <- factor(Y)
        }

        self$responses <- levels(Y)
        self$y_dim <- length(levels(Y))

        prob <- as.vector(table(Y)/sum(table(Y)))
        Y_base <- torch::torch_tensor(matrix(prob, nrow = length(Y), ncol = length(levels(Y)), byrow = TRUE), dtype = torch::torch_float32())
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
      },
      forward = function(pred, true) {
        return(torch::nnf_cross_entropy(input = pred, target = true, reduction = "none"))
      },
      link = function(x) {log(x) + log(ncol(x))},
      invlink = function(x) {torch::nnf_softmax(x, dim = 2)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1))

        if(is.data.frame(Y)) Y <- factor(Y[,1], self$responses)
        else Y <- factor(Y, self$responses)

        if(anyNA(Y)) stop("Unknown class labels. This probably means that new provided target data (e.g. for continued training) includes class labels that did not exist in the data used for the initial training.\n
                          If this was intended, make sure to provide the target data for the initial training as factor that includes all required class labels as level, even those with zero occurences in the initial data.")
        return(torch::torch_tensor(as.integer(Y), dtype = torch::torch_long()))
      }
    )
  } else if(loss=="bernoulli") {
    create_loss <- torch::nn_module(
      classname = "bernoulli loss",
      initialize = function() {
        checkmate::assert(checkmate::checkIntegerish(Y, any.missing = F, all.missing = F, lower = 0, upper = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F))

        Y <- as.matrix(Y)
        if(!all(Y %in% c(0,1))) stop("Model expects target data to be provided as integerish vector/matrix/data.frame containing only zeroes and ones.")
        self$y_dim <- ncol(Y)
        self$responses <- colnames(Y)

        Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
      },
      forward = function(pred, true) {
        return(torch::distr_bernoulli(probs = torch::torch_sigmoid(pred))$log_prob(true)$negative())
      },
      link = function(x) {torch::torch_tensor(stats::binomial("logit")$linkfun(as.matrix(x$cpu())), dtype = torch::torch_float32())},
      invlink = function(x) {torch::torch_sigmoid(x)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkIntegerish(Y, any.missing = F, all.missing = F, lower = 0, upper = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F))

        Y <- as.matrix(Y)
        if((ncol(Y) != self$y_dim) || (!all(Y %in% c(0,1)))) {
          if(self$y_dim == 1) stop("Model expects target data to be provided as integerish vector or matrix/data.frame with 1 column, containing only zeroes and ones.")
          else stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
        }

        return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
      }
    )
  } else if(loss=="mse") {
    create_loss <- torch::nn_module(
      classname = "mean squared error loss",
      initialize = function() {
        checkmate::assert(checkmate::checkNumeric(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "numeric", any.missing = F, all.missing = F),
                          checkmate::checkDataFrame(Y, types = "numeric", any.missing = F, all.missing = F))

        Y <- as.matrix(Y)
        self$y_dim <- ncol(Y)
        self$responses <- colnames(Y)

        Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
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

        Y <- as.matrix(Y)
        self$y_dim <- ncol(Y)
        self$responses <- colnames(Y)

        Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
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
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))

        if(is.factor(Y)) {
          if(length(levels(Y)) != length(unique(Y))) warning("The provided factor containing the labels has levels with zero occurences. Make sure this is intended, as for each level a node in the output layer will be created.")
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
        else self$y_dim <- length(self$responses)

        Y_base <- self$format_Y(Y)
        Y_base <- Y_base$mean(1)$expand(dim(Y_base))
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
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
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))

        if(is.factor(Y) || is.vector(Y) || ncol(Y)==1) {
          if(is.null(self$responses)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns."))
          if(is.data.frame(Y)) Y <- factor(Y[,1], levels = self$responses)
          else Y <- factor(Y, levels = self$responses)
          if(anyNA(Y)) stop("Unknown class labels. This probably means that new provided target data (e.g. for continued training) includes class labels that did not exist in the data used for the initial training.\n
                          If this was intended, make sure to provide the target data for the initial training as factor that includes all required class labels as level, even those with zero occurences in the initial data.")

          return(torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32()))
        } else {
          if(!is.null(self$responses) && all(self$responses %in% colnames(Y))) {
            return(torch::torch_tensor(as.matrix(Y[, self$responses]), dtype = torch::torch_float32()))
          } else{
            if(self$y_dim != ncol(Y)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns."))
            return(torch::torch_tensor(as.matrix(Y), dtype = torch::torch_float32()))
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
        Y <- as.matrix(Y)
        if(!all(Y %in% c(0,1))) stop("Model expects target data to be provided as integerish matrix/data.frame containing only zeroes and ones.")

        self$y_dim <- ncol(Y)
        self$responses <- colnames(Y)

        df = floor(ncol(Y)/2)
        self$parameter <- torch::nn_parameter(torch::torch_tensor(matrix(stats::runif(ncol(Y)*df, -0.001, 0.001), ncol(Y), df), requires_grad = TRUE))

        Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
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
        Y <- as.matrix(Y)
        if(!all(Y %in% c(0,1))) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
        if(!is.null(self$responses) && all(self$responses %in% colnames(Y))) {
          return(torch::torch_tensor(Y[, self$responses], dtype = torch::torch_float32()))
        } else{
          if(self$y_dim != ncol(Y)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
          return(torch::torch_tensor(Y, dtype = torch::torch_float32()))
        }
      }
    )
  } else if(loss == "clogit") {
    create_loss <- torch::nn_module(
      classname = "conditional binomial loss",
      initialize = function() {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F),
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
            if(!all(Y %in% c(0,1))) stop("Model expects target data to be provided as factor, character vector, character matrix/data.frame with 1 column or integerish matrix/data.frame containing only zeroes and ones.")
            self$responses <- colnames(Y)
          }
        } else if(is.data.frame(Y)) {
          if(is.character(Y[,1])) self$responses <- levels(factor(Y[,1]))
          else {
            if(!all(as.matrix(Y) %in% c(0,1))) stop("Model expects target data to be provided as factor, character vector, character matrix/data.frame with 1 column or integerish matrix/data.frame containing only zeroes and ones.")
            self$responses <- colnames(Y)
          }
        }

        if(is.null(self$responses)) self$y_dim <- ncol(Y)
        else self$y_dim <- length(self$responses)

        Y_base <- self$format_Y(Y)
        Y_base <- Y_base$mean(1)$expand(dim(Y_base))
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
      },
      forward = function(pred, true) {
        return(torch::distr_bernoulli(probs = torch::nnf_softmax(pred, dim = 2))$log_prob(true)$negative())
      },
      link = function(x) {log(x) + log(ncol(x))},
      invlink = function(x) {torch::nnf_softmax(x, dim = 2)},
      format_Y = function(Y) {
        checkmate::assert(checkmate::checkCharacter(Y, any.missing = F, all.missing = F),
                          checkmate::checkFactor(Y, any.missing = F, all.missing = F),
                          checkmate::checkMatrix(Y, mode = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkDataFrame(Y, types = "character", any.missing = F, all.missing = F, ncols = 1),
                          checkmate::checkMatrix(Y, mode = "integerish", any.missing = F, all.missing = F, min.cols = 2),
                          checkmate::checkDataFrame(Y, types = "integerish", any.missing = F, all.missing = F, min.cols = 2))

        if(is.factor(Y) || is.vector(Y) || ncol(Y)==1) {
          if(is.null(self$responses)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
          if(is.data.frame(Y)) Y <- factor(Y[,1], levels = self$responses)
          else Y <- factor(Y, levels = self$responses)
          if(anyNA(Y)) stop("Unknown class labels. This probably means that new provided target data (e.g. for continued training) includes class labels that did not exist in the data used for the initial training.\n
                                                If this was intended, make sure to provide the target data for the initial training as factor that includes all required class labels as level, even those with zero occurences in the initial data.")
          return(torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32()))
        } else {
          Y <- as.matrix(Y)
          if(!all(Y %in% c(0,1))) {
            if(is.null(self$responses)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
            else stop(paste0("Model expects target data to be provided as factor, character vector, character matrix/data.frame with 1 column or integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
          }
          if(!is.null(self$responses) && all(self$responses %in% colnames(Y))) {
            return(torch::torch_tensor(Y[, self$responses], dtype = torch::torch_float32()))
          } else {
            if(self$y_dim != ncol(Y)) {
              if(is.null(self$responses)) stop(paste0("Model expects target data to be provided as integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
              else stop(paste0("Model expects target data to be provided as factor, character vector, character matrix/data.frame with 1 column or integerish matrix/data.frame with ", self$y_dim, " columns containing only zeroes and ones."))
            }
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

        Y <- as.matrix(Y)
        self$y_dim <- ncol(Y)
        self$responses <- colnames(Y)
        self$parameter = torch::nn_parameter(torch::torch_tensor(rep(0.5, ncol(Y)), requires_grad=TRUE))

        Y_base <- torch::torch_tensor(matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
        self$baseloss <- as.numeric(self$forward(self$link(Y_base), self$format_Y(Y))$mean())
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

#Is this still used?
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

  pointer_check_net <- tryCatch(object$net$state_dict(), error = function(e) e)
  pointer_check_loss <- tryCatch(object$loss$state_dict(), error = function(e) e)
  if(inherits(pointer_check_net, "error") || inherits(pointer_check_loss, "error")) {
    object$loaded_model_epoch <- "none"
  }

  if(object$loaded_model_epoch != object$use_model_epoch) {

    if(object$use_model_epoch == "best") {
      object$net$load_state_dict(torch::torch_load(object$best_epoch_net_state_dict))
      object$loss$load_state_dict(torch::torch_load(object$best_epoch_loss_state_dict))
    } else if(object$use_model_epoch == "last") {
      object$net$load_state_dict(torch::torch_load(object$last_epoch_net_state_dict))
      object$loss$load_state_dict(torch::torch_load(object$last_epoch_loss_state_dict))
    } else {
      stop("'object$use_model_epoch' must be either 'best' or 'last'.")
    }

    object$loaded_model_epoch <- object$use_model_epoch
  }

  object$net$eval()
  object$loss$eval()

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
