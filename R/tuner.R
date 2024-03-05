#' Tune hyperparameter
#'
#' Control hyperparameter tuning
#'
#' @param lower numeric, numeric vector, character, lower boundaries of tuning space
#' @param upper numeric, numeric vector, character, upper boundaries of tuning space
#' @param fixed character, used for multi-dimensional hyperparameters such as hidden, which dimensions should be fixed
#' @param additional numeric, additional control parameter which sets the value of the fixed argument
#' @param values custom values from which hyperparameters are sampled, must be a matrix for hidden layers (first column == nodes, second column == number of layers)
#'
#'
#'
#' @export

tune = function(lower = NULL, upper = NULL, fixed = NULL, additional = NULL, values = NULL) {
  out = list()
  out$lower = lower
  out$upper = upper
  out$fixed = fixed
  out$additional = additional
  out$values = values
  class(out) = "tune"
  return(out)
}


#' Config hyperparameter tuning
#'
#' @param CV numeric, specifies k-folded cross validation
#' @param steps numeric, number of random tuning steps
#' @param parallel numeric, number of parallel cores (tuning steps are parallelized)
#' @param NGPU numeric, set if more than one GPU is available, tuning will be parallelized over CPU cores and GPUs, only works for NCPU > 1
#' @param cancel CV/tuning for specific hyperparameter set if model cannot reduce loss below baseline after burnin or returns NA loss
#' @param bootstrap_final bootstrap final model, if all models should be boostrapped it must be set globally via the bootstrap argument in the [dnn()] function
#' @param bootstrap_parallel should the bootstrapping be parallelized or not
#'
#'
#' @details
#' Note that hyperparameter tuning can be expensive. We have implemented an option to parallelize hyperparameter tuning, including parallelization over one or more GPUs (the hyperparameter evaluation is parallelized, not the CV). This can be especially useful for small models. For example, if you have 4 GPUs, 20 CPU cores, and 20 steps (random samples from the random search), you could run `dnn(..., device="cuda",lr = tune(), batchsize=tune(), tuning=config_tuning(parallel=20, NGPU=4)', which will distribute 20 model fits across 4 GPUs, so that each GPU will process 5 models (in parallel).
#'
#'
#' @export
config_tuning = function(CV = 5, steps = 10, parallel = FALSE, NGPU = 1, cancel = TRUE, bootstrap_final = NULL, bootstrap_parallel = FALSE) {
  out = list()
  out$CV = CV
  out$steps = steps
  out$parallel = parallel
  out$NGPU = NGPU
  out$bootstrap = bootstrap_final
  out$bootstrap_parallel = bootstrap_parallel
  return(out)
}



tuning_function = function(tuner, parameters, loss.fkt,loss_obj, X, Y,Z, data, formula, tuning, Y_torch, loss, device) {

  parallel = tuning$parallel
  NGPU = tuning$NGPU

  cat("Starting hyperparameter tuning...\n")

  set = cut(sample.int(nrow(X)), breaks = tuning$CV, labels = FALSE)
  test_indices = lapply(unique(set), function(s) which(set == s, arr.ind = TRUE))

  steps = tuning$steps
  tune_df = tibble::tibble(steps = 1:steps, test = 0, train = 0, models =  NA)
  for(i in 1:length(tuner)) {
    if(names(tuner)[[i]] == "hidden") {
      s = (lapply(1:steps, function(j) tuner[[i]]$sample()))
      tune_df[["hidden"]] = s
    } else {
      tune_df[[names(tuner)[i]]] = sapply(1:steps, function(j) tuner[[i]]$sample())
    }
  }

  parameters$formula = formula
  parameters$plot = FALSE
  parameters$verbose = FALSE


  if(parallel == FALSE) {
    pb = progress::progress_bar$new(total = steps,
                                    format = ":percent :eta || Hyperparameters: :hp || Test loss: :test_loss \n", clear = FALSE)
    results_boot = list()
    # start non parallel block
    for(i in 1:steps) {
      tmp_hp = tune_df[i,-(1:4)]
      format_hp = format_hyperparameters(tmp_hp)
        for(j in 1:ncol(tmp_hp)) {
          if(colnames(tmp_hp)[j] == "hidden") {
            parameters[[colnames(tmp_hp)[j]]] = rep(tmp_hp[1,j][[1]][[1]][1], tmp_hp[1,j][[1]][[1]][2])
          } else {
            parameters[[colnames(tmp_hp)[j]]] = unlist(tmp_hp[1,j])
          }
        }

        # start CV
        # Stop if training is aborted
        for(cv in test_indices) {
          #parameters$X = X[-cv,,drop=FALSE]
          #if(is.matrix(Y)) parameters$Y = Y[-cv,,drop=FALSE]
          #else parameters$Y = Y[-cv]
          parameters$data = data[-cv,,drop=FALSE]
          m = do.call(dnn, parameters)
          tune_df$models[[i]] = list(m)
          #tune_df$train[i] = tune_df$train[i]+ rev(m$losses$train_l[complete.cases(m$losses$train_l)])[1]*nrow(m$data$X)
          if(!m$successfull) {
            tune_df$test[i] = Inf
            break
          } else {
            pred = stats::predict(m, newdata = data[cv,,drop=FALSE], type = "response")
            tune_df$test[i] = tune_df$test[i]+as.numeric(loss.fkt(loss_obj$link(
              torch::torch_tensor(pred, dtype=torch::torch_float32())),
              Y_torch[cv,,drop=FALSE])$sum())
          }
        }
      pb$tick(tokens = list(hp = format_hp, test_loss = round(tune_df$test[i], digits = 3)))
    }
  } else {
    if(is.logical(parallel)) {
      if(parallel) {
        parallel = parallel::detectCores() -1
      }
    }
    if(is.numeric(parallel)) {
      backend = parabar::start_backend(parallel)
      nodes = parabar::evaluate(backend, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-'))
      parabar::export(backend, ls(environment()), environment())
    }
    # start parallel block
    parabar::configure_bar(type = "modern", format = "[:bar] :percent :eta", width = round(getOption("width")/2), clear=FALSE)

    tune_df <- parabar::par_lapply(backend, 1:steps, function(i) {

      require(tibble)

      loss_obj <- get_loss(loss, device = device, X= X, Y = Y)
      loss.fkt <- loss_obj$loss
      targets <- format_targets(Y, loss_obj)
      Y_torch <- targets$Y

      if(NGPU > 1) {
        # who am I
        myself = paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')
        dist = cbind(nodes,0:NGPU)
        dev = as.integer(as.numeric(dist[which(dist[,1] %in% myself, arr.ind = TRUE), 2]))
        Sys.setenv(CUDA_VISIBLE_DEVICES=dev)
      }

      if(!is.null(loss_obj$parameter)) loss_obj$parameter <- list(paramter = loss_obj$parameter)
      if(!is.null(loss_obj$parameter)) list2env(loss_obj$parameter,envir = environment(fun= loss.fkt))

      tmp_hp = tune_df[i,-(1:4)]
      format_hp = format_hyperparameters(tmp_hp)
      for(j in 1:ncol(tmp_hp)) {
        if(colnames(tmp_hp)[j] == "hidden") {
          parameters[[colnames(tmp_hp)[j]]] = rep(tmp_hp[1,j][[1]][[1]][1], tmp_hp[1,j][[1]][[1]][2])
        } else {
          parameters[[colnames(tmp_hp)[j]]] = unlist(tmp_hp[1,j])
        }
      }
      # start CV
      # Stop if training is aborted
      for(cv in test_indices) {
        #parameters$X = X[-cv,,drop=FALSE]
        #if(is.matrix(Y)) parameters$Y = Y[-cv,,drop=FALSE]
        #else parameters$Y = Y[-cv]
        parameters$data = data[-cv,,drop=FALSE]
        m = do.call(dnn, parameters)
        tune_df$models[[i]] = list(m)
        tune_df$train[i] = tune_df$train[i]+ rev(m$losses$train_l[stats::complete.cases(m$losses$train_l)])[1]*nrow(m$data$X)

        if(!m$successfull) {
          tune_df$test[i] = Inf
          break
        } else {
          pred = stats::predict(m, newdata = data[cv,,drop=FALSE], type = "response")
          tune_df$test[i] = tune_df$test[i]+as.numeric(loss.fkt(loss_obj$link(
           torch::torch_tensor(pred, dtype=torch::torch_float32())),
           Y_torch[cv,,drop=FALSE])$sum())
        }
      }
      return(tune_df[i,])
    })
    parabar::stop_backend(backend)
    tune_df = do.call(rbind, tune_df)
  }

  #parameters$X = X
  #parameters$Y = Y
  #parameters$Z = Z
  parameters$data = data

  parameters$bootstrap = tuning$bootstrap
  parameters$bootstrap_parallel = tuning$bootstrap_parallel

  tmp_hp = tune_df[which.min(tune_df$test),-(1:4)]
  for(j in 1:ncol(tmp_hp)) {
    if(colnames(tmp_hp)[j] == "hidden") {
      parameters[[colnames(tmp_hp)[j]]] = rep(tmp_hp[1,j][[1]][[1]][1], tmp_hp[1,j][[1]][[1]][2])
    } else {
      parameters[[colnames(tmp_hp)[j]]] = unlist(tmp_hp[1,j])
    }
  }

  # fit best model
  cat("Fitting final model...\n")
  m = do.call(dnn, parameters)
  m$tuning = tune_df
  return(m)

}

format_hyperparameters = function(hp) {
  res = ""
  for(i in 1:ncol(hp)) {
    if(colnames(hp)[i] == "hidden") {
      res = paste0(res, paste0("hidden = [", hp[i][[1]][[1]][1], " units, ", hp[i][[1]][[1]][2], " layers], "))
    } else {
      if(is.numeric(hp[1,i])) res = paste0(res, colnames(hp)[i], " = ", round(hp[1,i], 4), " ")
      else res = paste0(res, colnames(hp)[i], " = ", hp[1,i], " ")
    }
  }
  return(res)
}




check_hyperparameters = function(hidden  ,
                                 bias ,
                                 lambda ,
                                 alpha ,
                                 dropout,
                                 lr ,
                                 activation,
                                 batchsize,
                                 epochs) {

  out = list()
  if(inherits(hidden, "tune")) {

    if(is.null(hidden$values)) {
      if(is.null(hidden$lower)) hidden$lower = c(5, 1)
      if(is.null(hidden$upper)) hidden$upper = c(100, 10)
      if(is.null(hidden$fixed)) hidden$fixed = "both"

      if(hidden$fixed == "depth") {
        out$hidden$sampler = function() {
          return(c(sample(hidden$lower[1]:hidden$upper[1], 1), hidden$additional))
        }
      } else if(hidden$fixed == "width") {
        out$hidden$sampler = function() {
          return(c(hidden$additional, sample(hidden$lower[1]:hidden$upper[1], 1)))
        }
      } else {
        out$hidden$sampler = function() {
          return(c(sample(hidden$lower[1]:hidden$upper[1], 1), sample(hidden$lower[2]:hidden$upper[2], 1)))
        }
      }

    } else {
      checkmate::qassert(hidden$values,"M" )
      out$hidden$sampler = function() {

        indices = nrow(hidden$values)
        candidate = hidden$values[sample.int(indices, 1),]

        return(c(candidate[1], candidate[2]))
      }
    }

  }

  if(inherits(bias, "tune")) {

    if(is.null(bias$values)) {
      out$bias$sampler = function() {
        return(sample(c(TRUE, FALSE), 1))
      }
    } else {
      out$bias$sampler = function() {
        return(sample(bias$values, 1))
      }
    }
  } else {
    checkmate::qassert(bias, "B+")
  }

  if(inherits(lambda, "tune")) {

    if(is.null(lambda$values)) {
      if(is.null(lambda$lower)) lambda$lower = 0.0
      if(is.null(lambda$upper)) lambda$upper = 0.5
      out$lambda$sampler = function() {
        return(stats::runif(1, lambda$lower, lambda$upper))
      }
    } else {

      out$lambda$sampler = function() {
        return(sample(lambda$values, 1))
      }
    }
  } else {
    checkmate::qassert(lambda, "R1[0,)")
  }

  if(inherits(alpha, "tune")) {

    if(is.null(alpha$values)) {

      if(is.null(alpha$lower)) alpha$lower = 0.0
      if(is.null(alpha$upper)) alpha$upper = 1.0
      out$alpha$sampler = function() {
        return(stats::runif(1, alpha$lower, alpha$upper))
      }
    } else {
      out$alpha$sampler = function() {
        return(sample(alpha$values, 1))
      }
    }
  } else {
    checkmate::qassert(alpha, "R1[0,1]")
  }

  if(inherits(activation, "tune")) {

    if(is.null(activation$values)) {

      if(is.null(activation$lower)) activation$lower = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                                                         "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink",
                                                         "softshrink", "hardshrink", "log_sigmoid")
      out$activation$sampler = function() {
        return(sample(activation$lower, 1))
      }
    } else {
      out$activation$sampler = function() {
        return(sample(activation$values, 1))
      }
    }
  } else {
    checkmate::qassert(activation, "S+[1,)")
  }


  if(inherits(dropout, "tune")) {
    if(is.null(dropout$values )) {
      if(is.null(dropout$lower)) dropout$lower = 0.0
      if(is.null(dropout$upper)) dropout$upper = 1.0
      out$dropout$sampler = function() {
        return(stats::runif(1, dropout$lower, dropout$upper))
      }
    } else {
      out$dropout$sampler = function() {
        return(sample(dropout$values, 1))
      }
    }
  } else {
    checkmate::qassert(dropout, "R1[0,1]")
  }

  if(inherits(lr, "tune")) {
    if(is.null(lr$values)) {
      if(is.null(lr$lower)) lr$lower = 0.0
      if(is.null(lr$upper)) lr$upper = 1.0
      out$lr$sampler = function() {
        return(stats::runif(1, lr$lower, lr$upper))
      }
    } else {
      out$lr$sampler = function() {
        return(sample(lr$values, 1))
      }
    }
  } else {
    checkmate::qassert(lr, "R1[0,100]")
  }


  if(inherits(batchsize, "tune")) {
    if(is.null(batchsize$values)) {
      if(is.null(batchsize$lower)) batchsize$lower = 1
      if(is.null(batchsize$upper)) batchsize$upper = 100
      out$batchsize$sampler = function() {
        return(sample(batchsize$lower:batchsize$upper, 1))
      }
    } else {
      out$batchsize$sampler = function() {
        return(sample(batchsize$values, 1))
      }
    }
  }

  if(inherits(epochs, "tune")) {
    if(is.null(epochs$values)) {
      if(is.null(epochs$lower)) epochs$lower = 1
      if(is.null(epochs$upper)) epochs$upper = 300
      out$epochs$sampler = function() {
        return(sample(epochs$lower:epochs$upper, 1))
      }
    } else {
      out$epochs$sampler = function() {
        return(sample(epochs$values, 1))
      }
    }
  }
  return(out)
}



