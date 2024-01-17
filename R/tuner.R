#' Tune hyperparameter
#'
#' Is used to control which hyperparameter should be tuned
#'
#' @param lower numeric, numeric vector, character, lower boundaries of tuning space
#' @param upper numeric, numeric vector, character, upper foundaries of tuning space
#' @param fixed charachter, used for multi-dimensional hyperparameters such as hidden, which dimensions should be fixed
#' @param additional numeric, additional control parameter which sets the value of the fixed argument
#'
#' @export

tune = function(lower = NULL, upper = NULL, fixed = NULL, additional = NULL) {
  out = list()
  out$lower = lower
  out$upper = upper
  out$fixed = fixed
  out$additional = additional
  class(out) = "tune"
  return(out)
}


#' Config hyperparameter tuning
#'
#' @param CV numeric, specifies k-folded cross validation
#' @param steps numeric, number of random tuning steps
#' @param parallel numeric, number of parallel cores (tuning steps are parallelized)
#' @param NGPU numeric, set if more than one GPU is available, tuning will be parallelized over CPU cores and GPUs, only works for NCPU > 1
#' @param cancel CV/tuning for specific hyperparameterset if model cannot reduce loss below baseline after burnin or returns NA loss
#'
#'
#' @export
config_tuning = function(CV = 5, steps = 10, parallel = FALSE, NGPU = 1, cancel = TRUE) {
  out = list()
  out$CV = CV
  out$steps = steps
  out$parallel = parallel
  out$NGPU = NGPU
  return(out)
}



tuning_function = function(tuner, parameters, loss.fkt,loss_obj, X, Y, data, formula, tuning, Y_torch, loss) {

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
          parameters$X = X[-cv,,drop=FALSE]
          if(is.matrix(Y)) parameters$Y = Y[-cv,,drop=FALSE]
          else parameters$Y = Y[-cv]
          parameters$data = data[-cv,,drop=FALSE]
          m = do.call(dnn, parameters)
          tune_df$models[[i]] = list(m)
          #tune_df$train[i] = tune_df$train[i]+ rev(m$losses$train_l[complete.cases(m$losses$train_l)])[1]*nrow(m$data$X)

          if(!m$successfull) {
            tune_df$test[i] = Inf
            break
          } else {
            pred = predict(m, newdata = data[cv,,drop=FALSE], type = "response")
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
      loss_obj <- get_loss(loss)
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
      for(j in 1:ncol(tmp_hp)) {
        if(colnames(tmp_hp)[j] == "hidden") {
          parameters[[colnames(tmp_hp)[j] ]] = rep(tmp_hp[1,j][[1]][[1]], tmp_hp[1,j][[1]][[2]])
        } else {
          parameters[[colnames(tmp_hp)[j] ]] = unlist(tmp_hp[1,j])
        }
      }
      # start CV
      # Stop if training is aborted
      for(cv in test_indices) {
        parameters$X = X[-cv,,drop=FALSE]
        if(is.matrix(Y)) parameters$Y = Y[-cv,,drop=FALSE]
        else parameters$Y = Y[-cv]
        parameters$data = data[-cv,,drop=FALSE]
        m = do.call(dnn, parameters)
        tune_df$models[[i]] = list(m)
        tune_df$train[i] = tune_df$train[i]+ rev(m$losses$train_l[complete.cases(m$losses$train_l)])[1]*nrow(m$data$X)

        if(!m$successfull) {
          tune_df$test[i] = Inf
          break
        } else {
          pred = predict(m, newdata = data[cv,,drop=FALSE], type = "response")
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

  parameters$X = X
  parameters$Y = Y
  parameters$data = data

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
