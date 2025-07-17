#' Continues training of a model generated with \code{\link{dnn}}, \code{\link{cnn}} or \code{\link{mmn}} for additional epochs.
#'
#' @description
#' If the training/validation loss is still decreasing at the end of the training, it is often a sign that the NN has not yet converged. You can use this function to continue training instead of re-training the entire model.
#'
#'
#' @param model a model created by \code{\link{dnn}}, \code{\link{cnn}} or \code{\link{mmn}}
#' @param ... class-specific arguments
#' @param epochs additional epochs the training should continue for
#' @param data matrix or data.frame. If not provided data from original training will be used
#' @param X Predictor data. If not provided X from original training will be used
#' @param Y Target data. If not provided Y from original training will be used
#' @param dataList A list containing the data for training the model. The list should contain all variables used in the formula. If not provided dataList from original training will be used
#' @param device can be used to overwrite device used in previous training
#' @param changed_params list of arguments to change compared to original training setup, see \code{\link{dnn}} which parameter can be changed
#' @param parallel train bootstrapped model in parallel
#' @param init_optimizer re-initialize optimizer or not
#' @return a model of class citodnn, citodnnBootstrap, citocnn or citommn created by \code{\link{dnn}}, \code{\link{cnn}} or \code{\link{mmn}}
#'
#' @example /inst/examples/continue_training-example.R
#'
#' @import checkmate
#'
#' @export
continue_training <- function(model, ...){UseMethod("continue_training")}

#' @rdname continue_training
#' @export
continue_training.citodnn <- function(model,
                              epochs = 32,
                              data = NULL,
                              device = NULL,
                              changed_params = NULL,
                              init_optimizer = TRUE,
                              X = NULL,
                              Y = NULL,
                              ...) {

  if(is.null(device)) device = model$training_properties$device

  device <- check_device(device)

  model$use_model_epoch <- "last"
  model <- check_model(model)

  if(!is.null(changed_params)) {
    for(name in names(changed_params)) {
      model$training_properties[[name]] <- changed_params[[name]]
    }
  }

  new_validation_split = TRUE
  if(is.null(data) & is.null(X)) {
    data <- model$data$data
    new_validation_split = FALSE
  }

  if(!is.null(X) & is.null(Y)) stop("Y missing.")

  tmp_data = get_X_Y(model$old_formula, X, Y, data)
  X = tmp_data$X
  Y = tmp_data$Y
  Z = tmp_data$Z

  X <- torch::torch_tensor(X, dtype = torch::torch_float32())
  Y <- model$loss$format_Y(Y)
  if(!is.null(Z)) Z = torch::torch_tensor(Z, dtype=torch::torch_long())

  if(length(model$training_properties$validation) == 1 && model$training_properties$validation == 0) {
    if(is.null(Z)) {
      train_dl <- get_data_loader(X, Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    } else {
      train_dl <- get_data_loader(X, Z, Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    }
    valid_dl <- NULL
  } else {
    n_samples <- dim(Y)[1]
    if(new_validation_split | is.null(model$data$validation)) {
      if(length(model$training_properties$validation)>1) {
        if(!"validation" %in% names(changed_params)) warning("You provided new training data, but did not provide new validation indices. The validation indices of the initial training will be used.")
        if(any(model$training_properties$validation>n_samples)) stop("Validation indices mustn't exceed the number of samples.")
        valid  <- model$training_properties$validation
      } else {
        valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(model$training_properties$validation*n_samples)))
      }
    } else {
      valid <- model$data$validation
    }
    train <- c(1:n_samples)[-valid]
    if(is.null(Z)) {
      train_dl <- get_data_loader(X[train, drop=F], Y[train, drop=F], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
      valid_dl <- get_data_loader(X[valid, drop=F], Y[valid, drop=F], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    } else {
      train_dl <- get_data_loader(X[train, drop=F], Z[train, drop=F], Y[train, drop=F], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
      valid_dl <- get_data_loader(X[valid, drop=F], Z[valid, drop=F], Y[valid, drop=F], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    }
  }

  model <- train_model(model = model,epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, plot_new = TRUE, init_optimizer = init_optimizer)

  return(model)
}


#' @rdname continue_training
#' @export
continue_training.citodnnBootstrap <- function(model,
                                      epochs = 32,
                                      data = NULL,
                                      device = NULL,
                                      changed_params = NULL,
                                      parallel = FALSE,
                                      init_optimizer = TRUE,
                                      X = NULL,
                                      Y = NULL,
                                      ...) {

  if(parallel == FALSE) {
    pb = progress::progress_bar$new(total = length(model$models), format = "[:bar] :percent :eta", width = round(getOption("width")/2))

    for(b in 1:length(model$models)) {
      model$models[[b]] = continue_training(model$models[[b]], epochs = epochs, data = data, device = device, changed_params = NULL, init_optimizer = init_optimizer, X = X, Y = Y)
      pb$tick()
    }
  } else {
    if(is.logical(parallel)) {
      if(parallel) {
        parallel = parallel::detectCores() -1
      }
    }
    if(is.numeric(parallel)) {
      backend = parabar::start_backend(parallel)
      parabar::export(backend, ls(environment()), environment())
    }

    parabar::configure_bar(type = "modern", format = "[:bar] :percent :eta", width = round(getOption("width")/2))
    model$models <- parabar::par_lapply(backend, 1:length(model$models), function(b) {
      return(continue_training(model$models[[b]], epochs = epochs, data = data, device = device, changed_params = NULL, init_optimizer = init_optimizer, X = X, Y = Y))

    })
    parabar::stop_backend(backend)

  }
  return(model)
}

#' @rdname continue_training
#' @export
continue_training.citocnn <- function(model,
                                      epochs = 32,
                                      X = NULL,
                                      Y = NULL,
                                      device = NULL,
                                      changed_params = NULL,
                                      init_optimizer = TRUE,
                                      ...) {

  checkmate::assert(checkmate::checkArray(X, mode = "numeric", min.d = 3, max.d = 5, any.missing = FALSE), checkmate::check_character(X), checkmate::checkNull(X))
  checkmate::qassert(epochs, "X1[1,)")

  if(is.null(device)) device <- model$training_properties$device
  device <- check_device(device)

  model$use_model_epoch <- "last"
  model <- check_model(model)

  if(!is.null(changed_params)) {
    for(name in names(changed_params)) {
      model$training_properties[[name]] <- changed_params[[name]]
    }
  }

  if((is.null(X) & !is.null(Y)) | (!is.null(X) & is.null(Y))) stop("X and Y must either be both assigned or both NULL.")

  if(!is.character(X) & !is.character(model$data$X)) {
    if(!is.null(X) && !all(dim(X)[-1] == dim(model$data$X)[-1])) stop(paste0("Wrong dimensions of X [",paste0(dim(X), collapse = ","),"]. Correct dimensions: [N,",paste0(dim(model$data$X)[-1]), "]"))
  }

  new_validation_split = TRUE
  if(is.null(X) & is.null(Y)) {
    X <- model$data$X
    Y <- model$data$Y
    new_validation_split = FALSE
  }

  from_folder = FALSE

  Y <- model$loss$format_Y(Y)

  if(is.character(X)) {
    X <- list.files(X, full.names = TRUE)
    from_folder = TRUE
    if(dim(Y)[1] != length(X)) stop(paste0("Y (", dim(Y)[1], ") has to have an equal number of samples as X (", length(X), ")."))
  } else {
    X <- torch::torch_tensor(X, dtype = torch::torch_float32())
    if(dim(Y)[1] != dim(X)[1]) stop(paste0("Y (", dim(Y)[1], ") has to have an equal number of samples as X (", dim(X)[1], ")."))
  }

  if(length(model$training_properties$validation) == 1 && model$training_properties$validation == 0) {
    train_dl <- get_data_loader(X, Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, from_folder = from_folder)
    valid_dl <- NULL
  } else {
    n_samples <- dim(Y)[1]
    if(new_validation_split | is.null(model$data$validation)) {
      if(length(model$training_properties$validation)>1) {
        if(!"validation" %in% names(changed_params)) warning("You provided new training data, but did not provide new validation indices. The validation indices of the initial training will be used.")
        if(any(model$training_properties$validation>n_samples)) stop("Validation indices mustn't exceed the number of samples.")
        valid  <- model$training_properties$validation
      } else {
        valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(model$training_properties$validation*n_samples)))
      }
    } else {
      valid <- model$data$validation
    }
    train <- c(1:n_samples)[-valid]
    train_dl <- get_data_loader(X[train, drop=FALSE], Y[train, drop=FALSE], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, from_folder = from_folder)
    valid_dl <- get_data_loader(X[valid, drop=FALSE], Y[valid, drop=FALSE], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, from_folder = from_folder)
  }

  model <- train_model(model = model, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, plot_new = TRUE, init_optimizer = init_optimizer)

  return(model)
}

#' @rdname continue_training
#' @export
continue_training.citommn <- function(model,
                                      epochs = 32,
                                      dataList = NULL,
                                      device = NULL,
                                      changed_params = NULL,
                                      init_optimizer = TRUE,
                                      ...) {

  checkmate::assert(checkmate::checkList(dataList), checkmate::checkNull(dataList))
  checkmate::qassert(epochs, "X1[1,)")


  if(is.null(device)) device <- model$training_properties$device
  device <- check_device(device)

  model$use_model_epoch <- "last"
  model <- check_model(model)

  if(!is.null(changed_params)) {
    for(name in names(changed_params)) {
      model$training_properties[[name]] <- changed_params[[name]]
    }
  }

  new_validation_split = TRUE
  if(is.null(dataList)) {
    dataList <- model$data$dataList
    new_validation_split = FALSE
  }

  if(!as.character(model$call$formula[[2]]) %in% names(dataList)) stop(paste0("In '", deparse1(model$call$formula), "': Couldn't find '", as.character(model$call$formula[[2]]), "' in names of dataList."))
  Y <- eval(model$call$formula[[2]], envir = dataList)

  X <- format_input_data(model$call$formula[[3]], dataList)

  from_folder = FALSE
  if(any(sapply(X, is.character))) from_folder = TRUE

  X <- lapply(X, function(x) {
    if(is.character(x)) x = list.files(x, full.names = TRUE)
    return(x)
  })

  Y <- model$loss$format_Y(Y)

  if(length(model$training_properties$validation) == 1 && model$training_properties$validation == 0) {
    train_dl <- do.call(get_data_loader, append(X, list(Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, from_folder = from_folder)))
    valid_dl <- NULL
  } else {
    n_samples <- dim(Y)[1]
    if(new_validation_split | is.null(model$data$validation)) {
      if(length(model$training_properties$validation)>1) {
        if(!"validation" %in% names(changed_params)) warning("You provided new training data, but did not provide new validation indices. The validation indices of the initial training will be used.")
        if(any(model$training_properties$validation>n_samples)) stop("Validation indices mustn't exceed the number of samples.")
        valid  <- model$training_properties$validation
      } else {
        valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(model$training_properties$validation*n_samples)))
      }
    } else {
      valid <- model$data$validation
    }
    train <- c(1:n_samples)[-valid]
    train_dl <- do.call(get_data_loader, append(lapply(append(X, Y), function(x) x[train, drop=F]), list(batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, from_folder = from_folder)))
    valid_dl <- do.call(get_data_loader, append(lapply(append(X, Y), function(x) x[valid, drop=F]), list(batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, from_folder = from_folder)))
  }

  model <- train_model(model = model, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, plot_new = TRUE, init_optimizer = init_optimizer)

  return(model)
}



