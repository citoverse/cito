#' Continues training of a model generated with \code{\link{dnn}} or \code{\link{cnn}} for additional epochs.
#'
#' @description
#' If the training/validation loss is still decreasing at the end of the training, it is often a sign that the NN has not yet converged. You can use this function to continue training instead of re-training the entire model.
#'
#'
#' @param model a model created by \code{\link{dnn}} or \code{\link{cnn}}
#' @param ... class-specific arguments
#' @param epochs additional epochs the training should continue for
#' @param data matrix or data.frame. If not provided data from original training will be used
#' @param X array. If not provided X from original training will be used
#' @param Y vector, factor, numerical matrix or logical matrix. If not provided Y from original training will be used
#' @param device device on which network should be trained on, either "cpu" or "cuda"
#' @param verbose print training and validation loss of epochs
#' @param changed_params list of arguments to change compared to original training setup, see \code{\link{dnn}} which parameter can be changed
#' @param parallel train bootstrapped model in parallel
#' @return a model of class citodnn, citodnnBootstrap or citocnn created by \code{\link{dnn}} or \code{\link{cnn}}
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
                              data=NULL,
                              device= c("cpu","cuda", "mps"),
                              verbose = TRUE,
                              changed_params=NULL,
                              ...){

  checkmate::qassert(device, "S+[3,)")
  device <- match.arg(device)
  device <- check_device(device)

  if(device == "cpu" && device != model$device) print(paste0("Original training was performed on ", model$device, ". This training is performed on cpu. If this is not intended, use the parameter 'device'!"))

  model<- check_model(model)


  ### set training environment ###
  if(!is.null(changed_params)){
    for (i in 1:length(changed_params)){
      if(is.character(unlist(changed_params[i]))) parantheses<- "\"" else parantheses<- ""
      eval(parse(text=paste0("model$training_properties$",names(changed_params)[i], " <- ", parantheses,changed_params[i],parantheses)))
    }
  }

  ### set dataloader  ###
  fm<- stats::as.formula(model$call$formula)
  if(is.null(data)) data <- model$data$data

  X = stats::model.matrix(fm, data)
  Y = stats::model.response(stats::model.frame(fm, data))

  targets <- format_targets(Y, model$loss, model$data$ylvls)
  Y <- targets$Y
  y_dim <- targets$y_dim
  ylvls <- targets$ylvls

  X <- torch::torch_tensor(as.matrix(X))


  ### dataloader  ###
  if(model$training_properties$validation != 0) {
    n_samples <- nrow(X)
    valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(model$training_properties$validation*n_samples)))
    train <- c(1:n_samples)[-valid]
    train_dl <- get_data_loader(X[train,], Y[train,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    valid_dl <- get_data_loader(X[valid,], Y[valid,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
  } else {
    train_dl <- get_data_loader(X, Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    valid_dl <- NULL
  }


  model <- train_model(model = model,epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose, plot_new = TRUE)

    return(model)
}


#' @rdname continue_training
#' @export
continue_training.citodnnBootstrap <- function(model,
                                      epochs = 32,
                                      data=NULL,
                                      device= c("cpu","cuda", "mps"),
                                      verbose = TRUE,
                                      changed_params=NULL,
                                      parallel = FALSE,
                                      ...){

  if(parallel == FALSE) {
    pb = progress::progress_bar$new(total = length(model$models), format = "[:bar] :percent :eta", width = round(getOption("width")/2))

    for(b in 1:length(model$models)) {
      model$models[[b]] = continue_training(model$models[[b]], epochs = epochs, data = data, device = device, verbose = FALSE, changed_params = NULL)
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
      return(continue_training(model$models[[b]], epochs = epochs, data = data, device = device, verbose = FALSE, changed_params = NULL))

    })
    parabar::stop_backend(backend)

  }
  return(model)
}

#' @rdname continue_training
#' @export
continue_training.citocnn <- function(model,
                                      epochs = 32,
                                      X=NULL,
                                      Y=NULL,
                                      device= c("cpu", "cuda", "mps"),
                                      verbose = TRUE,
                                      changed_params=NULL,
                                      ...){

  checkmate::qassert(epochs, "X1[0,)")
  checkmate::qassert(device, "S+[3,)")
  device <- match.arg(device)
  device <- check_device(device)

  if((is.null(X) & !is.null(Y)) | (!is.null(X) & is.null(Y))) stop("X and Y must either be both assigned or both NULL")
  if(!is.null(X) && !all(dim(X)[-1] == dim(model$data$X)[-1])) stop(paste0("Wrong dimensions of X [",paste0(dim(X), collapse = ","),"]. Correct dimensions: [N,",paste0(dim(model$data$X)[-1]), "]"))
  if(!is.null(Y) && is.matrix(Y) && ncol(Y) != ncol(model$data$Y)) stop(paste0("Wrong dimensions of Y [",paste0(dim(Y), collapse = ","),"]. Correct dimensions: [N,",ncol(model$data$Y), "]"))
  if(!is.null(Y) && is.matrix(Y) && nrow(Y) != dim(X)[1]) stop(paste0("nrow(Y)=", nrow(Y), " has to be equal to dim(X)[1]=", dim(X)[1]))
  if(!is.null(Y) && !is.matrix(Y) && length(Y) != dim(X)[1]) stop(paste0("length(Y)=", length(Y), " has to be equal to dim(X)[1]=", dim(X)[1]))

  model<- check_model(model)


  ### set training environment ###
  if(!is.null(changed_params)){
    for (i in 1:length(changed_params)){
      if(is.character(unlist(changed_params[i]))) parantheses<- "\"" else parantheses<- ""
      eval(parse(text=paste0("model$training_properties$",names(changed_params)[i], " <- ", parantheses,changed_params[i],parantheses)))
    }
  }

  ### set dataloader  ###
  if(is.null(X)) X <- model$data$X
  if(is.null(Y)) Y <- model$data$Y

  targets <- format_targets(Y, model$loss, model$data$ylvls)
  Y <- targets$Y
  y_dim <- targets$y_dim
  ylvls <- targets$ylvls

  X <- torch::torch_tensor(X, dtype = torch::torch_float32())


  ### dataloader  ###
  if(model$training_properties$validation != 0) {
    n_samples <- dim(X)[1]
    valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(model$training_properties$validation*n_samples)))
    train <- c(1:n_samples)[-valid]
    train_dl <- get_data_loader(X[train,], Y[train,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    valid_dl <- get_data_loader(X[valid,], Y[valid,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
  } else {
    train_dl <- get_data_loader(X, Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle)
    valid_dl <- NULL
  }


  model <- train_model(model = model,epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose, plot_new = TRUE)

  return(model)
}




