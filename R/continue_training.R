#' Continues training of a model generated with \code{\link{dnn}} for additional epochs.
#'
#' @param model a model created by \code{\link{dnn}}
#' @param data matrix or data.frame if not provided data from original training will be used
#' @param epochs additional epochs the training should continue for
#' @param device device on which network should be trained on, either "cpu" or "cuda"
#' @param verbose print training and validation loss of epochs
#' @param changed_params list of arguments to change compared to original training setup, see \code{\link{dnn}} which parameter can be changed
#' @param parallel train bootstrapped model in parallel
#' @return a model of class citodnn or citodnnBoostrap created by  \code{\link{dnn}}
#'
#' @example /inst/examples/continue_training-example.R
#'
#' @import checkmate
#'
#' @export
continue_training <- function(model,
                              epochs = 32,
                              data=NULL,
                              device= "cpu",
                              verbose = TRUE,
                              changed_params=NULL,
                              parallel = FALSE){UseMethod("continue_training")}

#' @rdname continue_training
#' @export
continue_training.citodnn <- function(model,
                              epochs = 32,
                              data=NULL,
                              device= "cpu",
                              verbose = TRUE,
                              changed_params=NULL,
                              parallel = FALSE){

  checkmate::qassert(device, "S+[3,)")

  ### Training device ###
  if(device== "cuda"){
    if (torch::cuda_is_available()) {
      device<- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device<- torch::torch_device("cpu")
    }

  }else {
    if(device!= "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device<- torch::torch_device("cpu")
  }

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
  Y = as.matrix(Y)

  y_dim = ncol(Y)
  y_dtype = torch::torch_float32()
  if(is.character(Y)) {
    y_dim = length(unique(as.integer(as.factor(Y[,1]))))
    Y = matrix(as.integer(as.factor(Y[,1])), ncol = 1L)
    if(inherits(model$loss$call, "family")){
      if(model$loss$call$family == "binomial") {
        Y = torch::as_array(torch::nnf_one_hot( torch::torch_tensor(Y, dtype=torch::torch_long() ))$squeeze() )
      }
    }
  }
  if(!is.function(model$loss$call)){
    if(all(model$loss$call == "softmax")) y_dtype = torch::torch_long()
  }

  if(model$training_properties$validation != 0){

    train <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(model$training_properties$validation*nrow(X))))
    valid <- c(1:nrow(X))[-train]
    train_dl <- get_data_loader(X[train,],Y[train,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, y_dtype=y_dtype)
    valid_dl <- get_data_loader(X[valid,],Y[valid,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, y_dtype=y_dtype)

  }else{
    train_dl <- get_data_loader(X,Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, y_dtype=y_dtype)
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
                                      device= "cpu",
                                      verbose = TRUE,
                                      changed_params=NULL,
                                      parallel = FALSE){

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






