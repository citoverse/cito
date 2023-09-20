#' Continues training of a model for additional periods
#'
#' @param model a model created by \code{\link{dnn}}
#' @param data matrix or data.frame if not provided data from original training will be used
#' @param epochs additional epochs the training should continue for
#' @param continue_from define which epoch should be used as starting point for training, 0 if last epoch should be used
#' @param device device on which network should be trained on, either "cpu" or "cuda"
#' @param verbose print training and validation loss of epochs
#' @param changed_params list of arguments to change compared to original training setup, see \code{\link{dnn}} which parameter can be changed
#' @return a model of class cito.dnn same as created by  \code{\link{dnn}}
#'
#' @example /inst/examples/continue_training-example.R
#'
#' @import checkmate
#'
#' @export
continue_training <- function(model,
                              epochs = 32,
                              continue_from= NULL,
                              data=NULL,
                              device= "cpu",
                              verbose = TRUE,
                              changed_params=NULL){

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


  ### initiate model ###
  if(!is.null(continue_from)){
    model$use_model_epoch <- continue_from
  }else{
    model$use_model_epoch <- max(which(!is.na(model$losses$train_l)))
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

  targets <- format_targets(Y, model$loss)
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
