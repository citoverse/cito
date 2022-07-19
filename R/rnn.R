#' RNN
#'
#' @description
#'
#' fits a recurrent neural network. rnn() supports the formula syntax and allows to customize the shape of it to a maximal degree.
#' It is possible to fit gated recurrent unit RNNs (GRU) or multi-layer long short-term memory RNNs (LSTM). To learn more about Deep Learning, see [here](https://www.nature.com/articles/nature14539)
#' @param formula an object of class "\code{\link[stats]{formula}}": a description of the model that should be fitted
#' @param data matrix or data.frame
#' @param type Defines which type of RNN is fitted, either gated recurrent units (gru) or long short-term memory (lstm)
#' @param hidden_size Number of features in hidden state of lstm or gru
#' @param num_layers Number of rnn layers, if >1 then a stacked gru/lstm is created
#' @param lag How many iterations before \eqn{Y_i} are used for prediction, \eqn{X_{i-1,i-(lag+1)}}
#' @param validation percentage of data set that should be taken as validation set (the last iterations are chosen)
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers + 1 (last layer)) of logicals for each layer.
#' @param alpha add L1/L2 regularization to training  \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2} will get added for each layer. Can be single integer between 0 and 1 or vector of alpha values if layers should be regularized differently.
#' @param lambda strength of regularization: lambda penalty, \eqn{\lambda * (L1 + L2)} (see alpha)
#' @param dropout dropout rate, probability of a node getting left out during training (see \code{\link[torch]{nn_dropout}})
#' @param optimizer which optimizer used for training the network, for more adjustments to optimizer see \code{\link{config_optimizer}}
#' @param lr learning rate given to optimizer
#' @param batchsize number of samples that are used to calculate one learning rate step
#' @param shuffle if TRUE, data in each batch gets reshuffled every epoch
#' @param epochs epochs the training goes on for
#' @param lr_scheduler learning rate scheduler created with \code{\link{config_lr_scheduler}}
#' @param plot plot training loss
#' @param verbose print training and validation loss of epochs
#' @param device device on which network should be trained on.
#' @param early_stopping if set to integer, training will stop if validation loss worsened between current defined past epoch.
#'
#' @details
#' value
#'
#'
#' @return an S3 object of class \code{"citornn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_sequential" "nn_module", originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call}
#' \item{loss}{A list which contains relevant information for the target variable and the used loss function}
#' \item{losses}{A data.frame containing training and validation losses of each epoch}
#' \item{data}{Contains data used for training the model}
#' \item{weigths}{List of weights for each training epoch}
#' @import checkmate
#' @example /inst/examples/dnn-example.R
#' @seealso \code{\link{predict.citornn}}, \code{\link{plot.citornn}},  \code{\link{coef.citornn}},\code{\link{print.citornn}}, \code{\link{summary.citornn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#' @export
rnn <- function(formula,
                data = NULL,
                loss = c("mae", "mse", "softmax", "cross-entropy", "gaussian", "binomial", "poisson"),
                type = c("gru", "lstm"),
                hidden_size = 64,
                num_layers = 1,
                lag = 3,
                validation = 0,
                bias = TRUE,
                lambda = 0.0,
                alpha = 0.5,
                dropout = 0.0,
                optimizer = c("adam","adadelta", "adagrad", "rmsprop", "rprop", "sgd", "lbfgs"),
                lr = 0.01,
                batchsize = 32L,
                shuffle = FALSE,
                epochs = 32,
                plot = TRUE,
                verbose = TRUE,
                lr_scheduler = NULL,
                device = c("cpu","cuda"),
                early_stopping = FALSE){

  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
  checkmate::qassert(num_layers, "X+")
  checkmate::qassert(hidden_size, "X+")
  checkmate::qassert(num_layers, "X+")
  checkmate::qassert(bias, "B+")
  checkmate::qassert(lambda, "R1[0,)")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(dropout, "R+[0,)")
  checkmate::qassert(lr, "R+[0,)")
  checkmate::qassert(plot,"B1")
  checkmate::qassert(early_stopping,c("R1[1,)","B1"))
  checkmate::qassert(device, "S+[3,)")

  if(!is.function(loss) & !inherits(loss,"family")){
    loss <- match.arg(loss)
  }
  type <- match.arg(type)
  device <- match.arg(device)
  if(device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")
    }else{
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }
  }

  if(grepl(pattern = as.character(formula[2]), x = as.character(formula[3])) &
     grepl(pattern = paste0("I(",as.character(formula[2]),")"), x = as.character(formula[3]))){
    warning(paste0("To avoid ", as.character(formula[2])," being scraped from your data set, please write ",as.character(formula[2]),"~I(",as.character(formula[2]),")"))
  }


  ### Generate X & Y data ###
  if(!is.data.frame(data)) data <- data.frame(data)

  if(!is.null(formula)){
    fct_call <- match.call()
    m <- match("formula", names(fct_call))
    if(inherits(fct_call[3]$formula, "name")) fct_call[3]$formula <- eval(fct_call[3]$formula, envir = parent.env(environment()))
    formula <- stats::as.formula(fct_call[m]$formula)
  } else {
    formula <- stats::as.formula("~.")
  }
  X <- stats::model.matrix(formula, data)
  Y <- stats::model.response(stats::model.frame(formula, data))
  if(!inherits(Y, "matrix")) Y = as.matrix(Y)


  loss_obj <- get_loss(loss)

  y_dim <- ncol(Y)
  x_dtype <- torch::torch_float32()
  y_dtype <- torch::torch_float32()
  if(is.character(Y)) {
    y_dim <- length(unique(as.integer(as.factor(Y[,1]))))
    Y <- matrix(as.integer(as.factor(Y[,1])), ncol = 1L)
    if(inherits(loss_obj$call, "family")){
      if(loss_obj$call$family == "binomial") {
        Y <- torch::as_array(torch::nnf_one_hot(torch::torch_tensor(Y, dtype=torch::torch_long() ))$squeeze())
      }}
  }

  if(!is.function(loss_obj$call)){
    if(all(loss_obj$call == "softmax")) y_dtype = torch::torch_long()
  }

  ### dataloader  ###
  if(validation != 0){
    valid <- c(ceiling((1-validation) * nrow(X)):nrow(X))
    train <- c(1:floor((1-validation) * nrow(X)))

    train_dl <- get_data_loader_rnn(X[train,, drop=FALSE] ,Y[train,, drop=FALSE], lag = lag,  batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
    valid_dl <- get_data_loader_rnn(X[valid,, drop=FALSE] ,Y[valid,, drop=FALSE], lag = lag, batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)

  }else{
    train_dl <- get_data_loader_rnn(X, Y, lag = lag, batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
    valid_dl <- NULL
  }

  model_properties <- list(input = ncol(X),
                           output = y_dim,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           bias = bias,
                           dropout = dropout)

  training_properties <- list(lag = lag,
                              lr = lr,
                              lr_scheduler = lr_scheduler,
                              optimizer = optimizer,
                              epochs = epochs,
                              early_stopping = early_stopping,
                              plot = plot,
                              validation = validation,
                              lambda = lambda,
                              alpha = alpha,
                              batchsize = batchsize,
                              shuffle = shuffle,
                              formula = formula)


  out <- list()
  class(out) <- "citornn"
  out$net <- build_rnn(type = type, num_layers = num_layers,
                       input_size = ncol(X), output_size = y_dim,
                       hidden_size = hidden_size, bias = bias,
                       dropout = dropout)
  out$call <- match.call()
  out$loss <- loss_obj
  out$data <- list(X = X, Y = Y, data = data)
  if(validation != 0) out$data <- append(out$data, list(validation = valid))
  out$weights <- list()
  out$use_model_epoch <- 0
  out$loaded_model_epoch <- 0
  out$model_properties <- model_properties
  out$training_properties <- training_properties


  ### training loop ###
  #out <- train_model(model = out,epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose)


  allglobal()

  return(out)
}

allglobal <- function() {
  lss <- ls(envir = parent.frame())
  for (i in lss) {
    assign(i, get(i, envir = parent.frame()), envir = .GlobalEnv)
  }
}



