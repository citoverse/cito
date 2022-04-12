#' DNN
#'
#' @param formula formula object
#' @param data matrix or data.frame
#' @param family error distribution with link function, see details for supported family functions
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of activation functions for each layer. Currently supported: tanh, relu, leakyrelu, selu, or sigmoid
#' @param validation percentage of data set that should be taken as validation set (chosen randomly)
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers + 1 (last layer)) of logicals for each layer.
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha add L1/L2 regularization to training  \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2} will get added for each layer. Can be single integer between 0 and 1 or vector of alpha values if layers should be regularized differently.
#' @param dropout probability of dropout rate
#' @param optimizer which optimizer used for training the network,
#' @param lr learning rate given to optimizer
#' @param batchsize how many samples data loader loads per batch
#' @param shuffle TRUE if data should be reshuffled every epoch (default: FALSE)
#' @param epochs epochs for training loop
#' @param lr_scheduler learning rate scheduler, can be "lambda", "multiplicative", "one_cycle" or "step" additional arguments bust be defined in config with "lr_scheduler." as prefix
#' @param plot plot training loss
#' @param device device on which network should be trained on, either "cpu" or "cuda"
#' @param early_stopping training will stop if validation loss worsened between current and past epoch, function expects the range how far back the comparison should be done
#' @param config list of additional arguments to be passed to optimizer or lr_scheduler should start with optimizer. and lr_scheduler.
#'
#'
#' @return an object of class \code{"cito.dnn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_sequential" "nn_module", originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call}
#' \item{family}{A list which contains relevant information for the target variable}
#' \item{losses}{A data.frame containing training and validation losses of each epoch}
#' \item{data}{Contains data used for training the model}
#' \item{weigths}{List of weights for each training epoch}
#' @import checkmate
#' @examples
#' \dontrun{
#' library(cito)
#'
#' set.seed(222)
#' validation_set<- sample(c(1:nrow(datasets::iris)),25)
#'
#' # Build and train  Network
#' nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])
#'
#' # Sturcture of Neural Network
#' print(nn.fit)
#'
#' # Use model on validation set
#' predictions <- predict(nn.fit, iris[validation_set,])
#'
#' # Scatterplot
#' plot(iris[validation_set,]$Sepal.Length,predictions)
#' # MAE
#' mean(abs(predictions-iris[validation_set,]$Sepal.Length))
#' }
#' @export
dnn <- function(formula,
               data = NULL,
               family = stats::gaussian(),
               hidden = c(10L, 10L, 10L),
               activation = "relu",
               validation = 0,
               bias = TRUE,
               lambda = 0.0,
               alpha = 0.5,
               dropout = 0.0,
               optimizer = "adam",
               lr = 0.01,
               batchsize = 32L,
               shuffle = FALSE,
               epochs = 64,
               plot = TRUE,
               lr_scheduler= FALSE,
               device= "cuda",
               early_stopping = FALSE,
               config=list()) {
  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
  checkmate::qassert(activation, "S+[1,)")
  checkmate::qassert(bias, "B+")
  checkmate::qassert(lambda, "R1[0,)")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(dropout, "R+[0,)")
  checkmate::qassert(lr, "R+[0,)")
  checkmate::qassert(lr_scheduler,c("S+[1,)","B1"))
  checkmate::qassert(plot,"B1")
  checkmate::qassert(early_stopping,c("R1[1,)","B1"))
  checkmate::qassert(device, "S+[3,)")

  self = NULL

  ### decipher config list ###
  config_optimizer<-c()
  config_lr_scheduler<- c()
  if(length(config)>0){
    config_optimizer<- config[which(startsWith(tolower(names(config)),"optimizer"))]
    names(config_optimizer)<- sapply(names(config_optimizer),function(x) substring(x,11))

    config_lr_scheduler<- config[which(startsWith(tolower(names(config)),"lr_scheduler"))]
    names(config_lr_scheduler)<- sapply(names(config_lr_scheduler),function(x) substring(x,14))
  }



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

  if(is.data.frame(data)) {

    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
      formula = stats::as.formula(mf[m]$formula)
      X = stats::model.matrix(formula, data)
      Y = stats::model.response(stats::model.frame(formula, data))
      if(!inherits(Y, "matrix")) Y = as.matrix(Y)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula, data)
    }

  } else {

    if(!is.null(formula)) {
      mf = match.call()
      m = match("formula", names(mf))
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
      formula = stats::as.formula(mf[m]$formula)
    } else {
      formula = stats::as.formula("~.")
    }
    data = data.frame(data)
    X = stats::model.matrix(formula, data)
    Y = stats::model.response(stats::model.frame(formula, data))
    Y = as.matrix(Y)
  }

  fam = get_family(family)

  y_dim = ncol(Y)
  x_dtype = torch::torch_float32()
  y_dtype = torch::torch_float32()
  if(is.character(Y)) {
    y_dim = length(unique(as.integer(as.factor(Y[,1]))))
    Y = matrix(as.integer(as.factor(Y[,1])), ncol = 1L)
    if(fam$family$family == "binomial") {
      Y = torch::as_array(torch::nnf_one_hot( torch::torch_tensor(Y, dtype=torch::torch_long() ))$squeeze() )
    }
  }
  if(fam$family$family == "softmax") y_dtype = torch::torch_long()


  ### dataset  ###
  if(validation != 0){
    train <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(validation*nrow(X))))
    valid <- c(1:nrow(X))[-train]
    train_dl <- get_data_loader(X[train,],Y[train,], batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
    valid_dl <- get_data_loader(X[valid,],Y[valid,], batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)

  }else{
    train_dl <- get_data_loader(X,Y, batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
  }


  ### build model ###
  # bias in first layer is set by formula intercept
  net = build_model(input = ncol(X), output = y_dim,
                    hidden = hidden, activation = activation,
                    bias = bias, dropout = dropout)



  ### set optimizer ###
  optim <- get_optimizer(optimizer = optimizer,
                         parameters = c(net$parameters, fam$parameter),
                         lr = lr,
                         config_optimizer = config_optimizer)


    ### LR Scheduler ###
  if(!isFALSE(lr_scheduler)){
    use_lr_scheduler <- TRUE

    param_lr_scheduler<- list(optimizer=optim)
    if(length(config_lr_scheduler)>0) {
      param_lr_scheduler<- c(param_lr_scheduler,config_lr_scheduler)
    }
    scheduler <- switch(tolower(lr_scheduler),
                        "step" = do.call(torch::lr_step,param_lr_scheduler),
                        "one_cycle" = do.call(torch::lr_one_cycle,param_lr_scheduler),
                        "multiplicative" = do.call(torch::lr_multiplicative,param_lr_scheduler),
                        "lambda" = do.call(torch::lr_lambda,param_lr_scheduler),
                        stop(paste0("lr_scheduler = ",lr_scheduler," is not supported, choose between step, one_cycle, multiplicative or lambda")))

  }else{
    use_lr_scheduler <- FALSE
  }

  ### training loop ###
  weights <- list()
  net$to(device = device)
  loss.fkt <- fam$loss
  losses <- data.frame(epoch=c(1:epochs),train_l=NA,valid_l= NA)
  if((length(hidden)+1) != length(alpha)) alpha <- rep(alpha[1],length(hidden)+1)
  for (epoch in 1:epochs) {
    train_l <- c()

    coro::loop(for (b in train_dl) {
      optim$zero_grad()
      output <- net(b[[1]]$to(device = device))

      loss <- loss.fkt(output, b[[2]]$to(device = device))$mean()
      loss <- generalize_alpha(parameters = net$parameters, alpha = alpha,
                               loss = loss,intercept= colnames(X)[1]=="(Intercept)")
      loss$backward()
      optim$step()

      if(use_lr_scheduler) scheduler$step()
      train_l <- c(train_l, loss$item())
      losses$train_l[epoch] <- mean(train_l)
    })

    if(validation != 0){

      valid_l <- c()

      coro::loop(for (b in valid_dl) {
        output <- net(b[[1]]$to(device = device))
        loss <- loss.fkt(output, b[[2]]$to(device = device))$mean()
        loss <- generalize_alpha(parameters = net$parameters, alpha = alpha,
                                 loss = loss,intercept= colnames(X)[1]=="(Intercept)")
        valid_l <- c(valid_l, loss$item())
        losses$valid_l[epoch] <- mean(valid_l)
      })

      cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f, lr: %3.5f\n",
                  epoch, losses$train_l[epoch], losses$valid_l[epoch],optim$param_groups[[1]]$lr))
      if(epoch>early_stopping && is.numeric(early_stopping) &&
         losses$valid_l[epoch-early_stopping]<losses$valid_l[epoch]) {
        weights[[epoch]]<- lapply(net$parameters,function(x) torch::as_array(x$to(device="cpu")))
        if(plot) visualize.training(losses,epoch)
        break
      }

    }else{
      cat(sprintf("Loss at epoch %d: %3f, lr: %3.5f\n", epoch, losses$train_l[epoch],optim$param_groups[[1]]$lr))
    }

    weights[[epoch]] <- lapply(net$parameters,function(x) torch::as_array(x$to(device="cpu")))


    ### create plot ###
    if(plot) visualize.training(losses,epoch)

  }

  ### Pass to global workspace ###
  allglobal <- function() {
    lss <- ls(envir = parent.frame())
    for (i in lss) {
      assign(i, get(i, envir = parent.frame()), envir = .GlobalEnv)
    }
  }
  allglobal()
  net$to(device = "cpu")

  model_properties <- list(input = ncol(X),
                          output = y_dim,
                          hidden = hidden,
                          activation = activation,
                          bias = bias,
                          dropout = dropout)

  z <- list()
  class(z) <- "citodnn"
  z$net <- net
  z$call <- match.call()
  z$family <- fam
  z$losses<- losses
  z$data <- list(X = X, Y = Y, data = data)
  z$weights <- weights
  z$use_model_epoch <- epoch
  z$loaded_model_epoch <- epoch
  z$model_properties <- model_properties

  return(z)
}

#' Print class citodnn
#'
#' @param x a model created by \code{\link{dnn}}
#' @param ... additional arguments
#' @return prediction matrix
#' @examples
#' \dontrun{
#' library(cito)
#'
#' set.seed(222)
#' validation_set<- sample(c(1:nrow(datasets::iris)),25)
#'
#' # Build and train  Network
#' nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])
#'
#' # Sturcture of Neural Network
#' print(nn.fit)
#' }
#' @import checkmate
#' @export
print.citodnn <- function(x,...){
  x <- check_model(x)
  print(x$call)
  print(x$net)
}

#' Returns list of parameters the neural network model currently has in use
#'
#' @param object a model created by \code{\link{dnn}}
#' @param ... nothing implemented yet
#' @return list of weigths of neural network
#'
#' @examples
#' \dontrun{
#' library(cito)
#'
#' set.seed(222)
#' validation_set<- sample(c(1:nrow(datasets::iris)),25)
#'
#' # Build and train  Network
#' nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])
#'
#' # Sturcture of Neural Network
#' print(nn.fit)
#'
#' #analyze weights of Neural Network
#' coef(nn.fit)
#' }
#' @export
coef.citodnn <- function(object,...){
  return(object$weights[object$use_model_epoch])
}


#' Predict from a fitted dnn model
#'
#' @param object a model created by \code{\link{dnn}}
#' @param newdata newdata for predictions
#' @param type link or response
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @examples
#' \dontrun{
#' library(cito)
#'
#' set.seed(222)
#' validation_set<- sample(c(1:nrow(datasets::iris)),25)
#'
#' # Build and train  Network
#' nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])
#'
#' # Use model on validation set
#' predictions <- predict(nn.fit, iris[validation_set,])
#' # Scatterplot
#' plot(iris[validation_set,]$Sepal.Length,predictions)
#' # MAE
#' mean(abs(predictions-iris[validation_set,]$Sepal.Length))
#' }
#' @export
predict.citodnn <- function(object, newdata = NULL,type=c("link", "response"),...) {

  checkmate::assert( checkmate::checkNull(newdata),
                     checkmate::checkMatrix(newdata),
                     checkmate::checkDataFrame(newdata),
                     checkmate::checkScalarNA(newdata))
  object <- check_model(object)

  type = match.arg(type)

  if(type == "link") link = function(a) object$family$invlink(a)
  else link = function(a) a

  ### TO DO: use dataloaders via get_data_loader function
  if(is.null(newdata)) newdata = torch::torch_tensor(object$data$X)
  else {
    if(is.data.frame(newdata)) {
        newdata <- stats::model.matrix(stats::as.formula(object$call$formula), newdata)
      } else {
        newdata <- stats::model.matrix(stats::as.formula(object$call$formula), data.frame(newdata))
      }
    newdata <- torch::torch_tensor(newdata)
  }


  pred <- torch::as_array(link(object$net(newdata,...)))
  return(pred)
}




# source("R/model.R")
# source("R/plot.R")
# source("R/utils.R")
# res <- dnn(Species~Sepal.Length+Petal.Length, hidden=rep(10,10), early_stopping = F,
#             data = iris, family = "softmax", activation= "selu", device ="cpu",
#            validation= 0.3,epochs = 32, alpha = 1, lr_scheduler = F)
# predict(res,iris[1:5,])
# # res = dnn(Sepal.Width~Species +Petal.Length+ I(Petal.Length^2), hidden=rep(10,5), data = iris ,validation= 0.3,epochs =100)
# # predict(res,iris[1:5,])
# # summary(lm(scale(Sepal.Length)~scale(Sepal.Width)+scale(Petal.Length), data = iris))
# # analyze_training(res)
# # predict(res)
# #
# saveRDS(res, "test.RDS")
# res2 <-  readRDS("test.RDS")
# predict(res2,iris[1:5,])
#
# #error
# res2$net$parameters
#
# #no error
# res2<- check_model(res2)
# res2$net$parameters
