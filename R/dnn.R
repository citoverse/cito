#' DNN
#'
#' @description
#'
#' dnn is used to fit a deep neural network. The function supports the formula syntax as well as different response families (see details).
#'
#' @param formula an object of class "\code{\link[stats]{formula}}": a description of the model that should be fitted
#' @param data matrix or data.frame
#' @param family error distribution with link function or standard loss function
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of different activation functions for each layer
#' @param validation percentage of data set that should be taken as validation set (chosen randomly)
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
#' @param device device on which network should be trained on.
#' @param early_stopping if set to integer, training will stop if validation loss worsened between current defined past epoch.
#'
#'
#' @return an S3 object of class \code{"cito.dnn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_sequential" "nn_module", originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call}
#' \item{family}{A list which contains relevant information for the target variable}
#' \item{losses}{A data.frame containing training and validation losses of each epoch}
#' \item{data}{Contains data used for training the model}
#' \item{weigths}{List of weights for each training epoch}
#' @import checkmate
#' @example /inst/examples/dnn-example.R
#' @seealso \code{\link{predict.citodnn}}, \code{\link{plot.citodnn}},  \code{\link{coef.citodnn}},\code{\link{print.citodnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#' @export
dnn <- function(formula,
                data = NULL,
                family = stats::gaussian(),
                hidden = c(10L, 10L, 10L),
                activation = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                               "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                               "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"),
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
                lr_scheduler = FALSE,
                device = c("cpu","cuda"),
                early_stopping = FALSE) {
  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
  checkmate::qassert(activation, "S+[1,)")
  checkmate::qassert(bias, "B+")
  checkmate::qassert(lambda, "R1[0,)")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(dropout, "R+[0,)")
  checkmate::qassert(lr, "R+[0,)")
  checkmate::qassert(plot,"B1")
  checkmate::qassert(early_stopping,c("R1[1,)","B1"))
  checkmate::qassert(device, "S+[3,)")

  device <- match.arg(device)
  if(identical (activation, c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                              "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                              "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"))) activation<- "relu"



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


  fam <- get_family(family)

  y_dim <- ncol(Y)
  x_dtype <- torch::torch_float32()
  y_dtype <- torch::torch_float32()
  if(is.character(Y)) {
    y_dim <- length(unique(as.integer(as.factor(Y[,1]))))
    Y <- matrix(as.integer(as.factor(Y[,1])), ncol = 1L)
    if(fam$family$family == "binomial") {
      Y <- torch::as_array(torch::nnf_one_hot(torch::torch_tensor(Y, dtype=torch::torch_long() ))$squeeze())
    }
  }
  if(fam$family$family == "softmax") y_dtype = torch::torch_long()


  ### dataloader  ###
  if(validation != 0){
    valid <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(validation*nrow(X))))
    train <- c(1:nrow(X))[-valid]
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



  ### optimizer ###
  optim <- get_optimizer(optimizer = optimizer,
                         parameters = c(net$parameters, fam$parameter),
                         lr = lr)

  ### LR Scheduler ###
  if(!isFALSE(lr_scheduler)){
    use_lr_scheduler <- TRUE
    scheduler <- get_lr_scheduler(lr_scheduler = lr_scheduler, optimizer = optim)
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
                               loss = loss, lambda = lambda,
                               intercept= colnames(X)[1]=="(Intercept)")
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
                                 loss = loss, lambda = lambda,
                                 intercept= colnames(X)[1]=="(Intercept)")
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


  model_properties <- list(input = ncol(X),
                           output = y_dim,
                           hidden = hidden,
                           activation = activation,
                           bias = bias,
                           dropout = dropout)

  training_properties<- list(lr = lr,
                             lr_scheduler = lr_scheduler,
                             optimizer = optimizer,
                             epochs = epoch,
                             early_stopping = early_stopping,
                             plot = plot,
                             validation = validation,
                             lambda = lambda,
                             alpha = alpha,
                             batchsize = batchsize,
                             shuffle = shuffle,
                             formula = formula)


  out <- list()
  class(out) <- "citodnn"
  out$net <- net
  out$call <- match.call()
  out$family <- fam
  out$called_with_family <- family
  out$losses<- losses
  out$data <- list(X = X, Y = Y, data = data)
  if(validation != 0) out$data<- append(out$data, list(validation = valid))
  out$weights <- weights
  out$use_model_epoch <- epoch
  out$loaded_model_epoch <- epoch
  out$model_properties <- model_properties
  out$training_properties <- training_properties

  return(out)
}

#' Print class citodnn
#'
#' @param x a model created by \code{\link{dnn}}
#' @param ... additional arguments
#' @return prediction matrix
#' @example /inst/examples/print.citodnn-example.R
#' @export
print.citodnn <- function(x,...){
  x <- check_model(x)
  print(x$call)
  print(x$net)
}

#' Summarize Neural Network of class citodnn
#'
#' Performs a Feature Importance calculation based on Permutations
#'
#' @details
#'
#' Performs the feautre importance calculation as suggested by  Fisher, Rudin, and Dominici (2018).
#' For each feature n permutation get done and original and permuted predictive mean squarred error (\eqn{e_{perm}} & \eqn{e_{orig}}) get evaluated with \eqn{ FI_j= e_{perm}/e_{orig}}. Based on Mean Squarred Error.
#'
#' @param object a model of class citodnn created by \code{\link{dnn}}
#' @param n_permute number of permutations performed, higher euqals more accurate importance results
#' @param ... additional arguments
#' @return summary.glm returns an object of class "summary.citodnn", a list with components
#' @export
summary.citodnn <- function(object, n_permute = 256, ...){
  object <- check_model(object)
  out <- list()
  class(out) <- "summary.citodnn"
  out$importance <- get_importance(object, n_permute)

  return(out)
}



#' Print method for class summary.citodnn
#'
#' @param x a summary object created by \code{\link{summary.citodnn}}
#' @param ... additional arguments
#' @export
print.summary.citodnn <- function(x, ... ){
print("Deep Neural Network Model summary")
print("Feature Importance:")
print(x$importance)

}


#' Returns list of parameters the neural network model currently has in use
#'
#' @param object a model created by \code{\link{dnn}}
#' @param ... nothing implemented yet
#' @return list of weigths of neural network
#'
#' @example /inst/examples/coef.citodnn-example.R
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
#' @example /inst/examples/predict.citodnn-example.R
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

#' Creates graph plot which gives an overview of the network architecture.
#'
#' @param x a model created by \code{\link{dnn}}
#' @param node_size size of node in plot
#' @param scale_edges edge weight gets scaled according to other weights (layer specific)
#' @param ... no further functionality implemented yet
#'
#' @example /inst/examples/plot.citodnn-example.R
#' @export
plot.citodnn<- function(x, node_size = 1, scale_edges = FALSE,...){

  sapply(c("igraph","ggraph","ggplot2"),function(x)
    if (!requireNamespace(x, quietly = TRUE)) {
      stop(
        paste0("Package \"",x,"\" must be installed to use this function."),
        call. = FALSE
      )
    })
  checkmate::qassert(node_size, "R+[0,)")
  checkmate::qassert(scale_edges, "B1")

  weights <- coef.citodnn(x)
  input <- ncol(weights[[1]][1][[1]])-1
  structure <- data.frame(expand.grid(from=paste0("1;",c(1:input)),
                                      to = paste0("2;",c(1:(nrow(weights[[1]][1][[1]]))))),
                          value = scale(c(t(weights[[1]][1][[1]][1:input])), center=scale_edges,scale= scale_edges))
  x_pos<- c(rep(1,input))
  y_pos<- c(0,rep(1:input,each=2) *c(1,-1))[1:input]
  num_layer <-  2

  if(length(weights[[1]])>1){
    for (i in 2:length(weights[[1]])){
      if (grepl("weight",names(weights[[1]][i]))){
        structure <- rbind(structure, data.frame(expand.grid(from=paste0(num_layer,";",c(1:(ncol(weights[[1]][i][[1]])))),
                                                             to = paste0(num_layer + 1,";",c(1:(nrow(weights[[1]][i][[1]]))))),
                                                 value= scale(c(t(weights[[1]][i][[1]])), center=scale_edges,scale= scale_edges)))
        x_pos <- c(x_pos, rep(num_layer, x$model_properties$hidden[num_layer-1]))
        y_pos <- c(y_pos, c(0,rep(1:x$model_properties$hidden[num_layer-1],each=2) *c(1,-1))[1:x$model_properties$hidden[num_layer-1]])
        num_layer <- num_layer + 1

      }
    }
  }
  x_pos <- c(x_pos, rep(num_layer,x$model_properties$output))
  y_pos <- c(y_pos, c(0,rep(1:input,each=2) *c(1,-1))[1:x$model_properties$output])


  graph<- igraph::graph_from_data_frame(structure)
  layout <- ggraph::create_layout(graph, layout= "manual", x = x_pos, y = y_pos)

  p<- ggraph::ggraph(layout)+
    ggraph::geom_edge_link( ggplot2::aes(width = abs(structure$value))) +
    ggraph::geom_node_point(size = node_size)
  print(p)
}
