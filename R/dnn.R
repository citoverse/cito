#' DNN
#'
#' @description
#'
#' fits a custom deep neural network. dnn() supports the formula syntax and allows to customize the neural network to a maximal degree.
#' So far, only Multilayer Perceptrons are possible.
#' @param formula an object of class "\code{\link[stats]{formula}}": a description of the model that should be fitted
#' @param data matrix or data.frame
#' @param loss loss after which network should be optimized. Can also be distribution from the stats package or own function
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
#' @param verbose print training and validation loss of epochs
#' @param custom_parameters List of parameters/variables to be optimized. Can be used in a custom loss function. See Vignette for example.
#' @param device device on which network should be trained on.
#' @param early_stopping if set to integer, training will stop if loss has gotten higher for defined number of epochs in a row, will use validation loss is available.
#'
#' @details
#'
#' In a Multilayer Perceptron (MLP) network every neuron is connected with all neurons of the previous layer and connected to all neurons of the layer afterwards.
#' The value of each neuron is calculated with:
#'
#' \eqn{ a (\sum_j{ w_j * a_j})}
#'
#' Where \eqn{w_j} is the weight and \eqn{a_j} is the value from neuron j to the current one. a() is the activation function, e.g. \eqn{ relu(x) = max(0,x)}
#' As regularization methods there is dropout and elastic net regularization available. These methods help you avoid over fitting.
#'
#' Training on graphic cards:
#' If you want to train on your cuda devide, you have to install the NVIDIA CUDA toolkit version 11.3. and cuDNN 8.4. beforehand. Make sure that you have exactly these versions installed, since it does not work with other version.
#' For more information see [mlverse: 'torch'](https://torch.mlverse.org/docs/articles/installation.html)
#'
#' @return an S3 object of class \code{"cito.dnn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_sequential" "nn_module", originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call}
#' \item{loss}{A list which contains relevant information for the target variable and the used loss function}
#' \item{data}{Contains data used for training the model}
#' \item{weigths}{List of weights for each training epoch}
#' \item{use_model_epoch}{Integer, which defines which model from which training epoch should be used for prediction.}
#' \item{loaded_model_epoch}{Integer, shows which model from which epoch is loaded currently into model$net.}
#' \item{model_properties}{A list of properties of the neural network, contains number of input nodes, number of output nodes, size of hidden layers, activation functions, whether bias is included and if dropout layers are included.}
#' \item{training_properties}{A list of all training parameters that were used the last time the model was trained. It consists of learning rate, information about an learning rate scheduler, information about the optimizer, number of epochs, whether early stopping was used, if plot was active, lambda and alpha for L1/L2 regularization, batchsize, shuffle, was the data set split into validation and training, which formula was used for training and at which epoch did the training stop.}
#' \item{losses}{A data.frame containing training and validation losses of each epoch}
#' @import checkmate
#' @example /inst/examples/dnn-example.R
#' @seealso \code{\link{predict.citodnn}}, \code{\link{plot.citodnn}},  \code{\link{coef.citodnn}},\code{\link{print.citodnn}}, \code{\link{summary.citodnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}, \code{\link{PDP}}, \code{\link{ALE}},
#' @export
dnn <- function(formula,
                data = NULL,
                loss = c("mse", "mae", "softmax", "cross-entropy", "gaussian", "binomial", "poisson"),
                hidden = c(10L, 10L, 10L),
                activation = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                               "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                               "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"),
                validation = 0,
                bias = TRUE,
                lambda = 0.0,
                alpha = 0.5,
                dropout = 0.0,
                optimizer = c("sgd","adam","adadelta", "adagrad", "rmsprop", "rprop"),
                lr = 0.01,
                batchsize = 32L,
                shuffle = TRUE,
                epochs = 100,
                plot = TRUE,
                verbose = TRUE,
                lr_scheduler = NULL,
                custom_parameters = NULL,
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
  checkmate::qassert(verbose,"B1")
  checkmate::qassert(early_stopping,c("R1[1,)","B1"))
  checkmate::qassert(device, "S+[3,)")

  device <- match.arg(device)
  if(identical (activation, c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                              "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                              "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"))) activation<- "relu"
  if(!is.function(loss) & !inherits(loss,"family")){
    loss <- match.arg(loss)
  }


  if(device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  }else {
    if(device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
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
  ylvls <- NULL
  if(is.factor(Y)) ylvls <- levels(Y)
  if(!inherits(Y, "matrix")) Y = as.matrix(Y)


  loss_obj <- get_loss(loss)
  if(!is.null(loss_obj$parameter)) loss_obj$parameter <- list(scale = loss_obj$parameter)
  if(!is.null(custom_parameters)){
    if(!inherits(custom_parameters,"list")){
      warning("custom_parameters has to be list")
    }else{
      custom_parameters<- lapply(custom_parameters,function(x) torch::torch_tensor(x, requires_grad = TRUE, device = device))
      loss_obj$parameter <- append(loss_obj$parameter,unlist(custom_parameters))

      }
  }

  y_dim <- ncol(Y)
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
    valid <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(validation*nrow(X))))
    train <- c(1:nrow(X))[-valid]
    train_dl <- get_data_loader(X[train,],Y[train,], batch_size = batchsize, shuffle = shuffle, y_dtype=y_dtype)
    valid_dl <- get_data_loader(X[valid,],Y[valid,], batch_size = batchsize, shuffle = shuffle, y_dtype=y_dtype)

  }else{
    train_dl <- get_data_loader(X,Y, batch_size = batchsize, shuffle = shuffle, y_dtype=y_dtype)
    valid_dl <- NULL
  }


  net <- build_model(input = ncol(X), output = y_dim,
                    hidden = hidden, activation = activation,
                    bias = bias, dropout = dropout)



  model_properties <- list(input = ncol(X),
                           output = y_dim,
                           hidden = hidden,
                           activation = activation,
                           bias = bias,
                           dropout = dropout)

  training_properties <- list(lr = lr,
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
  class(out) <- "citodnn"
  out$net <- net
  out$call <- match.call()
  out$call$formula <- stats::terms.formula(formula,data = data)
  out$loss <- loss_obj
  out$data <- list(X = X, Y = Y, data = data)
  out$data$xlvls <- lapply(data[,sapply(data, is.factor), drop = F], function(j) levels(j) )
  if(!is.null(ylvls))  {
    out$data$ylvls <- ylvls
    out$data$xlvls <- out$data$xlvls[-which(as.character(formula[2]) == names(out$data$xlvls))]
  }
  if(validation != 0) out$data <- append(out$data, list(validation = valid))
  out$weights <- list()
  out$use_model_epoch <- 0
  out$loaded_model_epoch <- 0
  out$model_properties <- model_properties
  out$training_properties <- training_properties



  ### training loop ###
  out <- train_model(model = out,epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose)


  return(out)
}

#' Print class citodnn
#'
#' @param x a model created by \code{\link{dnn}}
#' @param ... additional arguments
#' @return prediction matrix
#' @example /inst/examples/print.citodnn-example.R
#' @return original object x gets returned
#' @export
print.citodnn <- function(x,...){
  x <- check_model(x)
  print(x$call)
  print(x$net)
  return(invisible(x))
}

#' Extract Model Residuals
#'
#' Returns residuals of training set.
#'
#' @param object a model created by \code{\link{dnn}}
#' @param ... no additional arguments implemented
#' @return residuals of training set
#' @export
residuals.citodnn <- function(object,...){
  object <- check_model(object)
  out <- data.frame(
    true = object$data$Y,
    pred = stats::predict(object, object$data$data)
  )
  return(out)
}





#' Summarize Neural Network of class citodnn
#'
#' Performs a Feature Importance calculation based on Permutations
#'
#' @details
#'
#' Performs the feature importance calculation as suggested by  Fisher, Rudin, and Dominici (2018).
#' For each feature n permutation get done and original and permuted predictive mean squared error (\eqn{e_{perm}} & \eqn{e_{orig}}) get evaluated with \eqn{ FI_j= e_{perm}/e_{orig}}. Based on Mean Squared Error.
#'
#' @param object a model of class citodnn created by \code{\link{dnn}}
#' @param n_permute number of permutations performed. Default is \eqn{3 * \sqrt{n}}, where n euqals then number of samples in the training set
#' @param ... additional arguments
#' @return summary.glm returns an object of class "summary.citodnn", a list with components
#' @export
summary.citodnn <- function(object, n_permute = NULL, ...){
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
#' @return original object x gets returned
#' @export
print.summary.citodnn <- function(x, ... ){
  cat("Deep Neural Network Model summary\n")
  cat("Model generated on basis of: \n")
  #cat(paste(as.character(x$call$formula)[c(2,1,3)],collapse =" ")) # Unncessary, right? Variables names are the column names of the importance matrix
  cat("Feature Importance:\n")
  print(x$importance)
  return(invisible(x))
}


#' Returns list of parameters the neural network model currently has in use
#'
#' @param object a model created by \code{\link{dnn}}
#' @param ... nothing implemented yet
#' @return list of weights of neural network
#'
#' @example /inst/examples/coef.citodnn-example.R
#' @export
coef.citodnn <- function(object,...){
  return(object$weights[object$use_model_epoch])
}


#' Predict from a fitted dnn model
#'
#' @param object a model created by \code{\link{dnn}}
#' @param newdata new data for predictions
#' @param type which value should be calculated, either raw response, output of link function or predicted class (in case of classification)
#' @param device device on which network should be trained on.
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @example /inst/examples/predict.citodnn-example.R
#' @export
predict.citodnn <- function(object, newdata = NULL, type=c("link", "response", "class"),device = c("cpu","cuda"),...) {

  checkmate::assert( checkmate::checkNull(newdata),
                     checkmate::checkMatrix(newdata),
                     checkmate::checkDataFrame(newdata),
                     checkmate::checkScalarNA(newdata))
  object <- check_model(object)

  type <- match.arg(type)

  device <- match.arg(device)

  if(type %in% c("link","class")) {
    link <- object$loss$invlink
  }else{
    link = function(a) a
  }

  if(device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  }else {
    if(device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }

  object$net$to(device = device)


  ### TO DO: use dataloaders via get_data_loader function
  if(is.null(newdata)){
    newdata = torch::torch_tensor(object$data$X, device = device)
  } else {
    if(is.data.frame(newdata)) {
      newdata <- stats::model.matrix(stats::as.formula(stats::delete.response(object$call$formula)), newdata,xlev = object$data$xlvls)
    } else {
      newdata <- stats::model.matrix(stats::as.formula(stats::delete.response(object$call$formula)), data.frame(newdata),xlev = object$data$xlvls)
    }
    newdata <- torch::torch_tensor(newdata, device = device)
  }

  pred <- torch::as_array(link(object$net(newdata))$to(device="cpu"))

  if(!is.null(object$data$ylvls)) colnames(pred) <- object$data$ylvls
  if(type == "class") pred <- as.factor(apply(pred,1, function(x) object$data$ylvls[which.max(x)]))

  rownames(pred) <- rownames(newdata)


  return(pred)
}

#' Creates graph plot which gives an overview of the network architecture.
#'
#' @param x a model created by \code{\link{dnn}}
#' @param node_size size of node in plot
#' @param scale_edges edge weight gets scaled according to other weights (layer specific)
#' @param ... no further functionality implemented yet
#' @return A plot made with 'ggraph' + 'igraph' that represents the neural network
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
