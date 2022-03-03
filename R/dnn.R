#' DNN
#'
#' @param formula formula object
#' @param data matrix or data.frame
#' @param family error distribution with link function, see details for supported family functions
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of activation functions for each layer. Currently supported: tanh, relu, leakyrelu, selu, or sigmoid
#' @param validation percantage of data set that should be taken as validation set (chosen randomly)
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers + 1 (last layer)) of logicals for each layer.
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2}
#' @param dropout probability of dropout rate
#' @param optimizer which optimizer used for training the network,
#' @param lr learning rate given to optimizer
#' @param batchsize how many samples data loader loads per batch
#' @param shuffle TRUE if data should be reshuffled every epoch (default: FALSE)
#' @param epochs epochs for training loop
#' @param ... additional arguments to be passed to optimizer
#'
#' @import checkmate
#' @export
dnn = function(formula,
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
               ...) {
  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
  checkmate::qassert(activation, "S+[1,)")
  checkmate::qassert(bias, "B+")
  checkmate::qassert(lambda, "R1[0,)")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(alpha, "R1[0,)")
  checkmate::qassert(dropout, "R1[0,)")

  self = NULL

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
    if(fam$family$family == "softmax") y_dtype = torch::torch_long()
    if(fam$family$family == "binomial") {
      Y = torch::as_array(torch::nnf_one_hot( torch::torch_tensor(Y, dtype=torch::torch_long() ))$squeeze() )
    }
  }

  ### dataset  ###
  torch.dataset <- torch::dataset(
    name = "dataset",
    initialize = function(X,Y) {
      self$X <- torch::torch_tensor(as.matrix(X), dtype = x_dtype)
      self$Y <- torch::torch_tensor(as.matrix(Y), dtype = y_dtype)
    },
    .getitem = function(index) {
      x <- self$X[index,]
      y <- self$Y[index,]
      list(x, y)
    },
    .length = function() {
      self$Y$size()[[1]]
    }
  )
  if(validation!= 0){
    train <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(validation*nrow(X))))
    valid <- c(1:nrow(X))[-train]
    train_ds <- torch.dataset(X[train,],Y[train,])
    train_dl <- torch::dataloader(train_ds, batch_size = batchsize, shuffle = shuffle)
    valid_ds <- torch.dataset(X[valid,],Y[valid,])
    valid_dl <- torch::dataloader(valid_ds, batch_size = batchsize, shuffle = shuffle)
  }else{
    train_ds <- torch.dataset(X,Y)
    train_dl <- torch::dataloader(train_ds, batch_size= batchsize, shuffle= shuffle)
  }


  ### build model ###
  # bias in first layer is set by formula intercept
  layers = list()
  if(is.null(hidden)) {
    layers[[1]] = torch::nn_linear(ncol(X), out_features = y_dim,bias = FALSE)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden) != length(bias)) bias = rep(bias, (length(hidden)+1))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        layers[[1]] = torch::nn_linear(ncol(X), out_features = hidden[1], bias = FALSE)
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i-1])
      }
      counter = counter+1
      if(activation[i] == "relu") layers[[counter]] = torch::nn_relu()
      if(activation[i] == "leaky_relu") layers[[counter]] = torch::nn_leaky_relu()
      if(activation[i] == "tanh") layers[[counter]] = torch::nn_tanh()
      counter = counter+1
    }
    layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = y_dim, bias = bias[i])
  }
  net = do.call(torch::nn_sequential, layers)

  parameters = c(net$parameters, fam$parameter)

  ### set optimizer ###
  optim<- switch(tolower(optimizer),
                 "adam"= torch::optim_adam(parameters, lr=lr,...),
                 "adadelta" = torch::optim_adadelta(parameters, lr=lr,...),
                 "adagrad" =  torch::optim_adagrad(parameters, lr=lr,...),
                 "rmsprop"  = torch::optim_rmsprop(parameters, lr=lr,...),
                 "rprop" = torch::optim_rprop(parameters, lr=lr,...),
                 "sgd" = torch::optim_sgd(parameters, lr=lr,...),
                 "lbfgs" = torch::optim_lbfgs(parameters, lr=lr,...)

  )

  ### training loop ###

  loss.fkt<- fam$loss
  losses<- data.frame(epoch=c(1:epochs),train_l=NA,valid_l= NA)
  for (epoch in 1:epochs) {
    train_l <- c()

    coro::loop(for (b in train_dl) {
      optim$zero_grad()
      output <- net(b[[1]])
      loss <- loss.fkt(output, b[[2]])$mean()
      loss$backward()
      optim$step()
      train_l <- c(train_l, loss$item())
      losses$train_l[epoch]<- mean(train_l)
    })

    if(validation!= 0){

      valid_l <- c()

      coro::loop(for (b in valid_dl) {
        output <- net(b[[1]])
        loss <- loss.fkt(output, b[[2]])$mean()
        valid_l <- c(valid_l, loss$item())
        losses$valid_l[epoch]<- mean(valid_l)
      })
      cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f\n",
                  epoch, losses$train_l[epoch], losses$valid_l[epoch]))
    }else{
      cat(sprintf("Loss at epoch %d: %3f\n", epoch, losses$train_l[epoch]))
    }

    ### create plot ###
    visualize.training(losses,epoch)

   }

  z<- list()
  class(z)<- "citodnn"
  z$net<- net
  z$call <- match.call()
  z$family = fam

  return(z)
}

#' Predict from a fitted dnn model
#'
#' @param object a model created by \code{\link{dnn}}
#' @param newdata newdata for predictions
#'
#' @return prediction matrix
#'
#' @export
predict.citodnn = function(object, newdata = NULL,...) {

  checkmate::assert( checkmate::checkNull(newdata),
                     checkmate::checkMatrix(newdata),
                     checkmate::checkDataFrame(newdata))

  if(is.data.frame(newdata)) {
    newdata <- stats::model.matrix(as.formula(object$call$formula), newdata)
  } else {
    newdata <- stats::model.matrix(as.formula(object$call$formula), data.frame(newdata))
  }
  newdata<- torch::torch_tensor(newdata)

  pred = object$net(newdata,...)
  return(pred)
}


visualize.training <- function(losses,epoch){
  if (epoch==1){

    graphics::plot(c(),c(),xlim=c(1,nrow(losses)),ylim=c(0,max(losses$train_l[1],losses$valid_l[1],na.rm=T)),
         main= "Training of DNN",
         xlab= "epoch",
         ylab= "loss")
           col= c("#000080","#FF8000"),lty=1:2, cex=0.8,
           title="Line types", text.font=4, bg='grey91')

    graphics::points(x=c(1),y=c(losses$train_l[1]),pch=19, col="#000080", lty=1)
    graphics::points(x=c(1),y=c(losses$valid_l[1]),pch=18, col="#FF8000", lty=2)

  } else{

  graphics::lines(c(epoch-1,epoch), c(losses$train_l[epoch-1],losses$train_l[epoch]), pch=19, col="#000080", type="b", lty=1)
  graphics::lines(c(epoch-1,epoch), c(losses$valid_l[epoch-1],losses$valid_l[epoch]), pch=18, col="#FF8000", type="b", lty=2)
  }
}

get_family = function(family) {
  out = list()
  out$parameter = NULL

  if(!inherits(family, "family")){
    if(family == "softmax") {
      family = list(family="softmax")
    } else {
      stop("Family is not supported")
    }
  }
  if(family$family == "gaussian") {
    out$parameter = torch::torch_tensor(0.1, requires_grad = TRUE)
    out$invlink = function(a) a
    out$loss = function(pred, true) {
      return(torch::distr_normal(pred, torch::torch_clamp(out$parameter, 0.0001, 20))$log_prob(true)$negative())
    }
  } else if(family$family == "binomial") {
    if(family$link == "logit") {
      out$invlink = function(a) torch::torch_sigmoid(a)
    } else if(family$link == "probit")  {
      out$invlink = function(a) torch::torch_sigmoid(a*1.7012)
    } else {
      out$invlink = function(a) a
    }
    out$loss = function(pred, true) {
      return(torch::distr_bernoulli( out$invlink(pred) )$log_prob(true)$negative())
    }
  } else if(family$family == "poisson") {
    if(family$link == "log") {
      out$invlink = function(a) torch::torch_exp(a)
    } else {
      out$invlink = function(a) a
    }
    out$loss = function(pred, true) {
      return(torch::distr_poisson( out$invlink(pred) )$log_prob(true)$negative())
    }
  } else if(family$family == "softmax") {
    out$invlink = function(a) torch::nnf_softmax(a, dim = 2)
    out$loss = function(pred, true) {
        return( torch::nnf_cross_entropy(pred, true$squeeze(), reduction = "none"))
      }
  }

  out$family = family
  return(out)
}



#res = dnn(Species~scale(Sepal.Width)+scale(Petal.Length), data = iris, family = "softmax")
#summary(lm(scale(Sepal.Length)~scale(Sepal.Width)+scale(Petal.Length), data = iris))

