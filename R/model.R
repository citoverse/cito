get_data_loader = function(X, Y, batch_size=25L, shuffle=TRUE, x_dtype, y_dtype) {
  self = NULL
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
  ds <- torch.dataset(X,Y)
  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)
  return(dl)
}




build_model = function(input, output, hidden, activation, bias, dropout) {
  layers = list()
  if(is.null(hidden)) {
    layers[[1]] = torch::nn_linear(input, out_features = output,bias = FALSE)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden) != length(bias)) bias = rep(bias, (length(hidden)+1))
    if(length(hidden) != length(dropout)) dropout = rep(dropout,length(hidden))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        layers[[1]] = torch::nn_linear(input, out_features = hidden[1], bias = FALSE)
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i-1])
      }
      counter = counter+1
      if(activation[i] == "relu") layers[[counter]] = torch::nn_relu()
      if(activation[i] == "leaky_relu") layers[[counter]] = torch::nn_leaky_relu()
      if(activation[i] == "tanh") layers[[counter]] = torch::nn_tanh()
      counter = counter+1
      if(dropout[i]>0) {
        layers[[counter]] = torch::nn_dropout(dropout[i])
        counter = counter+1
      }
    }
    layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = output, bias = bias[i])
  }
  net = do.call(torch::nn_sequential, layers)
  return(net)
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
