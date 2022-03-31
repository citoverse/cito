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
      layers[[counter]]<- switch(tolower(activation[i]),
             "relu" = torch::nn_relu(),
             "leaky_relu" = torch::nn_leaky_relu(),
             "tanh" = torch::nn_tanh(),
             "elu" = torch::nn_elu(),
             "rrelu" = torch::nn_rrelu(),
             "prelu" = torch::nn_prelu(),
             "softplus" = torch::nn_softplus(),
             "celu" = torch::nn_celu(),
             "selu" = torch::nn_selu(),
             "gelu" = torch::nn_gelu(),
             "relu6" = torch:: nn_relu6(),
             "sigmoid" = torch::nn_sigmoid(),
             "softsign" = torch::nn_softsign(),
             "hardtanh" = torch::nn_hardtanh(),
             "tanhshrink" = torch::nn_tanhshrink(),
             "softshrink" = torch::nn_softshrink(),
             "hardshrink" = torch::nn_hardshrink(),
             "log_sigmoid" = torch::nn_log_sigmoid(),
             stop(paste0(activation[i], " as an activation function is not supported"))
             )

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


generalize_alpha<- function (parameters, alpha, loss){

  weight_layers<-  grepl("weight",names(parameters),fixed=T)
  if(!all(alpha==F)|anyNA(alpha)){
    counter<- 1
    if(is.na(alpha[counter])){
      counter<- counter+1
    }else{
      if(colnames(X)[1]=="(Intercept)"){
        l1 <- torch::torch_sum(torch::torch_abs(torch::torch_cat(parameters$`0.weight`$hsplit(1)[[2]])))
        l2 <- torch::torch_norm(torch::torch_cat(parameters$`0.weight`$hsplit(1)[[2]]),p=2L)
        regularization <- ((1-alpha[counter])* l1) + (alpha[counter]* l2)
        loss<-  torch::torch_add(loss,regularization)
        counter<- counter + 1
      }
    }
    for (i in c(counter:length(parameters))){
      if(is.na(alpha[counter])){
        counter<- counter+1
      }else{
        if (weight_layers[i]){
          l1 <- torch::torch_sum(torch::torch_abs(torch::torch_cat(parameters[i])))
          l2 <- torch::torch_norm(torch::torch_cat(parameters[i]),p=2L)
          regularization <- ((1-alpha[counter])* l1) + (alpha[counter]* l2)
          loss<-  torch::torch_add(loss,regularization)
          counter <- counter+1
        }
      }
    }
  }


  return(loss)
}
