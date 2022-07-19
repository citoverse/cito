get_data_loader = function(X, Y, batch_size=25L, shuffle=TRUE, x_dtype, y_dtype) {
  self = NULL
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

build_dnn = function(input, output, hidden, activation, bias, dropout) {
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
