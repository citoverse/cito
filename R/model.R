get_data_loader = function(X, Y, batch_size=25L, shuffle=TRUE,y_dtype) {

  ds <- torch::tensor_dataset(X =torch::torch_tensor(as.matrix(X)),
                              Y = torch::torch_tensor(as.matrix(Y),dtype = y_dtype))

  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = TRUE)


  return(dl)
}





build_model = function(input, output, hidden, activation, bias, dropout) {
  layers = list()
  if(is.null(hidden)) {
    layers[[1]] = torch::nn_linear(input, out_features = output,bias = FALSE)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden)+1 != length(bias)) bias = rep(bias, (length(hidden)+1))
    if(length(hidden) != length(dropout)) dropout = rep(dropout,length(hidden))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        layers[[1]] = torch::nn_linear(input, out_features = hidden[1], bias = FALSE)
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i-1])
      }
      counter = counter+1
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
    layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = output, bias = bias[i+1])
  }
  net = do.call(torch::nn_sequential, layers)
  return(net)
}



get_loss <- function(loss) {
  out <- list()
  out$parameter <- NULL

  if(is.character(loss)) loss <- tolower(loss)
  if(!inherits(loss, "family")& is.character(loss)){
    loss <- switch(loss,
             "gaussian" = stats::gaussian(),
             "binomial" = stats::binomial(),
             "poisson" = stats::poisson(),
             loss
      )
  }

  if(inherits(loss, "family")){
    if(loss$family == "gaussian") {
      out$parameter <- torch::torch_tensor(1.0, requires_grad = TRUE)
      out$invlink <- function(a) a
      out$link <- function(a) a
      out$loss <- function(pred, true) {
        return(torch::distr_normal(pred, torch::torch_clamp(out$parameter, 0.0001, 20))$log_prob(true)$negative())
      }
    } else if(loss$family == "binomial") {
      if(loss$link == "logit") {
        out$invlink <- function(a) torch::torch_sigmoid(a)
        out$link <- function(a) stats::binomial("logit")$linkfun(as.matrix(a))
      } else if(loss$link == "probit")  {
        out$invlink <- function(a) torch::torch_sigmoid(a*1.7012)
        out$link <- function(a) stats::binomial("probit")$linkfun(as.matrix(a))
      } else {
        out$invlink <- function(a) a
        out$link <- function(a) a
      }
      out$loss <- function(pred, true) {
        return(torch::distr_bernoulli(probs = out$invlink(pred))$log_prob(true)$negative())
      }
    } else if(loss$family == "poisson") {
      if(loss$link == "log") {
        out$invlink <- function(a) torch::torch_exp(a)
        out$link <- function(a) log(a)
      } else {
        out$invlink <- function(a) a
        out$invlink <- function(a) a
      }
      out$loss <- function(pred, true) {
        return(torch::distr_poisson( out$invlink(pred) )$log_prob(true)$negative())
      }
    } else { stop("family not supported")}
  } else  if (is.function(loss)){
    if(is.null(formals(loss)$pred) | is.null(formals(loss)$true)){
      stop("loss function has to take two arguments, \"pred\" and \"true\"")
    }
    out$loss <- loss
    out$invlink <- function(a) a
    out$link <- function(a) a
  } else {
    if(loss == "mae"){
      out$invlink <- function(a) a
      out$link <- function(a) a
      out$loss <- function(pred, true) return(torch::nnf_l1_loss(input = pred, target = true))
    }else if(loss == "mse"){
      out$invlink <- function(a) a
      out$link <- function(a) a
      out$loss <- function(pred,true) return(torch::nnf_mse_loss(input= pred, target = true))
    }else if(loss == "softmax" | loss == "cross-entropy") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$link <- function(a) log(a) + log(ncol(a))
      #Y_base = log(Y_base) + log(ncol(Y_base))
      out$loss <- function(pred, true) {
        return(torch::nnf_cross_entropy(pred, true$squeeze(), reduction = "none"))
      }
    } else if(loss == "mvp") {
      df = floor(ncol(Y)/2)
      out$parameter <- torch::torch_tensor(matrix(runif(ncol(Y)*df, -0.001, 0.001), ncol(Y), df), requires_grad = TRUE)
      out$invlink <- function(a) torch::torch_sigmoid(a*1.7012)
      out$link <- function(a) stats::binomial("probit")$linkfun(as.matrix(a))
      out$loss <- function(pred, true) {
        sigma = out$parameter
        Ys = true
        df = ncol(sigma)
        noise = torch::torch_randn(list(100L, nrow(pred), df))
        E = plogisT((torch::torch_einsum("ijk, lk -> ijl", list(noise, sigma))+pred)*1.702)*0.999999+0.0000005
        logprob = -(log(E)*Ys + log(1.0-E)*(1.0-Ys))
        logprob = - logprob$sum(3)
        maxlogprob = torch::torch_amax(logprob, dim = 1)
        Eprob = (exp(logprob-maxlogprob))$mean(dim = 1)
        return((-log(Eprob) - maxlogprob)$mean())
      }
    }
    else{
      cat( "unidentified loss \n")
      }

  }
  out$call <- loss

  return(out)
}


