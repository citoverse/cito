format_targets <- function(Y, loss_obj, ylvls=NULL) {

  if(!inherits(Y, "matrix")) Y = as.matrix(Y)
  responses <- colnames(Y)

  if(inherits(loss_obj$call, "family") && loss_obj$call$family == "binomial") {
    if(all(Y %in% c(0,1))) {
      Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())

    } else if(is.character(Y)) {
      if (is.null(ylvls)) {
        Y <- factor(Y[,1])
        ylvls <- levels(Y)
      } else {
        Y <- factor(Y[,1], levels = ylvls)
      }
      Y <- torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32())

    } else {
      Y <- as.integer(Y[,1])
      Y <- torch::torch_tensor(torch::nnf_one_hot(torch::torch_tensor(Y, dtype = torch::torch_long())), dtype = torch::torch_float32())
    }
    y_dim <- ncol(Y)
    Y_base <- torch::torch_tensor(matrix(apply(as.matrix(Y), 2, mean), nrow(Y), ncol(Y), byrow = TRUE))

  } else if(!is.function(loss_obj$call) && all(loss_obj$call == "softmax")) {
    if (is.character(Y)) {
      if (is.null(ylvls)) {
        Y <- factor(Y[,1])
        ylvls <- levels(Y)
      } else {
        Y <- factor(Y[,1], levels = ylvls)
      }
      Y <- as.matrix(as.integer(Y), ncol=1L)
    } else {
      Y <- as.matrix(as.integer(Y[,1]), ncol=1L)
    }
    y_dim <- length(unique(Y))
    prop <- as.vector(table(Y)/sum(table(Y)))
    Y_base <- matrix(prop, nrow = nrow(Y), ncol = length(prop), byrow = TRUE)
    Y <- torch::torch_tensor(Y, dtype = torch::torch_long())
  } else {
    y_dim <- ncol(Y)
    Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
    Y_base = torch::torch_tensor(matrix(apply(as.matrix(Y), 2, mean), nrow(Y), ncol(Y), byrow = TRUE))
  }

  if(!is.null(ylvls)) responses <- ylvls

  return(list(Y=Y, Y_base=Y_base, y_dim=y_dim, ylvls=ylvls, responses=responses))
}


get_data_loader = function(X, Y, batch_size=25L, shuffle=TRUE) {

  ds <- torch::tensor_dataset(X = X, Y=Y)

  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = TRUE)

  return(dl)
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
        out$link <- function(a) a
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

get_activation_layer <- function(activation) {
  return(switch(tolower(activation),
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
         stop(paste0(activation, " as an activation function is not supported"))
  ))
}

get_var_names <- function(formula, data){
  X_helper <- stats::model.matrix(formula,data[1,])
  var_names <- c()
  for(i in seq_len(ncol(data))){
    if(colnames(data)[i]%in%colnames(X_helper)){
      var_names<- append(var_names, colnames(data)[i])

    }else if (is.factor(data[,i])){
      count <- startsWith(colnames(X_helper),colnames(data)[i])
      count <- sum(count, na.rm = TRUE) + 1
      if(count >= nlevels(data[,i])){
        var_names<- append(var_names, colnames(data)[i])

      }
    }
  }
  return(var_names)
}

get_output_shape <- function(input_shape, n_kernels, kernel_size, stride, padding, dilation) {
  input_shape[1] <- n_kernels
  for(i in 2:length(input_shape)) {
    l <- input_shape[i] + 2*padding[i-1]
    k <- kernel_size[i-1] + (kernel_size[i-1]-1)*(dilation[i-1]-1)
    s <- stride[i-1]
    input_shape[i] <- floor((l-k)/s)+1
  }
  return(input_shape)
}


adjust_architecture <- function(architecture, input_dim) {

  adjusted_architecture <- list()
  for(layer in architecture) {
    if(class(layer)[1] %in% c("avgPool", "maxPool")) {
      if(is.null(layer$stride)) layer$stride <- layer$kernel_size
    }

    if(input_dim != 1) {
      if(class(layer)[1] %in% c("conv", "avgPool", "maxPool")) {
        if(length(layer$kernel_size) == 1) layer$kernel_size <- rep(layer$kernel_size, input_dim)
        if(length(layer$stride) == 1) layer$stride <- rep(layer$stride, input_dim)
        if(length(layer$padding) == 1) layer$padding <- rep(layer$padding, input_dim)
      }

      if(class(layer)[1] %in% c("conv", "maxPool")) {
        if(length(layer$dilation) == 1) layer$dilation <- rep(layer$dilation, input_dim)
      }
    }
    adjusted_architecture <- append(adjusted_architecture, list(layer))
  }
  class(adjusted_architecture) <- "citoarchitecture"
  return(adjusted_architecture)
}

