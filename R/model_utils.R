format_targets <- function(Y, loss_obj, ylvls=NULL) {

  if(is.null(ylvls) && is.factor(Y)) ylvls <- levels(Y)
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


get_data_loader = function(..., batch_size=25L, shuffle=TRUE) {

  ds <- torch::tensor_dataset(...)

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

#Output shapes of the avgpool layers right before the classifier
get_transfer_output_shape <- function(name) {
  return(switch(name,
         "alexnet" = c(256, 6, 6),
         "inception_v3" = c(2048, 1, 1),
         "mobilenet_v2" = c(1280, 1, 1),
         "resnet101" = c(2048, 1, 1),
         "resnet152" = c(2048, 1, 1),
         "resnet18" = c(512, 1, 1),
         "resnet34" = c(512, 1, 1),
         "resnet50" = c(2048, 1, 1),
         "resnext101_32x8d" = c(2048, 1, 1),
         "resnext50_32x4d" = c(2048, 1, 1),
         "vgg11" = c(512, 7, 7),
         "vgg11_bn" = c(512, 7, 7),
         "vgg13" = c(512, 7, 7),
         "vgg13_bn" = c(512, 7, 7),
         "vgg16" = c(512, 7, 7),
         "vgg16_bn" = c(512, 7, 7),
         "vgg19" = c(512, 7, 7),
         "vgg19_bn" = c(512, 7, 7),
         "wide_resnet101_2" = c(2048, 1, 1),
         "wide_resnet50_2" = c(2048, 1, 1),
         stop(paste0(name, " not supported."))))
}

# Load the pretrained models.
# In inception_v3 the auxiliary part is omitted since we don't use it, and the input transformation is moved to the forward function since we always use pretrained weights.
# In mobilenet_v2 the global average pool is moved from the forward function to a module, so the last 2 modules of all models are avgpool and classifier, respectively.
get_pretrained_model <- function(transfer, pretrained) {
  if(transfer == "inception_v3") {
    inception_v3 <- torchvision::model_inception_v3(pretrained = pretrained)

    forward <- c(sub("        ", "    ", deparse(inception_v3$.transform_input)[c(1,2,4:9)]),
                 "    x <- torch::torch_cat(list(x_ch0, x_ch1, x_ch2), 2)",
                 deparse(inception_v3$.forward)[c(3:17, 24:30)], "    x", "}")

    torch_model <- torch::nn_module(
      classname = "inception_v3",
      initialize = function(inception_v3) {
        for (child in names(inception_v3$children)) {
          if(child != "AuxLogits") {
            eval(parse(text=paste0("self$", child, " <- inception_v3$", child)))
          }
        }
      },
      forward = eval(parse(text=forward))
    )(inception_v3)

  } else if(transfer == "mobilenet_v2") {
    mobilenet_v2 <- torchvision::model_mobilenet_v2(pretrained = pretrained)

    forward <- deparse(mobilenet_v2$forward)
    forward[4] <- "    x <- self$avgpool(x)"

    torch_model <- torch::nn_module(
      classname = "mobilenet_v2",
      initialize = function(mobilenet_v2) {
        self$features <- mobilenet_v2$features
        self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
        self$classifier <- mobilenet_v2$classifier
      },
      forward = eval(parse(text=forward))
    )(mobilenet_v2)

  } else {
    eval(parse(text = paste0("torch_model <- torchvision::model_", transfer, "(pretrained = pretrained)")))
  }
  return(torch_model)
}

replace_output_layer <- function(transfer_model, output_shape) {
  n <- length(transfer_model$children)
  if(names(transfer_model$children)[n] == "fc") {
    if(is.null(output_shape)) {
      transfer_model$fc <- torch::nn_identity()
    } else {
      transfer_model$fc <- torch::nn_linear(transfer_model$fc$in_features, output_shape)
    }
  } else if(names(transfer_model$children)[n] == "classifier") {
    classifier <- transfer_model$classifier
    k <- length(classifier$children)
    if(is.null(output_shape)) {
      classifier[[names(classifier$children)[k]]] <- torch::nn_identity()
    } else {
      classifier[[names(classifier$children)[k]]] <- torch::nn_linear(classifier[[names(classifier$children)[k]]]$in_features, output_shape)
    }
    transfer_model$classifier <- classifier
  } else {
    stop("Error in replacing output layer of pretrained model")
  }
  return(transfer_model)
}

replace_classifier <- function(transfer_model, cito_model) {

  forward <- deparse(transfer_model$forward)
  forward <- c(forward[1:(which(grepl("flatten", forward))-1)], "    x <- self$cito(x)", "    x", "}")

  net <- torch::nn_module(
    initialize = function(transfer_model, cito_model) {
      for (child in names(transfer_model$children)[-length(transfer_model$children)]) {
        eval(parse(text=paste0("self$", child, " <- transfer_model$", child)))
      }
      self$cito <- cito_model
    },
    forward = eval(parse(text=forward))
  )

  return(net(transfer_model, cito_model))
}

 freeze_weights <- function(transfer_model) {
   for(child in transfer_model$children[-length(transfer_model$children)]) {
     for(parameter in child$parameters) {
       parameter$requires_grad_(FALSE)
     }
   }
   return(transfer_model)
 }


