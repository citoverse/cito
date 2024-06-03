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


    ##### TODO move Y preparation to loss objects!!!!

  } else if(!is.function(loss_obj$call) && any(loss_obj$call == "softmax")) {
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
    Y_base <- torch::torch_tensor( matrix(prop, nrow = nrow(Y), ncol = length(prop), byrow = TRUE), dtype = torch::torch_float32() )
    if(any(loss_obj$call == "softmax") ) Y <- torch::torch_tensor(Y, dtype = torch::torch_long())
    else Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())


  }  else if(!is.function(loss_obj$call) && any(loss_obj$call == "multinomial" || loss_obj$call == "clogit" )) {

    if(ncol(Y) > 1.5) {
      Y_base = torch::torch_tensor(matrix(colMeans(Y), nrow = nrow(Y), ncol = ncol(Y), byrow = TRUE), dtype = torch::torch_float32())
      Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
    } else {

      if(is.character(Y)) {
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
      YY = apply(as.matrix(Y), 1, which.max)

      prop <- as.vector(table(YY)/sum(table(YY)))
      Y_base <- torch::torch_tensor( matrix(prop, nrow = nrow(Y), ncol = length(prop), byrow = TRUE), dtype = torch::torch_float32() )
    }

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


#' Multinomial log likelihood
#'
#' @param probs probabilities
#' @param value observed values
#'
#' Multinomial log likelihood
#'
#' @export
multinomial_log_prob = function(probs, value) {
  logits = probs$log()
  log_factorial_n = torch::torch_lgamma(value$sum(-1) + 1)
  log_factorial_xs = torch::torch_lgamma(value + 1)$sum(-1)
  logits[(value == 0) & (logits == -Inf)] = 0
  log_powers = (logits * value)$sum(-1)
  return(log_factorial_n - log_factorial_xs + log_powers)
}

# binomial_log_prob = function(probs, value, total_count) {
#   log_factorial_n = torch_lgamma(total_count + 1)
#   log_factorial_k = torch_lgamma(value + 1)
#   log_factorial_nmk = torch_lgamma(total_count - value + 1)
#
#   normalize_term = (
#     self.total_count * _clamp_by_zero(self.logits)
#     + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
#     - log_factorial_n
#   )
#   return (
#     value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
#   )
#
# }


get_loss <- function(loss, device = "cpu", X = NULL, Y = NULL) {

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
      out$parameter <- torch::torch_tensor(1.0, requires_grad = TRUE, device = device)
      out$parameter_r = as.numeric(out$parameter$cpu())
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
        return(torch::nnf_cross_entropy(pred, true$squeeze(dim = 2), reduction = "none"))
      }
    } else if(loss == "mvp") {

      if(!exists("Y")) Y = matrix(1., 1,1)

      df = floor(ncol(Y)/2)
      out$parameter <- torch::torch_tensor(matrix(stats::runif(ncol(Y)*df, -0.001, 0.001), ncol(Y), df), requires_grad = TRUE, device = device)
      out$invlink <- function(a) torch::torch_sigmoid(a*1.7012)
      out$link <- function(a) stats::binomial("probit")$linkfun(as.matrix(a$cpu()))
      out$loss <- function(pred, true) {
        sigma = out$parameter
        Ys = true
        df = ncol(sigma)
        noise = torch::torch_randn(list(100L, nrow(pred), df), device = device)
        E = torch::torch_sigmoid((torch::torch_einsum("ijk, lk -> ijl", list(noise, sigma))+pred)*1.702)*0.999999+0.0000005
        logprob = -(log(E)*Ys + log(1.0-E)*(1.0-Ys))
        logprob = - logprob$sum(3)
        maxlogprob = torch::torch_amax(logprob, dim = 1)
        Eprob = (exp(logprob-maxlogprob))$mean(dim = 1)
        return((-log(Eprob) - maxlogprob)$mean())
      }
    } else if(loss == "nbinom") {

      if(is.matrix(Y)) out$parameter = torch::torch_tensor(rep(0.5, ncol(Y)), requires_grad=TRUE, device = device)
      else out$parameter = torch::torch_tensor(0.5, requires_grad=TRUE, device = device)
      out$parameter_r = as.numeric(out$parameter$cpu())
      out$invlink <- function(a) torch::torch_exp(a)
      out$link <- function(a) log(as.matrix(a$cpu()))
      out$parameter_link = function() {
        out$parameter = re_init(out$parameter, out$parameter_r)
        as.numeric((1.0/(torch::nnf_softplus(out$parameter)+0.0001))$cpu())
      }
      out$simulate = function(pred) {
        theta_tmp = out$parameter_link()
        probs = 1.0 - theta_tmp/(theta_tmp + pred)
        total_count = theta_tmp

        if(is.matrix(pred)) {
          sim = sapply(1:ncol(pred), function(i) {
            logits = log(probs[,i]) - log1p(-probs[,i])
            stats::rpois(length(logits), exp(-logits))
            return( stats::rpois(length(logits), stats::rgamma(length(logits),total_count[i], exp(- logits ))) )
          })
        } else {
          logits = log(probs) - log1p(-probs)
          stats::rpois(length(pred), exp(-logits))
          sim = stats::rpois(length(pred), stats::rgamma(length(pred),total_count, exp(- logits )))
        }
        return(sim)
      }

      out$loss = function(pred, true) {
        eps = 0.0001
        pred = torch::torch_exp(pred)
        if(pred$device$type != out$parameter$device$type) pred = pred$to(device = out$parameter$device)
        theta_tmp = 1.0/(torch::nnf_softplus(out$parameter)+eps)
        probs = torch::torch_clamp(1.0 - theta_tmp/(theta_tmp+pred)+eps, 0.0, 1.0-eps)
        total_count = theta_tmp
        value = true
        logits = torch::torch_log(probs) - torch::torch_log1p(-probs)
        log_unnormalized_prob <- total_count * torch::torch_log(torch::torch_sigmoid(-logits)) + value * torch::torch_log(torch::torch_sigmoid(logits))
        log_normalization <- -torch::torch_lgamma(total_count + value) + torch::torch_lgamma(1 + value) + torch::torch_lgamma(total_count)
        log_normalization <- torch::torch_where(total_count + value == 0, torch::torch_tensor(0, dtype = log_normalization$dtype, device = out$parameter$device), log_normalization)
        return( - (log_unnormalized_prob - log_normalization))
      }

    } else if(loss == "multinomial") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$link <- function(a) log(a) + log(ncol(a))
      out$loss <- function(pred, true) {
        return(multinomial_log_prob(torch::nnf_softmax(pred, dim = 2), true)$negative())
      }
    } else if(loss == "clogit") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$link <- function(a) log(a) + log(ncol(a))
      out$loss <- function(pred, true) {
        # return(binomial_log_prob(torch::nnf_softmax(pred, dim = 2), true))
        return(torch::distr_bernoulli(probs = torch::nnf_softmax(pred, dim = 2))$log_prob(true)$negative())
      }
    }else{
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




re_init = function(param, param_r) {
  pointer_check <- tryCatch(torch::as_array(param), error = function(e) e)

  if(inherits(pointer_check,"error")){
    param = torch::torch_tensor(param_r)
  }
  return(param)
}


# check if model is loaded and if current parameters are the desired ones
check_model <- function(object) {

  if(!inherits(object, c("citodnn", "citocnn"))) stop("model not of class citodnn or citocnn")
  pointer_check <- tryCatch(torch::as_array(object$net$parameters[[1]]), error = function(e) e)
  if(inherits(pointer_check,"error")){
    object$net <- build_model(object)
    object$loaded_model_epoch <- 0
    object$loss<- get_loss(object$loss$call)
    }

  if(object$loaded_model_epoch!= object$use_model_epoch){

    module_params<- names(object$weights[[object$use_model_epoch]])
    module_name<- sapply(module_params, function(x) {
      period_indices <- which(strsplit(x,"")[[1]]==".")
      last_period_index <- period_indices[length(period_indices)]
      substr(x,1,last_period_index-1)
    })
    module_type<- sapply(module_params, function(x) {
      period_indices <- which(strsplit(x,"")[[1]]==".")
      last_period_index <- period_indices[length(period_indices)]
      substring(x,last_period_index+1)
    })

    for ( i in names(object$net$modules)){
      if(i %in% module_name){
          k<- which(i == module_name)
          sapply(k, function(x) eval(parse(text=paste0("object$net$modules$`",i,"`$parameters$",module_type[k],"$set_data(object$weights[[object$use_model_epoch]]$`",module_params[k],"`)"))))

      }
    }
    object$loaded_model_epoch <-  object$use_model_epoch
  }

  if(!is.null(object$parameter)) object$loss$parameter <- lapply(object$parameter, torch::torch_tensor)

  return(object)
}

check_call_config <- function(mc, variable ,standards, dim = 1, check_var = FALSE, verbose = FALSE){
  value <- NULL
  if(variable %in% names(mc)){
    if(dim ==1){
      eval(parse(text = paste0("value  <- mc$",variable)))
    }else{
      eval(parse(text= paste0("value <- tryCatch(as.numeric(eval(mc$",variable,")), error = function(err)
              print(\"must be numeric input\")) ")))
    }

    if(!isFALSE(check_var)) checkmate::qassert(value,check_var)

  } else{
    value <- unlist(standards[which(names(standards) == variable)])
  }

  if(verbose) cat( paste0(variable,": [", paste(value, collapse = ", "),"] \n"))
  return(value)
}


check_listable_parameter <- function(parameter, check, vname = checkmate::vname(parameter)) {
  checkmate::qassert(parameter, c(check, "l+"), vname)
  if(inherits(parameter, "list")) {
    for (i in names(parameter)) {
      checkmate::qassert(parameter[[i]], check, paste0(vname, "$", i))
    }
  }
}

check_device = function(device) {
  if(device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  } else if(device == "mps") {
    if (torch::backends_mps_is_available()) {
      device <- torch::torch_device("mps")}
    else{
      warning("No mps device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }
  } else {
    if(device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }
  return(device)
}


# taken and adopted from lme4:::RHSForm
LHSForm = function (form, as.form = FALSE)
{
  rhsf <- form[[2]]
  if (as.form)
    stats::reformulate(deparse(rhsf))
  else rhsf
}


cast_to_r_keep_dim = function(x) {
  d = dim(x)
  if(length(d) == 1) return(as.numeric(x$cpu()))
  else return(as.matrix(x$cpu()))
}


get_X_Y = function(formula, X, Y, data) {

  if(!is.null(formula)) old_formula = formula

  if(!is.null(X)) {
    if(!is.null(Y)) {
      if(!is.matrix(Y)) Y <- data.frame(Y)
      if(ncol(Y) == 1) {
        if(is.null(colnames(Y))) colnames(Y) <- "Y"
        formula <- stats::as.formula(paste0(colnames(Y), " ~ . - 1"))
      } else {
        if(is.null(colnames(Y))) colnames(Y) <- paste0("Y", 1:ncol(Y))
        formula <- stats::as.formula(paste0("cbind(", paste(colnames(Y), collapse=","), ") ~ . - 1"))
      }
      data <- cbind(data.frame(Y), data.frame(X))
    } else {
      formula <- stats::as.formula("~ . - 1")
      old_formula = formula
      data <- data.frame(X)
    }
    formula <- formula(stats::terms.formula(formula, data = data))
    old_formula = formula

    Specials = NULL
  } else if(!is.null(formula)) {
    if(!is.null(data)) {
      data <- data.frame(data)
    }
    old_formula = formula
    parsed_formula = splitForm(formula)
    formula = parsed_formula$fixedFormula
    Specials = list(terms = parsed_formula$reTrmFormulas, types = parsed_formula$reTrmClasses, args = parsed_formula$reTrmAddArgs)
    formula <- formula(stats::terms.formula(formula, data = data))
    formula <- stats::update.formula(formula, ~ . - 1)
  } else {
    stop("Either formula (and data) or X (and Y) have to be specified.")
  }

  if(!is.null(data)) {
    char_cols <- sapply(data, is.character)
    data[,char_cols] <- lapply(data[,char_cols,drop=F], as.factor)
  }
  X <- stats::model.matrix(formula, data)
  Y <- stats::model.response(stats::model.frame(formula, data))
  if(is.null(Specials$terms)) {
    out = list(X = X, Y = Y, formula = formula, data = data, Z = NULL, Z_terms = NULL)
  } else {
    terms = sapply(Specials$terms, as.character)
    Z =
      lapply(terms, function(i) {
        return(as.integer(data[,i]))
      })
    Z = do.call(cbind, Z)
    colnames(Z) = terms
    out = list(X = X, Y = Y, formula = formula, data = data, Z = Z, Z_terms = terms, Z_args = Specials$args)
  }
  out$old_formula = old_formula
  return(out)
}


