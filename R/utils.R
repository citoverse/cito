# check if model is loaded and if current parameters are the desired ones
check_model <- function(object) {

  if(!inherits(object, c("citodnn"))) stop("model not of class citodnn")

  pointer_check <- tryCatch(torch::as_array(object$net$parameters[[1]]), error = function(e) e)
  if(inherits(pointer_check,"error")){
    object$net<- build_dnn(input =object$model_properties$input,
                    output = object$model_properties$output,
                    hidden = object$model_properties$hidden,
                    activation = object$model_properties$activation,
                    bias = object$model_properties$bias,
                    dropout = object$model_properties$dropout)
    object$loaded_model_epoch<- 0
    object$loss<- get_loss(object$loss$call)
    }

  if(object$loaded_model_epoch!= object$use_model_epoch){

    module_params<- names(object$weights[[object$use_model_epoch]])
    module_number<- sapply(module_params, function(x) substr(x,1,which(strsplit(x,"")[[1]]==".")-1))
    module_type<-sapply(module_params, function(x) substring(x,which(strsplit(x,"")[[1]]==".")+1))

    for ( i in names(object$net$modules)){
      if(i %in% module_number){
          k<- which(i == module_number)
          sapply(k, function(x) eval(parse(text=paste0("object$net$modules$`",i,"`$parameters$",module_type[k],"$set_data(object$weights[[object$use_model_epoch]]$`",module_params[k],"`)"))))

      }
    }
    object$loaded_model_epoch <-  object$use_model_epoch
  }
  return(object)
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
      out$parameter <- torch::torch_tensor(0.1, requires_grad = TRUE)
      out$invlink <- function(a) a
      out$loss <- function(pred, true) {
        return(torch::distr_normal(pred, torch::torch_clamp(out$parameter, 0.0001, 20))$log_prob(true)$negative())
      }
    } else if(loss$family == "binomial") {
      if(loss$link == "logit") {
        out$invlink <- function(a) torch::torch_sigmoid(a)
      } else if(loss$link == "probit")  {
        out$invlink <- function(a) torch::torch_sigmoid(a*1.7012)
      } else {
        out$invlink <- function(a) a
      }
      out$loss <- function(pred, true) {
        return(torch::distr_bernoulli( out$invlink(pred))$log_prob(true)$negative())
      }
    } else if(loss$family == "poisson") {
      if(loss$link == "log") {
        out$invlink <- function(a) torch::torch_exp(a)
      } else {
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
  } else {
    if(loss == "mae"){
      out$invlink <- function(a) a
      out$loss <- function(pred, true) return(torch::nnf_l1_loss(input = pred, target = true))
    }else if(loss == "mse"){
      out$invlink <- function(a) a
      out$loss <- function(pred,true) return(torch::nnf_mse_loss(input= pred, target = true))
    }else if(loss == "softmax" | loss == "cross-entropy") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$loss <- function(pred, true) {
        return(torch::nnf_cross_entropy(pred, true$squeeze(), reduction = "none"))
      }
    }
    else{
      cat( "unidentified loss \n")
    }

  }
  out$call <- loss

  return(out)
}


check_call_config <- function(mc, variable ,standards, print = T, dim = 1, check_var = F){
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

  if(print) cat( paste0(variable,": [", paste(value, collapse = ", "),"] \n"))
  return(value)
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

get_importance<- function(model, n_permute){

  model<- check_model(model)

  if(!any(model$loss$call  == c("softmax","mse", "mae"))){ return(0)}
  loss<- model$loss$loss

  org_err <- as.numeric(loss(pred = torch::torch_tensor(stats::predict(model,model$data$data, type = "link")),
                  true = torch::torch_tensor(model$data$Y))$mean())


  importance <- data.frame(variable = get_var_names(model$training_properties$formula, model$data$data[1,]),
                          importance= c(0))

  for(i in seq_len(nrow(importance))){

    perm_preds <- c()
    true <- c()

    if(n_permute < ((nrow(model$data$data)**2)-1)){
      for(k in seq_len(n_permute)){

        perm_data <- model$data$data
        perm_data[,importance$variable[i]] <- perm_data[sample.int(n = nrow(perm_data),replace = F),importance$variable[i]]

        perm_preds <- rbind(perm_preds,stats::predict(model,perm_data, type = "response"))
        true <- append(true, model$data$Y)

      }
    }else{
      for(j in seq_len(nrow(model$data$data))){
        perm_data <- model$data$data[j,]
        for(k in seq_len(nrow(model$data$data))[-j]){
          perm_data[i] <- model$data$data[k,i]
          perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link"))
          true <- append(true, model$data$Y[j])

        }

      }
    }

    new_err <- as.numeric(loss(pred = torch::torch_tensor(perm_preds),
                    true = torch::torch_tensor(true))$mean())
    importance$importance[i] <- new_err/org_err

  }

  return(importance)
}
