# check if model is loaded and if current parameters are the desired ones
check_model <- function(object) {

  if(!inherits(object, c("citodnn"))) stop("model not of class citodnn")

  pointer_check <- tryCatch(torch::as_array(object$net$parameters[[1]]), error = function(e) e)
  if(inherits(pointer_check,"error")){
    object$net<- build_model(input =object$model_properties$input,
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
  checkmate::qassert(parameter, c(check, "L+"), vname)
  if(inherits(parameter, "list")) {
    for (i in names(parameter)) {
      checkmate::qassert(parameter[[i]], check, paste0(vname, "$", i))
    }
  }
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

get_importance<- function(model, n_permute= NULL, data = NULL){

  if(is.null(n_permute)) n_permute <- ceiling(sqrt(nrow(model$data$data))*3)
  model<- check_model(model)

  if(inherits(model$loss$call, "character")) {
    if(!any(model$loss$call  == c("softmax","mse", "mae"))){ return(0)}
  } else {
    if(!any(model$loss$call$family  == c("binomial")  )){ return(0)}
  }
  loss<- model$loss$loss

  true = model$data$Y

  if(inherits(model$loss$call, "character")) {
    true = torch::torch_tensor(model$data$Y)
  } else {
    if(model$loss$call$family  == c("binomial") ){
      mode(true) = "numeric"
      true = torch::torch_tensor(true)
      }
  }
  org_err <- as.numeric(loss( pred = torch::torch_tensor(stats::predict(model,model$data$data, type = "link")) ,true = true)$mean())


  importance <- data.frame(variable = get_var_names(model$training_properties$formula, model$data$data[1,]),
                          importance = c(0))

  for(i in seq_len(nrow(importance))){

    new_err <-c()
    if(n_permute < ((nrow(model$data$data)**2)-1)){
      for(k in seq_len(n_permute)){

        perm_preds <- c()

        perm_data <- model$data$data
        perm_data[, importance$variable[i]] <- perm_data[sample.int(n = nrow(perm_data),replace = FALSE),importance$variable[i]]

        perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link"))


        new_err <- append(new_err, as.numeric(loss(pred = torch::torch_tensor(perm_preds),
                                   true = true)$mean() ))


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

    importance$importance[i] <- mean(new_err)/org_err

  }

  return(importance)
}
