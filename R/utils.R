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
    object$family<- get_family(object$called_with_family)
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




check_call_config <- function(mc, variable ,standards, print = T, dim = 1, check_var = F){
  value <- NULL
  if(variable %in% names(mc)){
    if(dim ==1){
      eval(parse(text = paste0("value  <- mc$",variable)))
    }else{
      eval(parse(text= paste0("value <- tryCatch(as.numeric(eval(mc$",variable,")), error = function(err)
              print(\"betas must be numeric\")) ")))
    }

    if(!isFALSE(check_var)) checkmate::qassert(value,check_var)

  } else{
    value <- unlist(standards[which(names(standards) == variable)])
  }

  if(print) print( paste0(variable,": [", paste(value, collapse = ", "),"]"))
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

  loss<- function(pred, true){
    return(mean(abs(pred-true)**2))
  }

  org_err <- loss(pred = stats::predict(model,model$data$data),
                  true = model$data$Y)

  importance<- data.frame(variable = get_var_names(model$training_properties$formula, model$data$data[1,]),
                          importance= c(0))

  for(i in seq_len(nrow(importance))){

    perm_preds <- c()
    true <- c()

    if(n_permute < ((nrow(model$data$data)**2)-1)){
      for(k in seq_len(n_permute)){

        perm_data <- model$data$data
        perm_data[,importance$variable[i]] <- perm_data[sample.int(n = nrow(perm_data),replace = F),importance$variable[i]]

        perm_preds <- append(perm_preds,stats::predict(model,perm_data)[,1])
        true <- append(true, model$data$Y)
      }
    }else{
      for(j in seq_len(nrow(model$data$data))){
        perm_data <- model$data$data[j,]
        for(k in seq_len(nrow(model$data$data))[-j]){
          perm_data[i] <- model$data$data[k,i]
          perm_preds <- append(perm_preds, stats::predict(model, perm_data))
          true <- append(true, model$data$Y[j])

        }
      }
    }

    new_err <- loss(pred = perm_preds,
                    true = true)
    importance$importance[i] <- new_err/org_err

  }

  return(importance)
}

