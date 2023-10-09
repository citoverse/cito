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
  if(length(d) == 1) return(as.numeric(x))
  else return(as.matrix(x))
}
