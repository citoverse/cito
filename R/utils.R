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

### decipher config list ###
get_config_optimizer<- function(config_list){

  config_optimizer<-c()

  if(length(config_list)>0){
    config_optimizer<- config_list[which(startsWith(tolower(names(config_list)),"optimizer"))]
    names(config_optimizer)<- sapply(names(config_optimizer),function(x) substring(x,11))

  }
  return(config_optimizer)
}

get_config_lr_scheduler <- function(config_list){

  config_lr_scheduler<- c()
  if(length(config_list)>0){
    config_lr_scheduler<- config_list[which(startsWith(tolower(names(config_list)),"lr_scheduler"))]
    names(config_lr_scheduler)<- sapply(names(config_lr_scheduler),function(x) substring(x,14))
  }
  return(config_lr_scheduler)
}







