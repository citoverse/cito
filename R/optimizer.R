#' Creation of customized optimizer objects
#'
#' Helps you create custom optimizer for \code{\link{dnn}}. It is recommended to set learning rate in \code{\link{dnn}}.
#'
#' @param type character string defining which optimizer should be used. See Details.
#' @param ... additional arguments to be passed to optimizer. See Details.
#' @return object of class cito.optim to give to \code{\link{dnn}}
#'
#' @details
#'
#' different optimizer need different variables, this function will tell you how the variables are set.
#' For more information see the corresponding functions:
#' - adam: \code{\link[torch]{optim_adam}}
#' - adadelta: \code{\link[torch]{optim_adadelta}}
#' - adagrad: \code{\link[torch]{optim_adagrad}}
#' - rmsprop: \code{\link[torch]{optim_rmsprop}}
#' - rprop: \code{\link[torch]{optim_rprop}}
#' - sgd: \code{\link[torch]{optim_sgd}}
#'
#'
#' @example /inst/examples/config_optimizer.R
#'
#' @export

config_optimizer<- function(type = c("adam", "adadelta", "adagrad", "rmsprop", "rprop", "sgd"), ... ){

  type <- match.arg(tolower(type), choices =  c("adam", "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "lbfgs"))
  out <- list()
  out$optim <- type
  class(out) <- "cito_config_optim"
  mc <- match.call(expand.dots = TRUE)
  if("lr" %in% names(mc)){
    print("learning rate set here will overwrite lr you give to dnn()")
    checkmate::qassert(mc$lr,"R1[0,)")
    out$lr <- mc$lr
  }
  if (out$optim == "adam"){

    print("adam optimizer with following values")

    out$betas <- check_call_config(mc = mc, "betas", standards =formals(torch::optim_adam),
                                   check_var = "R2", dim = 2)
    out$eps <- check_call_config(mc = mc, "eps", standards =formals(torch::optim_adam),
                                 check_var = "R1")
    out$weight_decay <- check_call_config(mc = mc, "weight_decay", standards =formals(torch::optim_adam),
                                          check_var = "R1")
    out$amsgrad <- check_call_config(mc = mc, "amsgrad", standards =formals(torch::optim_adam),
                                     check_var = "B1")

  }else if(out$optim == "adadelta"){

    print("set adadelta optimizer with following values")
    out$rho <- check_call_config(mc = mc, "rho", standards =formals(torch::optim_adadelta),
                                 check_var = "R1")
    out$eps <- check_call_config(mc = mc, "eps", standards =formals(torch::optim_adadelta),
                                 check_var = "R1")
    out$weight_decay <- check_call_config(mc = mc, "weight_decay", standards =formals(torch::optim_adadelta),
                                          check_var = "R1")




  }else if(out$optim == "adagrad"){

    print("set adagrad optimizer with following values")
    out$lr_decay <- check_call_config(mc = mc, "lr_decay", standards =formals(torch::optim_adagrad),
                                      check_var = "R1")
    out$weight_decay <- check_call_config(mc = mc, "weight_decay", standards =formals(torch::optim_adagrad),
                                          check_var = "R1")
    out$initial_accumulator_value <- check_call_config(mc = mc, "initial_accumulator_value", standards = formals(torch::optim_adagrad),
                                                       check_var = "R1")
    out$eps <- check_call_config(mc = mc, "eps", standards =formals(torch::optim_adagrad),
                                 check_var = "R1")



  }else if(out$optim == "rmsprop"){

    print("set rmsprop optimizer with following values")
    out$alpha <- check_call_config(mc = mc, "alpha", standards =formals(torch::optim_rmsprop),
                                   check_var = "R1")
    out$eps <- check_call_config(mc = mc, "eps", standards =formals(torch::optim_rmsprop),
                                 check_var = "R1")
    out$weight_decay <- check_call_config(mc = mc, "weight_decay", standards =formals(torch::optim_rmsprop),
                                          check_var = "R1")
    out$momentum <- check_call_config(mc = mc, "momentum", standards =formals(torch::optim_rmsprop),
                                      check_var = "R1")
    out$centered <- check_call_config(mc = mc, "centered", standards =formals(torch::optim_rmsprop),
                                      check_var = "B1")


  }else if(out$optim == "rprop"){

    print("set rprop optimizer with following values")
    out$etas <- check_call_config(mc = mc, "etas", standards =formals(torch::optim_rprop),
                                  check_var = "R2", dim = 2)
    out$step_sizes <- check_call_config(mc = mc, "step_sizes", standards =formals(torch::optim_rprop),
                                        check_var = "R2", dim = 2)


  }else if(out$optim == "sgd"){

    print("set sgd optimizer with following values")
    out$momentum <- check_call_config(mc = mc, "momentum", standards =formals(torch::optim_sgd),
                                      check_var = "R1")
    out$dampening <- check_call_config(mc = mc, "dampening", standards =formals(torch::optim_sgd),
                                       check_var = "R1")
    out$weight_decay <- check_call_config(mc = mc, "weight_decay", standards =formals(torch::optim_sgd),
                                          check_var = "R1")
    out$nesterov <- check_call_config(mc = mc, "nesterov", standards =formals(torch::optim_sgd),
                                      check_var = "B1")
  }

  for(var in names(mc)[2:length(names(mc))]){
    if(!(var%in% names(out)) & var != "type"){
      print(paste0(var, " could not be assigned for ", out$optim," optimizer"))
    }
  }

  return(out)
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



get_optimizer <- function(optimizer, lr, parameters){

  if(!inherits(optimizer, "cito_config_optim")){
    optimizer <- match.arg(tolower(optimizer), choices = c("adam","adadelta", "adagrad", "rmsprop", "rprop", "sgd", "lbfgs"))

    optim <- switch(optimizer,
                    "adam"= torch::optim_adam(params= parameters, lr=lr),
                    "adadelta" = torch::optim_adadelta(params= parameters,lr=lr),
                    "adagrad" =  torch::optim_adagrad(params= parameters, lr=lr),
                    "rmsprop"  = torch::optim_rmsprop(params= parameters, lr=lr),
                    "rprop" = torch::optim_rprop(params= parameters, lr=lr),
                    "sgd" = torch::optim_sgd(params= parameters, lr=lr),
                    stop(paste0("optimizer = ",optimizer," is not supported, choose between adam, adadelta, adagrad, rmsprop, rprop, sgd or lbfgs")))

  }else{

    param_optimizer <- list(params = parameters)
    for (i in seq_len(length(names(optimizer)))){
      if(names(optimizer)[i]!= "optim") {
        param_optimizer <- append(param_optimizer,unlist(unname(optimizer[i])))
      }
    }
    if("lr"%in% names(param_optimizer)){
      warning("two lr available, lr of config_optimizer object gets set as lr")
    }else{
      param_optimizer$lr <- lr
    }



    optim <- switch(optimizer$optim,
                    "adam"= do.call(torch::optim_adam,param_optimizer),
                    "adadelta" = do.call(torch::optim_adadelta,param_optimizer),
                    "adagrad" =  do.call(torch::optim_adagrad,param_optimizer),
                    "rmsprop"  = do.call(torch::optim_rmsprop,param_optimizer),
                    "rprop" = do.call(torch::optim_rprop,param_optimizer),
                    "sgd" = do.call(torch::optim_sgd,param_optimizer))
  }

  return(optim)
}
