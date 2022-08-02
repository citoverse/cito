#' Creation of customized learning rate scheduler objects
#'
#' Helps create custom learning rate schedulers for \code{\link{dnn}}.
#'
#' @param type String defining which type of scheduler should be used. See Details.
#' @param verbose If TRUE, additional information about scheduler will be printed to console.
#' @param ... additional arguments to be passed to scheduler. See Details.
#' @return object of class cito_lr_scheduler to give to \code{\link{dnn}}
#'
#' @details
#'
#' different learning rate scheduler need different variables, these functions will tell you which variables can be set:
#' - lambda: \code{\link[torch]{lr_lambda}}
#' - multiplicative: \code{\link[torch]{lr_multiplicative}}
#' - reduce_on_plateau: \code{\link[torch]{lr_reduce_on_plateau}}
#' - one_cycle: \code{\link[torch]{lr_one_cycle}}
#' - step: \code{\link[torch]{lr_step}}
#'
#'
#' @example /inst/examples/config_lr_scheduler-example.R
#'
#' @export

config_lr_scheduler <- function(type = c("lambda", "multiplicative", "reduce_on_plateau", "one_cycle", "step"),
                                verbose = FALSE, ...){

  checkmate::qassert(verbose,"B1")
  type <- match.arg(tolower(type), choices =  c("lambda", "multiplicative", "reduce_on_plateau", "one_cycle", "step"))
  out <- list()
  out$lr_scheduler <- type
  class(out) <- "cito_lr_scheduler"
  mc <- match.call(expand.dots = TRUE)
  if (verbose) cat(paste0("Learning rate Scheduler ",out$lr_scheduler, "\n"))
  if (out$lr_scheduler == "lambda"){
    if("lr_lambda" %in% names(mc)){
      out$lr_lambda <- mc$lr_lambda
      if (verbose) cat(paste0("lr_lambda: [", out$lr_lambda, "]\n"))
    } else{
      warning("You need to supply lr_lambda to this function")
    }
  out$last_epoch <- check_call_config(mc = mc, "last_epoch", standards = formals(torch::lr_lambda),
                                   check_var = "R1", verbose = verbose)
    out$verbose <- check_call_config(mc = mc, "verbose", standards =formals(torch::lr_lambda),
                                 check_var = "B1", verbose = verbose)

  }else if (out$lr_scheduler == "multiplicative"){
    out$last_epoch <- check_call_config(mc = mc, "last_epoch", standards = formals(torch::lr_multiplicative),
                                        check_var = "R1", verbose = verbose)
    out$verbose <- check_call_config(mc = mc, "verbose", standards = formals(torch::lr_multiplicative),
                                        check_var = "B1", verbose = verbose)

  }else if (out$lr_scheduler == "reduce_on_plateau"){
    out$mode <- check_call_config(mc = mc, "mode", standards = formals(torch::lr_reduce_on_plateau),
                                  check_var = F, verbose = verbose)
    out$factor <- check_call_config(mc = mc, "factor", standards = formals(torch::lr_reduce_on_plateau),
                                  check_var = "R1", verbose = verbose)
    out$patience <- check_call_config(mc = mc, "patience", standards = formals(torch::lr_reduce_on_plateau),
                                                  check_var = "R1", verbose = verbose)
    out$threshold <-  check_call_config(mc = mc, "threshold", standards = formals(torch::lr_reduce_on_plateau),
                                                   check_var = "R1", verbose = verbose)
    out$threshold_mode <- check_call_config(mc = mc, "threshold_mode", standards = formals(torch::lr_reduce_on_plateau),
                                                   check_var = F, verbose = verbose)
    out$cooldown <- check_call_config(mc = mc, "cooldown", standards = formals(torch::lr_reduce_on_plateau),
                                                   check_var = "R1", verbose = verbose)
    out$min_lr <- check_call_config(mc = mc, "min_lr", standards = formals(torch::lr_reduce_on_plateau),
                                                   check_var = F, verbose = verbose)
    out$eps <- check_call_config(mc = mc, "eps", standards = formals(torch::lr_reduce_on_plateau),
                                                   check_var = "R1", verbose = verbose)
    out$threshold <- check_call_config(mc = mc, "verbose", standards = formals(torch::lr_reduce_on_plateau),
                                                   check_var = "B1", verbose = verbose)




  }else if (out$lr_scheduler == "one_cycle"){
    if("max_lr" %in% names(mc)){
      out$max_lr <- mc$max_lr
      if (verbose) cat(paste0("max_lr: [", out$max_lr, "]\n"))
    } else{
      warning("You need to supply max_lr to this function")
    }
    out$total_steps <- check_call_config(mc = mc, "total_steps", standards = formals(torch::lr_one_cycle),
                                        check_var = F, verbose = verbose)
    out$epochs <- check_call_config(mc = mc, "epochs", standards = formals(torch::lr_one_cycle),
                                        check_var = F, verbose = verbose)
    out$steps_per_epoch <- check_call_config(mc = mc, "steps_per_epoch", standards = formals(torch::lr_one_cycle),
                                        check_var = F, verbose = verbose)
    out$pct_start <- check_call_config(mc = mc, "pct_start", standards = formals(torch::lr_one_cycle),
                                        check_var = "R1", verbose = verbose)
    out$anneal_strategy <- check_call_config(mc = mc, "anneal_strategy", standards = formals(torch::lr_one_cycle),
                                        check_var = "S+", verbose = verbose)
    out$cycle_momentum <- check_call_config(mc = mc, "cycle_momentum", standards = formals(torch::lr_one_cycle),
                                        check_var = "B1", verbose = verbose)
    out$base_momentum <- check_call_config(mc = mc, "base_momentum", standards = formals(torch::lr_one_cycle),
                                        check_var = F, verbose = verbose)
    out$max_momentum <- check_call_config(mc = mc, "max_momentum", standards = formals(torch::lr_one_cycle),
                                        check_var = F, verbose = verbose)
    out$div_factor <- check_call_config(mc = mc, "div_factor", standards = formals(torch::lr_one_cycle),
                                        check_var = "R1", verbose = verbose)

    out$last_epoch <- check_call_config(mc = mc, "last_epoch", standards = formals(torch::lr_one_cycle),
                                        check_var = "R1", verbose = verbose)

    out$final_div_factor <- check_call_config(mc = mc, "final_div_factor", standards = formals(torch::lr_one_cycle),
                                        check_var = "R1", verbose = verbose)
    out$verbose <- check_call_config(mc = mc, "verbose", standards = formals(torch::lr_one_cycle),
                                        check_var = "B1", verbose = verbose)


  }else if (out$lr_scheduler == "step"){
    if("step_size" %in% names(mc)){
      out$step_size <- mc$step_size
      if (verbose) cat(paste0("step_size: [", out$step_size, "]\n"))
    } else{
      warning("You need to supply step_size to this function")
    }
    out$gamma <- check_call_config(mc = mc, "gamma", standards = formals(torch::lr_step),
                                     check_var = "R1", verbose = verbose)
    out$last_epoch <- check_call_config(mc = mc, "last_epoch", standards = formals(torch::lr_step),
                                     check_var = "R1", verbose = verbose)



  }

  for(var in names(mc)[2:length(names(mc))]){
    if(!(var %in%c( "type", "verbose"))){
      if(!(var%in% names(out)) & var != "type"){
        warning(paste0(var, " could not be assigned to ", out$lr_scheduler," scheduler \n"))
      }
    }
  }

  return(out)
}


get_lr_scheduler<- function(lr_scheduler, optimizer){

if(!inherits(lr_scheduler, "cito_lr_scheduler")){
    stop( "Please provide a learning rate scheduler config object created by config_lr_scheduler()")
}

  param_lr_scheduler <- list(optimizer = optimizer)
  for (i in seq_len(length(names(lr_scheduler)))){
    if(names(lr_scheduler)[i]!= "lr_scheduler") {
      param_lr_scheduler <- append(param_lr_scheduler,unlist(unname(lr_scheduler[i])))
    }
  }
  scheduler <- switch(tolower(lr_scheduler$lr_scheduler),
                      "step" = do.call(torch::lr_step,param_lr_scheduler),
                      "one_cycle" = do.call(torch::lr_one_cycle,param_lr_scheduler),
                      "multiplicative" = do.call(torch::lr_multiplicative,param_lr_scheduler),
                      "reduce_on_plateau" = do.call(torch::lr_reduce_on_plateau,param_lr_scheduler),
                      "lambda" = do.call(torch::lr_lambda,param_lr_scheduler),
                      stop(paste0("lr_scheduler = ",lr_scheduler," is not supported")))

  return(scheduler)
}
