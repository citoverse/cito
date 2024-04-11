train_model <- function(model,  epochs, device, train_dl, valid_dl=NULL, verbose = TRUE, plot_new = FALSE, init_optimizer=TRUE){
  model$net$to(device = device)
  model$net$train()
  model$successfull = 1

  ### Optimizer ###
  if(init_optimizer) {
    optimizer <- get_optimizer(optimizer = model$training_properties$optimizer,
                               parameters = c(model$net$parameters, unlist(model$loss$parameter)),
                               lr = model$training_properties$lr)
  } else {
    optimizer = model$optimizer # TODO check that optimizer exists
  }
  ### LR Scheduler ###
  scheduler <- NULL
  if(!is.null(model$training_properties$lr_scheduler)){
    scheduler <- get_lr_scheduler(lr_scheduler = model$training_properties$lr_scheduler, optimizer = optimizer)
  }


  if(is.null(model$losses)){
    model$losses <- data.frame(epoch=c(1:epochs),train_l=NA,valid_l= NA)
  }else{
    model$losses <- rbind(model$losses,
                          data.frame(epoch=c((max(model$losses$epoch)+1):(max(model$losses$epoch)+epochs)),train_l=NA,valid_l= NA))
  }

  loss.fkt <- model$loss$loss
  if(!is.null(model$loss$parameter)) list2env(model$loss$parameter,envir = environment(fun= loss.fkt))

  regularize <- !(model$training_properties$lambda==0)

  best_train_loss = Inf
  best_val_loss = Inf
  counter = 0

  for (epoch in min(which(is.na(model$losses$train_l))):(epochs+ min(which(is.na(model$losses$train_l))) - 1)) {
    train_l <- c()
    model$training_properties$epoch <- epoch

    ### Batch evaluation ###
    coro::loop(for (b in train_dl) {
      optimizer$zero_grad()

      b <- lapply(b, function(x) x$to(device=device, non_blocking= TRUE))
      if(inherits(model, "citommn")) {
        output <- model$net(b[-length(b)])
      } else {
        if(is.null(model$training_properties$embeddings)) output <- model$net(b[[1]])
        else output <- model$net(b[[1]], b[[2]])
      }
      loss <- loss.fkt(output, b[[length(b)]])$mean()

      if(regularize){
        regularization_loss <- regularize_weights(parameters = model$net$parameters,
                                   alpha = model$training_properties$alpha,
                                   lambda = model$training_properties$lambda)
        total_loss = torch::torch_add(loss, regularization_loss)

      } else {
        total_loss = loss
      }

      if(!is.null(model$training_properties$embeddings)) {
        for(i in 1:length(model$training_properties$embeddings$dims)) {
          if(model$training_properties$embeddings$args[[i]]$lambda > 0) {
            total_loss = torch::torch_add(total_loss,  regularize_weights(model$net[[paste0("e_", i)]]$parameters,
                                                                          lambda = model$training_properties$embeddings$args[[i]]$lambda,
                                                                          alpha = model$training_properties$embeddings$args[[i]]$alpha))
          }

        }
      }

      total_loss$backward()

      optimizer$step()

      train_l <- c(train_l, loss$item())
    })

    if(is.na(loss$item())) {
      if(verbose) cat("Loss is NA. Bad training, please hyperparameters. See vignette('B-Training_neural_networks') for help.\n")
      model$successfull = 0
      break
    }

    model$losses$train_l[epoch] <- mean(train_l)

    if(epoch >= model$burnin) {
      if(model$losses$train_l[epoch] > model$base_loss) {
        if(verbose) cat("Cancel training because loss is still above baseline, please hyperparameters. See vignette('B-Training_neural_networks') for help.\n")
        model$successfull = 0
        break
      }
    }


    if(model$training_properties$validation != 0 & !is.null(valid_dl)){
      model$net$train(FALSE)

      valid_l <- c()

      coro::loop(for (b in valid_dl) {
        b <- lapply(b, function(x) x$to(device=device, non_blocking= TRUE))
        if(inherits(model, "citommn")) {
          output <- model$net(b[-length(b)])
        } else {
          if(is.null(model$training_properties$embeddings)) output <- model$net(b[[1]])
          else output <- model$net(b[[1]], b[[2]])
        }
        loss <- loss.fkt(output, b[[length(b)]])$mean()
        valid_l <- c(valid_l, loss$item())
      })
      model$losses$valid_l[epoch] <- mean(valid_l)

      model$net$train(TRUE)
    }

    ### learning rate scheduler ###
    if(!is.null(scheduler)){
      if(model$training_properties$lr_scheduler$lr_scheduler == "reduce_on_plateau"){
        if(model$training_properties$validation != 0 & !is.null(valid_dl)){
          scheduler$step(model$losses$valid_l[epoch])
        }else{
          scheduler$step(model$losses$train_l[epoch])
        }
      }else{
        scheduler$step()
      }
    }

    if(model$training_properties$validation != 0 & !is.null(valid_dl)){
      if(verbose) cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f, lr: %3.5f\n",
                              epoch, model$losses$train_l[epoch], model$losses$valid_l[epoch],optimizer$param_groups[[1]]$lr))
    }else{
      if (verbose) cat(sprintf("Loss at epoch %d: %3f, lr: %3.5f\n", epoch, model$losses$train_l[epoch],optimizer$param_groups[[1]]$lr))
    }

    ### create plot ###
    main <- switch (class(model),
      citodnn = "Training of DNN",
      citocnn = "Training of CNN",
      citommn = "Training of MMN"
    )
    if(model$training_properties$plot) visualize.training(model$losses,epoch, main = main, new = plot_new, baseline = model$base_loss)
    plot_new <- FALSE

    # Save best weights
    if(model$training_properties$validation != 0 & !is.null(valid_dl)) {
      if(model$losses$valid_l[epoch] < best_val_loss) {
        best_val_loss = model$losses$valid_l[epoch]
        model$weights[[1]] =  lapply(model$net$parameters,function(x) torch::as_array(x$to(device="cpu")))
        counter = 0
      }
    } else {
      if(model$losses$train_l[epoch] < best_train_loss) {
        best_train_loss = model$losses$train_l[epoch]
        model$weights[[1]] =  lapply(model$net$parameters,function(x) torch::as_array(x$to(device="cpu")))
        counter = 0
      }
    }

    ### early stopping ###
    if(is.numeric(model$training_properties$early_stopping)) {
      if(counter >= model$training_properties$early_stopping) {
        break
      }
      counter = counter + 1
    }

  }

  model$net$to(device = "cpu")

  model$weights[[2]] =  lapply(model$net$parameters,function(x) torch::as_array(x$to(device="cpu")))

  if(!is.null(model$loss$parameter)) model$parameter <- lapply(model$loss$parameter, cast_to_r_keep_dim)
  model$use_model_epoch <- 1
  model$loaded_model_epoch <- 1

  if(!is.null(model$loss$parameter)) {
      model$loss$parameter_r = unlist(lapply(model$loss$parameter, function(p) as.numeric(p$cpu())))
  }

  model$optimizer = optimizer

  model$net$eval()

  return(model)
}

regularize_weights <- function (parameters, alpha, lambda){

  weight_layers <- names(which(sapply(parameters, function(x) length(dim(x))) > 1))

  regularization = torch::torch_zeros(1L, dtype = parameters[[1]]$dtype, device = parameters[[1]]$device)

  for (i in 1:length(weight_layers)) {
      l1 <- torch::torch_sum(torch::torch_abs(parameters[[weight_layers[i]]]))
      l1 <- l1$mul(1-alpha)
      l2 <- torch::torch_norm(parameters[[weight_layers[i]]],p=2L)
      l2 <- l2$mul(alpha)

      regularization_tmp <- torch::torch_add(l1,l2)
      regularization_tmp <- regularization_tmp$mul(lambda)
      regularization = regularization$add(regularization_tmp)
  }

  return(regularization)
}



visualize.training <- function(losses,epoch, main, new = FALSE, baseline = NULL){
  if (epoch==1|new){

    graphics::plot(c(),c(),xlim=c(1,nrow(losses)),ylim=c(0,max(losses$train_l,losses$valid_l,baseline,na.rm=T)),
                   main= main,
                   xlab= "epoch",
                   ylab= "loss",
                   las = 1)
    if(!is.na(losses$valid_l[1])) {
      graphics::legend("topright",legend= c("training","validation", "baseline"),
                       col= c("#5B84B1FF","#FC766AFF", "#00c49aAA"),lty=1, cex=0.8,
                       title="Line types", bg='white', bty = "n")
    } else {
      graphics::legend("topright",legend= c("training", "baseline"),
                       col= c("#5B84B1FF","#00c49aAA"),lty=1, cex=0.8,
                       title="Line types", bg='white', bty = "n")

    }
    graphics::points(x=c(1),y=c(losses$train_l[1]),pch=19, col="#5B84B1FF", lty=1)
    graphics::points(x=c(1),y=c(losses$valid_l[1]),pch=18, col="#FC766AFF", lty=1)
    graphics::abline(h = baseline, col = "#00c49aAA", lwd = 1.8)
    if(epoch > 1){
      for ( i in c(2:epoch)){
        graphics::lines(c(i-1,i), c(losses$train_l[i-1],losses$train_l[i]), pch=19, col="#5B84B1FF", type="b", lty=1)
        graphics::lines(c(i-1,i), c(losses$valid_l[i-1],losses$valid_l[i]), pch=18, col="#FC766AFF", type="b", lty=1)
      }
    }
  } else{

    graphics::lines(c(epoch-1,epoch), c(losses$train_l[epoch-1],losses$train_l[epoch]), pch=19, col="#5B84B1FF", type="b", lty=1)
    graphics::lines(c(epoch-1,epoch), c(losses$valid_l[epoch-1],losses$valid_l[epoch]), pch=18, col="#FC766AFF", type="b", lty=1)
  }
}

#' Visualize training of Neural Network
#'
#' @description
#'
#' After training a model with cito, this function helps to analyze the training process and decide on best performing model.
#' Creates a 'plotly' figure which allows to zoom in and out on training graph
#'
#' @details
#' The baseline loss is the most important reference. If the model was not able to achieve a better (lower) loss than the baseline (which is the loss for a intercept only model), the model probably did not converge. Possible reasons include an improper learning rate, too few epochs, or too much regularization. See the `?dnn` help or the `vignette("B-Training_neural_networks")`.
#'
#'
#' @param object a model created by \code{\link{dnn}} or \code{\link{cnn}}
#' @return a 'plotly' figure
#' @example /inst/examples/analyze_training-example.R
#' @export

analyze_training<- function(object){

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop(
      "Package \"plotly\" must be installed to use this function.",
      call. = FALSE
    )
  }

  if(!inherits(object, c("citodnn", "citodnnBootstrap", "citocnn"))) stop("Function requires an object of class citodnn, citodnnBootstrap or citocnn")


  if(inherits(object, c("citodnn", "citocnn"))) {
    fig <- plotly::plot_ly(object$losses, type = 'scatter', mode = 'lines+markers',
                           width = 900)

    fig<- plotly::add_trace(fig,x = ~epoch, y = ~train_l,text = "Training Loss",
                            line = list(color = "#5B84B1FF"),
                            marker =list(color = "#5B84B1FF"), name = "Training loss" )
    if(object$call$validation>0 && !is.null(object$call$validation))  {
      fig<- plotly::add_trace(fig,x = ~epoch, y = ~valid_l, text ="Validation loss",
                              line = list(color = "#FC766AFF"),
                              marker =list(color = "#FC766AFF"), name = "Validation loss")
    }

    # "#5B84B1FF","#FC766AFF" training, validation
    fig <- plotly::layout(fig, shapes = list(type = "line",
                                             x0 = 0,
                                             x1 = 1,
                                             showlegend = TRUE,
                                             name = "Baseline loss",
                                             xref = "paper",
                                             y0 = object$base_loss,
                                             y1 = object$base_loss,
                                             line = list(color = "#00c49aAA")
    ))
    title <- ifelse(inherits(object, "citocnn"), 'CNN Training', 'DNN Training')
    fig<- plotly::layout(fig,
                         title=title,
                         xaxis = list(zeroline = FALSE),
                         yaxis = list(zeroline = FALSE,
                                      fixedrange = FALSE,
                                      title = "Trainings loss"))

    return(fig)
  }

  if(inherits(object, "citodnnBootstrap")) {

    train_l = sapply(object$models, function(i) i$losses$train_l)
    base_loss = mean(sapply(object$models, function(i) i$base_loss))
    fig <- plotly::plot_ly(type = 'scatter', mode = 'lines+markers',
                           width = 900)

    fig <- plotly::layout(fig, shapes = list(type = "line",
                                             x0 = 0,
                                             x1 = 1,
                                             showlegend = TRUE,
                                             name = "Baseline loss",
                                             xref = "paper",
                                             y0 = base_loss,
                                             y1 = base_loss,
                                             line = list(color = "#00c49aAA")
    ))

    for(i in 1:length(object$models)) {
      fig = plotly::add_trace(fig, x = object$models[[1]]$losses$epoch, name = i,
                              y = train_l[,i])
    }
    fig<- plotly::layout(fig,
                         title='DNN Training',
                         xaxis = list(zeroline = FALSE),
                         yaxis = list(zeroline = FALSE,
                                      fixedrange = FALSE,
                                      title = "Trainings loss"))
    return(fig)
  }

}




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
      if(!(var%in% names(out))){
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
