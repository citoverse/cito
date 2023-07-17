train_model <- function(model,  epochs, device, train_dl, valid_dl=NULL, verbose = TRUE, plot_new = FALSE){
  model$net$to(device = device)

  ### Optimizer ###
  optimizer <- get_optimizer(optimizer = model$training_properties$optimizer,
                             parameters = c(model$net$parameters, unlist(model$loss$parameter)),
                             lr = model$training_properties$lr)

  ### LR Scheduler ###
  scheduler <- NULL
  if(!is.null(model$training_properties$lr_scheduler)){
    scheduler <- get_lr_scheduler(lr_scheduler = model$training_properties$lr_scheduler, optimizer = optimizer)
  }


  if(is.null(model$losses)){
    model$losses <- data.frame(epoch=c(1:epochs),train_l=NA,valid_l= NA)
  }else{
    model$losses <- rbind(model$losses[1:model$use_model_epoch,],
                          data.frame(epoch=c((model$use_model_epoch+1):(model$use_model_epoch+epochs)),train_l=NA,valid_l= NA))
  }

  loss.fkt <- model$loss$loss
  if(!is.null(model$loss$parameter)) list2env(model$loss$parameter,envir = environment(fun= loss.fkt))

  regularize <- TRUE
  if(all(is.na(model$training_properties$alpha) | model$training_properties$lambda==0)) regularize <- FALSE

  intercept <- FALSE
  if(inherits(model, "citodnn")) intercept <- model$model_properties$bias[1]

  best_train_loss = Inf
  best_val_loss = Inf
  counter = 0

  for (epoch in min(which(is.na(model$losses$train_l))):(epochs+ min(which(is.na(model$losses$train_l))) - 1)) {
    train_l <- c()
    model$training_properties$epoch <- epoch

    ### Batch evaluation ###
    coro::loop(for (b in train_dl) {
      optimizer$zero_grad()
      output <- model$net(b[[1]]$to(device = device, non_blocking= TRUE))
      loss <- loss.fkt(output, b[[2]]$to(device = device, non_blocking= TRUE))$mean()
      if(regularize){
        loss <- regularize_weights(loss = loss, parameters = model$net$parameters,
                                   alpha = model$training_properties$alpha,
                                   lambda = model$training_properties$lambda,
                                   intercept = intercept)

      }

      loss$backward()

      optimizer$step()

      train_l <- c(train_l, loss$item())
    })
    model$losses$train_l[epoch] <- mean(train_l)


    if(model$training_properties$validation != 0 & !is.null(valid_dl)){

      valid_l <- c()

      coro::loop(for (b in valid_dl) {
        output <- model$net(b[[1]]$to(device = device, non_blocking= TRUE))
        loss <- loss.fkt(output, b[[2]]$to(device = device, non_blocking= TRUE))$mean()
        if (regularize) {
          loss <- regularize_weights(loss = loss, parameters = model$net$parameters,
                                     alpha = model$training_properties$alpha,
                                     lambda = model$training_properties$lambda,
                                     intercept = intercept)
        }

        valid_l <- c(valid_l, loss$item())
      })
      model$losses$valid_l[epoch] <- mean(valid_l)
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

    model$weights[[epoch]] <- lapply(model$net$parameters,function(x) torch::as_array(x$to(device="cpu")))


    ### create plot ###
    if(model$training_properties$plot) visualize.training(model$losses,epoch, new = plot_new)
    plot_new <- FALSE

    ### early stopping ###
    if(is.numeric(model$training_properties$early_stopping)) {
      if(model$training_properties$validation != 0 & !is.null(valid_dl)){
        if(model$losses$valid_l[epoch] < best_val_loss) {
          best_val_loss = model$losses$valid_l[epoch]
          counter = 0
        }

      } else {
        if(model$losses$train_l[epoch] < best_train_loss) {
          best_train_loss = model$losses$train_l[epoch]
          counter = 0
        }
      }

      if(counter >= model$training_properties$early_stopping) {
        break
      }
      counter = counter + 1

    }

  }

  if(!is.null(model$loss$parameter)) model$parameter <- lapply(model$loss$parameter, as.numeric)
  model$use_model_epoch <- epoch
  model$loaded_model_epoch <- epoch


  return(model)
}

regularize_weights <- function (loss, parameters, alpha, lambda, intercept = TRUE){

  weight_layers <- names(which(sapply(parameters, function(x) length(dim(x))) > 1))

  if(intercept){
    l1 <- torch::torch_sum(torch::torch_abs(parameters$`0.weight`$hsplit(1)[[2]]))
    l1 <- l1$mul(1-alpha[1])
    l2 <- torch::torch_norm(parameters$`0.weight`$hsplit(1)[[2]],p=2L)
    l2 <- l2$mul(alpha[1])
    regularization <- torch::torch_add(l1,l2)
    regularization <- regularization$mul(lambda)
    loss <-  torch::torch_add(loss, regularization)

    weight_layers <- weight_layers[-1]
    alpha <- alpha[-1]
  }

  for (i in 1:length(weight_layers)) {
    if(!is.na(alpha[i])) {
        l1 <- torch::torch_sum(torch::torch_abs(parameters[[weight_layers[i]]]))
        l1 <- l1$mul(1-alpha[i])
        l2 <- torch::torch_norm(parameters[[weight_layers[i]]],p=2L)
        l2 <- l2$mul(alpha[i])
        regularization <- torch::torch_add(l1,l2)
        regularization <- regularization$mul(lambda)
        loss <-  torch::torch_add(loss, regularization)
    }
  }

  return(loss)
}
