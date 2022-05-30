#' Contiinues training of a model for additional periods
#'
#' @param model a model created by \code{\link{dnn}}
#' @param data matrix or data.frame if not provided data from original training will be used
#' @param epochs additional epochs the training should continue for
#' @param continue_from define which epoch should be used as starting point for training, 0 if last epoch should be used
#' @param device device on which network should be trained on, either "cpu" or "cuda"
#' @param changed_params list of arguments to change compared to original training setup, see \code{\link{dnn}} which parameter can be changed
#' @return a model of class cito.dnn same as created by  \code{\link{dnn}}
#'
#' @example /inst/examples/continue_training-example.R
#'
#' @import checkmate
#'
#' @export
continue_training <- function(model,
                              epochs = 32,
                              continue_from= NULL,
                              data=NULL,
                              device= "cuda",
                              changed_params=NULL){

  checkmate::qassert(device, "S+[3,)")

  ### Training device ###
  if(device== "cuda"){
    if (torch::cuda_is_available()) {
      device<- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device<- torch::torch_device("cpu")
    }

  }else {
    if(device!= "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device<- torch::torch_device("cpu")
  }


  ### initiate model ###
  if(!is.null(continue_from)){
    model$use_model_epoch <- continue_from
  }else{
    model$use_model_epoch <- max(which(!is.na(model$losses$train_l)))
  }
  model<- check_model(model)



  ### set training environment ###

  x_dtype = torch::torch_float32()
  y_dtype = torch::torch_float32()

  if(!is.null(changed_params)){
    for (i in 1:length(changed_params)){
      if(is.character(unlist(changed_params[i]))) parantheses<- "\"" else parantheses<- ""
      eval(parse(text=paste0("model$training_properties$",names(changed_params)[i], " <- ", parantheses,changed_params[i],parantheses)))
    }
  }


  ### set dataloader  ###
  fm<- stats::as.formula(model$call$formula)
  if(is.null(data)) data <- model$data$data

  X = stats::model.matrix(fm, data)
  Y = stats::model.response(stats::model.frame(fm, data))
  Y = as.matrix(Y)

  y_dim = ncol(Y)
  x_dtype = torch::torch_float32()
  y_dtype = torch::torch_float32()
  if(is.character(Y)) {
    y_dim = length(unique(as.integer(as.factor(Y[,1]))))
    Y = matrix(as.integer(as.factor(Y[,1])), ncol = 1L)
    if(model$family$family$family == "binomial") {
      Y = torch::as_array(torch::nnf_one_hot( torch::torch_tensor(Y, dtype=torch::torch_long() ))$squeeze() )
    }
  }
  if(model$family$family$family == "softmax") y_dtype = torch::torch_long()



  if(model$training_properties$validation != 0){

    train <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(model$training_properties$validation*nrow(X))))
    valid <- c(1:nrow(X))[-train]
    train_dl <- get_data_loader(X[train,],Y[train,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
    valid_dl <- get_data_loader(X[valid,],Y[valid,], batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, x_dtype=x_dtype, y_dtype=y_dtype)

  }else{
    train_dl <- get_data_loader(X,Y, batch_size = model$training_properties$batchsize, shuffle = model$training_properties$shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
  }


  ### set optimizer ###
  optim <- get_optimizer(optimizer = model$training_properties$optimizer,
                         parameters = c(model$net$parameters, model$family$parameter),
                         lr = model$training_properties$lr,
                         config_optimizer = model$training_properties$config_optimizer)


  ### set LR Scheduler ###
  if(!isFALSE(model$training_properties$lr_scheduler)){
    use_lr_scheduler <- TRUE
    scheduler<- get_lr_scheduler(lr_scheduler = model$training_properties$lr_scheduler,
                                 config_lr_scheduler = model$training_properties$config_lr_scheduler,
                                 optimizer = optim)
  }else{
    use_lr_scheduler <- FALSE
  }

  ### training loop ###
  weights <- model$weights
  net <- model$net
  net$to(device = device)
  loss.fkt <- model$family$loss
  model$losses <- rbind(model$losses[1:model$use_model_epoch,],
                  data.frame(epoch=c((model$use_model_epoch+1):(model$use_model_epoch+epochs)),train_l=NA,valid_l= NA))

  for (epoch in (model$use_model_epoch+1):(model$use_model_epoch+epochs)) {
    train_l <- c()

    coro::loop(for (b in train_dl) {
      optim$zero_grad()
      output <- net(b[[1]]$to(device = device))

      loss <- loss.fkt(output, b[[2]]$to(device = device))$mean()
      loss <- generalize_alpha(parameters = net$parameters,
                               alpha = model$training_properties$alpha,
                               loss = loss,
                               intercept = colnames(X)[1]=="(Intercept)")
      loss$backward()
      optim$step()

      if(use_lr_scheduler) scheduler$step()
      train_l <- c(train_l, loss$item())
      model$losses$train_l[epoch] <- mean(train_l)
    })

    if(model$training_properties$validation != 0){

      valid_l <- c()

      coro::loop(for (b in valid_dl) {
        output <- net(b[[1]]$to(device = device))
        loss <- loss.fkt(output, b[[2]]$to(device = device))$mean()
        loss <- generalize_alpha(parameters = net$parameters,
                                 alpha = model$training_properties$alpha,
                                 loss = loss,
                                 intercept = colnames(X)[1]=="(Intercept)")
        valid_l <- c(valid_l, loss$item())
        model$losses$valid_l[epoch] <- mean(valid_l)
      })

      cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f, lr: %3.5f\n",
                  epoch, model$losses$train_l[epoch], model$losses$valid_l[epoch],optim$param_groups[[1]]$lr))
      if(epoch>model$training_properties$early_stopping && is.numeric(model$training_properties$early_stopping) &&
         model$losses$valid_l[epoch-model$training_properties$early_stopping]<model$losses$valid_l[epoch]) {
        model$weights[[epoch]]<- lapply(net$parameters,function(x) torch::as_array(x$to(device="cpu")))
        if(plot) visualize.training(model$losses,epoch, new = (epoch==model$use_model_epoch +1))
        break
      }

    }else{
      cat(sprintf("Loss at epoch %d: %3f, lr: %3.5f\n", epoch, model$losses$train_l[epoch],optim$param_groups[[1]]$lr))
    }

    model$weights[[epoch]] <- lapply(net$parameters,function(x) torch::as_array(x$to(device="cpu")))


    ### create plot ###
    if(model$training_properties$plot) visualize.training(model$losses,epoch,new = (epoch==model$use_model_epoch +1))

  }



  model$net <- net
  return(model)
  }
