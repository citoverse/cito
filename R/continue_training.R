#' Contiinues training of a model for additional periods
#'
#' @param model a model created by \code{\link{dnn}}
#' @param data matrix or data.frame
#' @param epochs additional epochs the training should continue for
#' @param validation percentage of data set that should be taken as validation set (chosen randomly)
#' @param continue_from define which epoch should be used as starting point for training, 0 if last epoch should be used
#' @param optimizer which optimizer used for training the network,
#' @param lr learning rate given to optimizer
#' @param batchsize how many samples data loader loads per batch
#' @param shuffle TRUE if data should be reshuffled every epoch (default: FALSE)
#' @param plot plot training loss
#' @param device device on which network should be trained on, either "cpu" or "cuda"
#' @param early_stopping training stops if validation error n epochs before was lower than in current epoch
#' @param lr_scheduler learning rate scheduler, can be "lambda", "multiplicative", "one_cycle" or "step" additional arguments bust be defined in config with "lr_scheduler." as prefix
#' @param config list of additional arguments to be passed to optimizer or lr_scheduler should start with optimizer. and lr_scheduler.
#' @return a model of class cito.dnn same as created by  \code{\link{dnn}}
#'
#' @examples
#' \dontrun{
#' library(cito)
#'
#' set.seed(222)
#' validation_set<- sample(c(1:nrow(datasets::iris)),25)
#'
#' # Build and train  Network
#' nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,], epochs = 32)
#'
#' # continue training for another 32 epochs
#' # nn.fit<- continue_training(nn.fit,data = datasets::iris[-validation_set,])
#'
#' # Use model on validation set
#' predictions <- predict(nn.fit, iris[validation_set,])
#' }
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

  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
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
  for (i  in 1:length(model$training_properties)){
    if(is.character(unlist(model$training_properties[i]))) parantheses<- "\"" else parantheses<- ""
    eval(parse(text=paste0(names(model$training_properties)[i], " <- ", parantheses,model$training_properties[i],parantheses)))
  }

  lr <- model$training_properties$lr
  lr_scheduler <- model$training_properties$lr_scheduler
  optimizer <- model$training_properties$optimizer
  config_optimizer <- model$training_properties$config_optimizer
  config_lr_scheduler <- model$training_properties$config_lr_scheduler
  epoch <- model$training_properties$epoch
  early_stopping <- model$training_properties$early_stopping
  plot <-  model$training_properties$plot
  validation <- model$training_properties$validation
  device <- model$training_properties$device
  alpha <- model$training_properties$alpha
  if(!is.null(changed_params)){
    for (i in 1:length(changed_params)){
      if(is.character(unlist(changed_params[i]))) parantheses<- "\"" else parantheses<- ""
      eval(parse(text=paste0(names(changed_params)[i], " <- ", parantheses,changed_params[i],parantheses)))
    }
  }


  ### set dataloader  ###

  if(is.null(data)) data <- model$data
  X = stats::model.matrix(model$formula, data)
  Y = stats::model.response(stats::model.frame(model$formula, data))

  if(validation != 0){

    train <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(validation*nrow(X))))
    valid <- c(1:nrow(X))[-train]
    train_dl <- get_data_loader(X[train,],Y[train,], batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
    valid_dl <- get_data_loader(X[valid,],Y[valid,], batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)

  }else{
    train_dl <- get_data_loader(X,Y, batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
  }


  ### set optimizer ###
  optim <- get_optimizer(optimizer = optimizer,
                         parameters = c(net$parameters, fam$parameter),
                         lr = lr,
                         config_optimizer = config_optimizer)


  ### set LR Scheduler ###
  if(!isFALSE(lr_scheduler)){
    use_lr_scheduler <- TRUE
    scheduler<- get_lr_scheduler(lr_scheduler = lr_scheduler, config_lr_scheduler = config_lr_scheduler, optimizer= optim)
  }else{
    use_lr_scheduler <- FALSE
  }

  ### training loop ###
  weights <- model$weights
  net<- model$net
  net$to(device = device)
  loss.fkt <- model$family$loss
  losses <- rbind(model$losses[1:model$use_model_epoch,],
                  data.frame(epoch=c(model$use_model_epoch:(model$use_model_epoch+epochs)),train_l=NA,valid_l= NA))
  if((length(model$model_properties$hidden)+1) != length(alpha)) alpha <- rep(alpha[1],length(hidden)+1)

  for (epoch in model$use_model_epoch:(model$use_model_epoch+epochs)) {
    train_l <- c()

    coro::loop(for (b in train_dl) {
      optim$zero_grad()
      output <- net(b[[1]]$to(device = device))

      loss <- loss.fkt(output, b[[2]]$to(device = device))$mean()
      loss <- generalize_alpha(parameters = net$parameters, alpha = alpha,
                               loss = loss,intercept= colnames(X)[1]=="(Intercept)")
      loss$backward()
      optim$step()

      if(use_lr_scheduler) scheduler$step()
      train_l <- c(train_l, loss$item())
      losses$train_l[epoch] <- mean(train_l)
    })

    if(validation != 0){

      valid_l <- c()

      coro::loop(for (b in valid_dl) {
        output <- net(b[[1]]$to(device = device))
        loss <- loss.fkt(output, b[[2]]$to(device = device))$mean()
        loss <- generalize_alpha(parameters = net$parameters, alpha = alpha,
                                 loss = loss,intercept= colnames(X)[1]=="(Intercept)")
        valid_l <- c(valid_l, loss$item())
        losses$valid_l[epoch] <- mean(valid_l)
      })

      cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f, lr: %3.5f\n",
                  epoch, losses$train_l[epoch], losses$valid_l[epoch],optim$param_groups[[1]]$lr))
      if(epoch>early_stopping && is.numeric(early_stopping) &&
         losses$valid_l[epoch-early_stopping]<losses$valid_l[epoch]) {
        weights[[epoch]]<- lapply(net$parameters,function(x) torch::as_array(x$to(device="cpu")))
        if(plot) visualize.training(losses,epoch)
        break
      }

    }else{
      cat(sprintf("Loss at epoch %d: %3f, lr: %3.5f\n", epoch, losses$train_l[epoch],optim$param_groups[[1]]$lr))
    }

    weights[[epoch]] <- lapply(net$parameters,function(x) torch::as_array(x$to(device="cpu")))


    ### create plot ###
    if(plot) visualize.training(losses,epoch)

  }




  return(model)
  }
