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
                              data = NULL,
                              epochs = 32,
                              continue_from = 0,
                              validation = 0,
                              optimizer = "adam",
                              lr = 0.01,
                              batchsize = 32L,
                              shuffle = FALSE,
                              plot = TRUE,
                              device= "cuda",
                              early_stopping=FALSE,
                              lr_scheduler= FALSE,
                              config=list()){

  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(epochs, "R1[0,)")
  checkmate::qassert(lr, "R+[0,)")
  checkmate::qassert(lr_scheduler,c("S+[1,)","B1"))
  checkmate::qassert(early_stopping,c("R1[1,)","B1"))
  checkmate::qassert(plot,"B1")
  checkmate::qassert(device, "S+[3,)")

  if(continue_from != 0){
    model$use_model_epoch <- continue_from
  }else{
    model$use_model_epoch <- max(which(!is.na(model$losses$train_l)))
  }
    model<- check_model(model)


  ### decipher config list ###
  config_optimizer<-c()
  config_lr_scheduler<- c()
  if(length(config)>0){
    config_optimizer<- config[which(startsWith(tolower(names(config)),"optimizer"))]
    names(config_optimizer)<- sapply(names(config_optimizer),function(x) substring(x,11))

    config_lr_scheduler<- config[which(startsWith(tolower(names(config)),"lr_scheduler"))]
    names(config_lr_scheduler)<- sapply(names(config_lr_scheduler),function(x) substring(x,14))
  }

  ### set training parameters ###

  # model$model_properties

  ### optimizer ###
  #optim<- get_optimizer()

  ###  ###





  return(model)
  }
