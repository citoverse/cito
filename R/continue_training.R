#' Contiinues training of a model for additional periods
#'
#' @param model a model created by \code{\link{dnn}}
#' @param epochs additional epochs the training should continue for
#' @param early_stopping training stops if validation error n epochs before was lower than in current epoch
#' @param lr_scheduler lr scheduler
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @export
continue_training <- function(model,
                              epochs =32,
                              early_stopping=FALSE,
                              lr_scheduler= FALSE){

  model<- check_model(model)






  return(model)
  }
