visualize.training <- function(losses,epoch,new = FALSE, baseline = NULL){
  if (epoch==1|new){

    graphics::plot(c(),c(),xlim=c(1,nrow(losses)),ylim=c(0,max(losses$train_l,losses$valid_l,baseline,na.rm=T)),
                   main= "Training of DNN",
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
#' After training a model with cito, this function helps to analyze the training process and decide on best performing model.
#' Creates a 'plotly' figure which allows to zoom in and out on training graph
#'
#' @param object a model created by \code{\link{dnn}}
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
  if(!inherits(object, c("citodnn", "citodnnBootstrap"))) stop("Function requires an object of class citodnn")


  if(inherits(object, "citodnn")) {
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
    fig<- plotly::layout(fig,
                         title='DNN Training',
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
