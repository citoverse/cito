visualize.training <- function(losses,epoch){
  if (epoch==1){

    graphics::plot(c(),c(),xlim=c(1,nrow(losses)),ylim=c(0,max(losses$train_l[1],losses$valid_l[1],na.rm=T)),
                   main= "Training of DNN",
                   xlab= "epoch",
                   ylab= "loss")
    graphics::legend("top",legend= c("training","validation"),
                     col= c("#000080","#FF8000"),lty=1:2, cex=0.8,
                     title="Line types", text.font=4, bg='grey91')

    graphics::points(x=c(1),y=c(losses$train_l[1]),pch=19, col="#000080", lty=1)
    graphics::points(x=c(1),y=c(losses$valid_l[1]),pch=18, col="#FF8000", lty=2)

  } else{

    graphics::lines(c(epoch-1,epoch), c(losses$train_l[epoch-1],losses$train_l[epoch]), pch=19, col="#000080", type="b", lty=1)
    graphics::lines(c(epoch-1,epoch), c(losses$valid_l[epoch-1],losses$valid_l[epoch]), pch=18, col="#FF8000", type="b", lty=2)
  }
}

#' Visualize training of Neural Network afterwards to decide on best performing model.
#' Creates a plotly figure which allows to zoom in and out on training graph
#'
#' @param object a model created by \code{\link{dnn}}
#' @return a plotly figure
#' @examples
#' \dontrun{
#' library(cito)
#' set.seed(222)
#' validation_set<- sample(c(1:nrow(datasets::iris)),25)
#'
#' # Build and train  Network
#' nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,],validation = 0.1)
#'
#' # show zoomable plot of training and validation losses
#' analyze_training(nn.fit)
#'
#' # set model which is used for predictions to model from epoch 46
#' nn.fit$use_model_epoch <- 46
#'
#' #' # Use model on validation set
#' predictions <- predict(nn.fit, iris[validation_set,])
#'
#' # Scatterplot
#' plot(iris[validation_set,]$Sepal.Length,predictions)
#' }
#' @export

analyze_training<- function(object){

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop(
      "Package \"plotly\" must be installed to use this function.",
      call. = FALSE
    )
  }
  fig <- plotly::plot_ly(object$losses, type = 'scatter', mode = 'lines+markers',
                         width = 900)

  fig<- plotly::add_trace(fig,x = ~epoch, y = ~train_l,text = "Training Loss")
  if(object$call$validation>0 && !is.null(object$call$validation))  {
    fig<- plotly::add_trace(fig,x = ~epoch, y = ~valid_l, text ="Validation loss")
  }
  fig<- plotly::layout(fig, showlegend = F, title='DNN Training',
                       xaxis = list(rangeslider = list(visible = T)),
                       yaxis = list(fixedrange = F))
  fig<- plotly::layout(fig,xaxis = list(zerolinecolor = '#ffff',
                                        zerolinewidth = 2,
                                        gridcolor = 'ffff'),
                       yaxis = list(zerolinecolor = '#ffff',
                                    zerolinewidth = 2,
                                    gridcolor = 'ffff'),
                       plot_bgcolor='#e5ecf6')

  return(fig)
}
