visualize.training <- function(losses,epoch,new = FALSE){
  if (epoch==1|new){

    graphics::plot(c(),c(),xlim=c(1,nrow(losses)),ylim=c(0,max(losses$train_l[1],losses$valid_l[1],na.rm=T)),
                   main= "Training of DNN",
                   xlab= "epoch",
                   ylab= "loss")
    graphics::legend("top",legend= c("training","validation"),
                     col= c("#000080","#FF8000"),lty=1:2, cex=0.8,
                     title="Line types", text.font=4, bg='grey91')

    graphics::points(x=c(1),y=c(losses$train_l[1]),pch=19, col="#000080", lty=1)
    graphics::points(x=c(1),y=c(losses$valid_l[1]),pch=18, col="#FF8000", lty=2)
    if(epoch > 1){
      for ( i in c(2:epoch)){
        graphics::lines(c(i-1,i), c(losses$train_l[i-1],losses$train_l[i]), pch=19, col="#000080", type="b", lty=1)
        graphics::lines(c(i-1,i), c(losses$valid_l[i-1],losses$valid_l[i]), pch=18, col="#FF8000", type="b", lty=2)
      }
    }
  } else{

    graphics::lines(c(epoch-1,epoch), c(losses$train_l[epoch-1],losses$train_l[epoch]), pch=19, col="#000080", type="b", lty=1)
    graphics::lines(c(epoch-1,epoch), c(losses$valid_l[epoch-1],losses$valid_l[epoch]), pch=18, col="#FF8000", type="b", lty=2)
  }
}

#' Visualize training of Neural Network
#'
#' After training a model with cito, this function helps to analyze the training process and decide on best performing model.
#' Creates a plotly figure which allows to zoom in and out on training graph
#'
#' @param object a model created by \code{\link{dnn}}
#' @return a plotly figure
#' @example /inst/examples/analyze_training-example.R
#' @export

analyze_training<- function(object){

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop(
      "Package \"plotly\" must be installed to use this function.",
      call. = FALSE
    )
  }
  if(!inherits(object,"citodnn")) stop("Function requires an object of class citodnn")

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


#' Partial Dependence Plot (PDP)
#'
#' Calculates the Partial Dependency Plot for one feature, either numeric or categorical. Returns it as a plot.
#'
#' @details
#'
#' Performs the estimation of the partial function \eqn{\hat{f}_S}{}
#'
#' \eqn{\hat{f}_S(x_S)=\frac{1}{n}\sum_{i=1}^n\hat{f}(x_S,x^{(i)}_{C})}{}
#'
#' with a Monte Carlo Estimation:
#'
#' \eqn{\hat{f}_S(x_S)=\frac{1}{n}\sum_{i=1}^n\hat{f}(x_S,x^{(i)}_{C})}{}
#'
#' If a categorical feature is analyzed, all data instances are used and set to each level.
#' Then an average is calculated per category and put out in a bar plot.
#'
#'
#' @param model a model created by \code{\link{dnn}}
#' @param variable variable as string for which the PDP should be done
#' @seealso \code{\link{ALE}}
#' @example /inst/examples/PDP-example.R
#'
#' @export

PDP <- function(model, variable){

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package \"ggplot2\" must be installed to use this function.",
      call. = FALSE
    )
  }

  if(!(variable %in% get_var_names(model$training_properties$formula, model$data$data[1,]))){
    stop("Variable unknown")
  }
  x <- NULL
  y <- NULL

  perm_data <- model$data$data

  if(is.numeric(model$data$data[,variable])){
    df <- data.frame(
      x = model$data$data[,variable],
      y = sapply(seq_len(nrow(model$data$data)),function(i){
        perm_data[,variable]<- perm_data[i,variable]
        return(mean(stats::predict(model,perm_data)))
        })
    )
    df <- df[order(df$x),]

    p <- ggplot2::ggplot(data=df, mapping = ggplot2::aes(x=x,y=y ))
    p <- p + ggplot2::geom_line()
    p <- p + ggplot2::geom_rug(sides="b")

  }else if (is.factor(model$data$data[,variable])){
    df <- data.frame(
      x = c(sapply(levels(model$data$data[,variable]), function(i){
        return(rep(i,nrow(perm_data)))
        })) ,
      y = c(sapply(levels(model$data$data[,variable]), function(i){
        for(j in seq_len(nrow(perm_data))){
          perm_data[j,variable] <- i
        }
        return(stats::predict(model,perm_data))
      }))
    )
    df$x<- as.factor(df$x)
    df<- data.frame(x = levels(df$x),
                     y = sapply(levels(df$x), function(i){
                      return(mean(df$y[which(df$x==i)]))
                      }))
    p <- ggplot2::ggplot(data = df,mapping = ggplot2::aes(x = x,y = y),)
    p <- p + ggplot2::geom_bar(stat= "identity")
    p <- p + ggplot2::theme_minimal()
    p <- p + ggplot2::geom_text(ggplot2::aes(label=y), vjust=1.6)
    p <- p + ggplot2::ggtitle("Partial Dependency Plot")
    p <- p + ggplot2::xlab(variable) + ggplot2::ylab(model$call$formula[2])
  }

  return(p)
}

#' Accumulated Local Effect Plot (ALE)
#'
#'
#' Performs an ALE for one feature and returns a centered plot.
#'
#' @details
#'
#' If the defined variable is a numeric feature, the ALE is performed.
#' Here, the non centered effect for feature j with k equally distant neighborhoods is defined as:
#'
#' \eqn{ \hat{\tilde{f}}_{j,ALE}(x)=\sum_{k=1}^{k_j(x)}\frac{1}{n_j(k)}\sum_{i:x_{j}^{(i)}\in{}N_j(k)}\left[\hat{f}(z_{k,j},x^{(i)}_{\setminus{}j})-\hat{f}(z_{k-1,j},x^{(i)}_{\setminus{}j})\right]}
#'
#' Where \eqn{N_j(k)} is the k-th neighborhood and \eqn{n_j(k)} is the number of observations in the k-th neighborhood.
#'
#' The last part of the equation,
#' \eqn{\left[\hat{f}(z_{k,j},x^{(i)}_{\setminus{}j})-\hat{f}(z_{k-1,j},x^{(i)}_{\setminus{}j})\right]}
#' represents the difference in model prediction when the value of feature j is exchanged with the upper and lower border of the current neighborhood.
#' @param model a model created by \code{\link{dnn}}
#' @param variable variable as string for which the PDP should be done
#' @param neighborhoods number of neighborhoods originial feautre space gets divided into
#' @seealso \code{\link{PDP}}
#' @example /inst/examples/ALE-example.R
#' @export

ALE <- function(model, variable, neighborhoods = 10){
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package \"ggplot2\" must be installed to use this function.",
      call. = FALSE
    )
  }

  if(!(variable %in% get_var_names(model$training_properties$formula, model$data$data[1,]))){
    warning("unknown variable")
    return(NULL)
  }
  x <- NULL
  y <- NULL

  if(is.numeric(model$data$data[,variable])){


    repeat{
      borders <- seq(from = min(model$data$data[,variable]),
                    to = max(model$data$data[,variable]),
                    length.out = neighborhoods+1)

      df <- data.frame(
        x = borders[1:neighborhoods] + ((borders[2]-borders[1])/2),

        y = sapply(seq_len(length(borders))[-1], function(i){

        region_indizes <- which(model$data$data[,variable]<= borders[i] &
                                  model$data$data[,variable]>= borders[i-1])

        if(length(region_indizes)>0){
          perm_data <- model$data$data[region_indizes,]

          perm_data[,variable] <- borders[i-1]
          lower_preds <- stats::predict(model, perm_data)

          perm_data[,variable] <- borders[i]
          upper_preds <- stats::predict(model, perm_data)

          return(mean(upper_preds - lower_preds))
        }else{

          return(NA)

        }
        })
      )

      if(any(is.na(df$y))){
          warning("There are neighborhoods with no observation, amount of neighborhoods gets reduced by one")
        neighborhoods <- neighborhoods - 1
      }else{
        break
      }
    }

    for ( i in seq_len(nrow(df))[-1]){
      df$y[i]<- df$y[i-1]+df$y[i]
    }

    df$y <- df$y - mean(df$y)

    p <- ggplot2::ggplot(data=df, mapping = ggplot2::aes(x = x,y = y))
    p <- p + ggplot2::geom_line()

    geom_df<- data.frame(x = model$data$data[,variable])
    p <- p + ggplot2::geom_rug(sides="b", data = geom_df,
                               mapping = ggplot2::aes(x = x),
                               inherit.aes = FALSE)


    return(p)
  }else{
    warning("Categorical features are not yet supported.")
    return(NULL)

  }


}

