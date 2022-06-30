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
#' If ice is set to true additional the individual conditional dependence will be shown and the original PDP will be colored yellow.
#' These lines show, how each individual data sample reacts to changes in the feature. This option is not available for categorical features.
#' Unlike PDP the ICE curves are computed with a value grid instead of utilizing every value of every data entry.
#'
#' @param model a model created by \code{\link{dnn}}
#' @param variable variable as string for which the PDP should be done
#' @param data specify new data PDP should be performed on, otherwise PDP is performed on the training data
#' @param ice Individual Conditional Dependence will be shown if TRUE
#' @param resolution.ice resolution in which ice will be computed
#' @seealso \code{\link{ALE}}
#' @example /inst/examples/PDP-example.R
#'
#' @export

PDP <- function(model, variable, data = NULL, ice = FALSE, resolution.ice = 20){

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package \"ggplot2\" must be installed to use this function.",
      call. = FALSE
    )
  }

  if(missing(data)){
    data <- model$data$data
  }


  if(!(variable %in% get_var_names(model$training_properties$formula, data[1,]))){
    stop("Variable unknown")
  }

  x <- NULL
  y <- NULL

  perm_data <- data

  if(is.numeric(data[,variable])){
    df <- data.frame(
      x = data[,variable],
      y = sapply(seq_len(nrow(data)),function(i){
        perm_data[,variable]<- perm_data[i,variable]
        return(mean(stats::predict(model,perm_data)))
      })
    )
    df <- df[order(df$x),]

    p <- ggplot2::ggplot(data=df, mapping = ggplot2::aes(x=x,y=y ))
    p <- p + ggplot2::geom_line()
    p <- p + ggplot2::ggtitle(label = "Partial Dependency Plot")
    p <- p + ggplot2::xlab(label = variable)
    p <- p + ggplot2::ylab(label = as.character(model$call$formula[2]))
    p <- p + ggplot2::geom_rug(sides="b")

    if(ice){
      instances <- seq(from = min(data[,variable]),
                       to = max(data[,variable]),
                       length.out = resolution.ice + 1)

      df_ <- sapply(seq_len(length(instances)), function(i){
        perm_dat<- data
        perm_dat[,variable] <- instances[i]
        return(stats::predict(model,perm_dat))
      })

      df_<- data.frame( x = instances,
                       y = c(t(df_)))

      p <- p + ggplot2::geom_line(data = df_, mapping = ggplot2::aes(x = x, y = y, group = rep(1:nrow(data), each= resolution.ice + 1)))
      p <- p + ggplot2::geom_line(colour = "yellow", size = 2, data=df, mapping = ggplot2::aes(x=x,y=y))
      }
  }else if (is.factor(data[,variable])){
    df <- data.frame(
      x = c(sapply(levels(data[,variable]), function(i){
        return(rep(i,nrow(perm_data)))
      })) ,
      y = c(sapply(levels(data[,variable]), function(i){
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

    if(ice) warning("ice not available for categorical features")
    p <- ggplot2::ggplot(data = df,mapping = ggplot2::aes(x = x,y = y),)
    p <- p + ggplot2::geom_bar(stat= "identity")
    p <- p + ggplot2::theme_minimal()
    p <- p + ggplot2::geom_text(ggplot2::aes(label=y), vjust=1.6)
    p <- p + ggplot2::ggtitle(label = "Partial Dependency Plot")
    p <- p + ggplot2::xlab(label = variable)
    p <- p + ggplot2::ylab(label = as.character(model$call$formula[2]))

    p <- p + ggplot2::xlab(variable) + ggplot2::ylab(model$call$formula[2])
    }

  return(p)
}

