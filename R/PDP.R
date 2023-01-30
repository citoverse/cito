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
#' @param variable variable as string for which the PDP should be done. If none is supplied it is done for all variables.
#' @param data specify new data PDP should be performed . If NULL, PDP is performed on the training data.
#' @param ice Individual Conditional Dependence will be shown if TRUE
#' @param resolution.ice resolution in which ice will be computed
#' @param plot plot ALE or not
#' @return A list of plots made with 'ggplot2' consisting of an individual plot for each defined variable.
#' @seealso \code{\link{ALE}}
#' @example /inst/examples/PDP-example.R
#' @export

PDP <- function(model,
                variable = NULL,
                data = NULL,
                ice = FALSE,
                resolution.ice = 20,
                plot=TRUE){

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package \"ggplot2\" must be installed to use this function.",
      call. = FALSE
    )
  }

  model <- check_model(model)

  if(is.null(data)){
    data <- model$data$data
  }

  if(is.null(variable)) variable <- get_var_names(model$training_properties$formula, data[1,])
  if(!any(variable %in% get_var_names(model$training_properties$formula, data[1,]))){
    warning("unknown variable")
    return(NULL)
  }

  x <- NULL
  y <- NULL

  perm_data <- stats::model.matrix(model$training_properties$formula, data)

  link <- model$loss$invlink


  p_ret <- sapply (variable,function(v){

    results =
      lapply(1:model$model_properties$output, function(n_output) {
    if(is.numeric(data[,v])){
      df <- data.frame(
        x = data[,v],
        y = sapply(seq_len(nrow(data)),function(i){
          perm_data[,v]<- perm_data[i,v]
          return(as.numeric(mean(link(model$net(torch::torch_tensor(perm_data))   )[,n_output,drop=FALSE] ))  )
        })
      )
      df <- df[order(df$x),]

      if(!is.null(model$data$ylvls)) {
        label = paste0("PDP - ", model$data$ylvls[n_output])
      } else {
        label = "PDP"
      }

      p <- ggplot2::ggplot(data=df, mapping = ggplot2::aes(x=x,y=y ))
      p <- p + ggplot2::geom_line()
      p <- p + ggplot2::ggtitle(label = label)
      p <- p + ggplot2::xlab(label = v)
      p <- p + ggplot2::ylab(label = as.character(model$call$formula[2]))
      p <- p + ggplot2::geom_rug(sides = "b")


      if(ice){
        perm_dat<-stats::model.matrix(model$training_properties$formula, data)
        instances <- seq(from = min(perm_dat[,v]),
                         to = max(perm_dat[,v]),
                         length.out = resolution.ice + 1)
        #instances = sample(unique(perm_dat[,v]), resolution.ice)

        df_ice <- lapply(seq_len(length(instances)), function(i){
          perm_dat<-stats::model.matrix(model$training_properties$formula, data)
          perm_dat[,v] <- instances[i]
          return(cbind(instances[i] ,as.numeric(link(model$net(torch::torch_tensor(perm_dat)))[,n_output,drop=FALSE] ), 1:nrow(perm_dat) ))
        })

        df_ice<- do.call(rbind, df_ice)
        colnames(df_ice) = c("x", "y", "group")
        df_ice = as.data.frame(df_ice)
        df_ice$group = as.factor(df_ice$group)

        p <- p + ggplot2::geom_line(data = df_ice, mapping = ggplot2::aes_string(x = "x", y = "y", group = "group" ))
        p <- p + ggplot2::geom_line(colour = "yellow", size = 2, data=df, mapping = ggplot2::aes(x=x,y=y))
        }
    }else if (is.factor(data[,v])){
      perm_data<- data
      df <- data.frame(
        x = c(sapply(levels(data[,v]), function(i){
          return(rep(i,nrow(perm_data)))
        })) ,
        y = c(sapply(levels(data[,v]), function(i){
          for(j in seq_len(nrow(perm_data))){
            perm_data[j,v] <- i
          }
          return(stats::predict(model,perm_data)[,n_output,drop=FALSE])
        }))
      )
      df$x<- as.factor(df$x)
      df<- data.frame(x = levels(df$x),
                      y = sapply(levels(df$x), function(i){
                        return(mean(df$y[which(df$x==i)]))
                      }))

      if(ice) message("ice not available for categorical features")

      if(!is.null(model$data$ylvls)) {
        label = paste0("PDP - ", model$data$ylvls[n_output])
      } else {
        label = "PDP"
      }

      p <- ggplot2::ggplot(data = df,mapping = ggplot2::aes(x = x,y = y),)
      p <- p + ggplot2::geom_bar(stat= "identity")
      p <- p + ggplot2::theme_minimal()
      p <- p + ggplot2::geom_text(ggplot2::aes(label=y), vjust=1.6)
      p <- p + ggplot2::ggtitle(label = label)
      p <- p + ggplot2::xlab(label = v)
      p <- p + ggplot2::ylab(label = as.character(model$call$formula[2]))
      p <- p + ggplot2::xlab(v) + ggplot2::ylab(model$call$formula[2])
      p <- p + ggplot2::theme_bw()
    }
    return(p)
  })
    results[sapply(results, is.null)] = NULL
    return(results)
  })

  p_ret = do.call(list, p_ret)
  if(plot) {
    if(model$model_properties$output >1) do.call(gridExtra::grid.arrange, c(p_ret, nrow = ceiling(length(p_ret)/model$model_properties$output)))
    else do.call(gridExtra::grid.arrange, c(p_ret, ncol = length(p_ret)))
  }
  if(!is.null(model$data$ylvls)) {
    names(p_ret) = paste0(model$data$ylvls, "_",names(p_ret))
  }
  return(invisible(p_ret))
}

