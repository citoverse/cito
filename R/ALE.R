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
#' @param neighborhoods number of neighborhoods original feature space gets divided into
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
        warning("There are neighborhoods with no observations, amount of neighborhoods gets reduced by one")
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
    p <- p + ggplot2::ggtitle(label = "Accumulated Local Effect Plot")
    p <- p + ggplot2::xlab(label = variable)
    p <- p + ggplot2::ylab(label = as.character(model$call$formula[2]))
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
