#' Partial Dependence Plot (PDP)
#'
#' Calculates the Partial Dependency Plot for one feature, either numeric or categorical. Returns it as a plot.
#'
#'
#' @param model a model created by \code{\link{dnn}}
#' @param variable variable as string for which the PDP should be done. If none is supplied it is done for all variables.
#' @param data specify new data PDP should be performed . If NULL, PDP is performed on the training data.
#' @param ice Individual Conditional Dependence will be shown if TRUE
#' @param resolution.ice resolution in which ice will be computed
#' @param plot plot PDP or not
#' @param parallel parallelize over bootstrap models or not
#' @param ... arguments passed to \code{\link{predict}}
#'
#'
#' @details
#'
#' # Description
#' Performs a Partial Dependency Plot (PDP) estimation to analyze the relationship between a selected feature and the target variable.
#'
#' The PDP function estimates the partial function \eqn{\hat{f}_S}{}:
#'
#' \eqn{\hat{f}_S(x_S)=\frac{1}{n}\sum_{i=1}^n\hat{f}(x_S,x^{(i)}_{C})}{}
#'
#' with a Monte Carlo Estimation:
#'
#' \eqn{\hat{f}_S(x_S)=\frac{1}{n}\sum_{i=1}^n\hat{f}(x_S,x^{(i)}_{C})}{}
#' using a Monte Carlo estimation method. It calculates the average prediction of the target variable for different values of the selected feature while keeping other features constant.
#'
#' For categorical features, all data instances are used, and each instance is set to one level of the categorical feature. The average prediction per category is then calculated and visualized in a bar plot.
#'
#' If the `ice` parameter is set to `TRUE`, the Individual Conditional Expectation (ICE) curves are also shown. These curves illustrate how each individual data sample reacts to changes in the feature value. Please note that this option is not available for categorical features. Unlike PDP, the ICE curves are computed using a value grid instead of utilizing every value of every data entry.
#'
#' Note: The PDP analysis provides valuable insights into the relationship between a specific feature and the target variable, helping to understand the feature's impact on the model's predictions.
#' If a categorical feature is analyzed, all data instances are used and set to each level.
#' Then an average is calculated per category and put out in a bar plot.
#'
#' If ice is set to true additional the individual conditional dependence will be shown and the original PDP will be colored yellow.
#' These lines show, how each individual data sample reacts to changes in the feature. This option is not available for categorical features.
#' Unlike PDP the ICE curves are computed with a value grid instead of utilizing every value of every data entry.
#'
#'
#' @return A list of plots made with 'ggplot2' consisting of an individual plot for each defined variable.
#' @seealso \code{\link{ALE}}
#' @example /inst/examples/PDP-example.R
#' @export

PDP <- function(model,
                variable = NULL,
                data = NULL,
                ice = FALSE,
                resolution.ice = 20,
                plot=TRUE,
                parallel = FALSE, ...) UseMethod("PDP")

#' @rdname PDP
#' @export
PDP.citodnn <- function(model,
                variable = NULL,
                data = NULL,
                ice = FALSE,
                resolution.ice = 20,
                plot=TRUE,
                parallel = FALSE, ...) {

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
  group <- NULL

  perm_data <- stats::model.matrix(model$training_properties$formula, data)[, -1, drop=FALSE]

  link <- model$loss$invlink


  p_ret <- sapply (variable,function(v){

    results = getPDP(model = model, v = v, data = data,
                     resolution.ice = resolution.ice, ice = ice, perm_data = perm_data, link = link, ...)
    results[sapply(results, is.null)] = NULL
    return(results)
  })

  p_ret = lapply(p_ret, function(res) {
    if(is.numeric(data[,res$v])){
      p <- ggplot2::ggplot(data=res$df, mapping = ggplot2::aes(x=x,y=y ))
      p <- p + ggplot2::geom_line()
      p <- p + ggplot2::ggtitle(label = res$label)
      p <- p + ggplot2::xlab(label = res$v)
      p <- p + ggplot2::ylab(label = as.character(model$call$formula[[2]]))
      p <- p + ggplot2::geom_rug(sides = "b")
      if(ice) {
        p <- p + ggplot2::geom_line(data = res$df_ice, mapping = ggplot2::aes(x = x, y = y, group = group ))
        p <- p + ggplot2::geom_line(colour = "yellow", linewidth = 2, data=res$df, mapping = ggplot2::aes(x=x,y=y))
      }
      p <- p + ggplot2::theme_bw()
    } else if (is.factor(data[,res$v])){
      p <- ggplot2::ggplot(data = res$df,mapping = ggplot2::aes(x = x,y = y),)
      p <- p + ggplot2::geom_bar(stat= "identity")
      p <- p + ggplot2::theme_minimal()
      p <- p + ggplot2::geom_text(ggplot2::aes(label=y), vjust=1.6)
      p <- p + ggplot2::ggtitle(label = res$label)
      p <- p + ggplot2::xlab(label = res$v)
      p <- p + ggplot2::ylab(label = as.character(model$call$formula[[2]]))
      p <- p + ggplot2::xlab(res$v) + ggplot2::ylab(model$call$formula[[2]])
      p <- p + ggplot2::theme_bw()
    }
    return(p)
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



#' @rdname PDP
#' @export
PDP.citodnnBootstrap <- function(model,
                        variable = NULL,
                        data = NULL,
                        ice = FALSE,
                        resolution.ice = 20,
                        plot=TRUE,
                        parallel = FALSE ,...) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package \"ggplot2\" must be installed to use this function.",
      call. = FALSE
    )
  }


  if(is.null(data)){
    data <- model$data$data
  }

  if(is.null(variable)) variable <- get_var_names(model$models[[1]]$training_properties$formula, data[1,])
  if(!any(variable %in% get_var_names(model$models[[1]]$training_properties$formula, data[1,]))){
    warning("unknown variable")
    return(NULL)
  }

  x <- NULL
  y <- NULL
  ci <- NULL

  perm_data <- stats::model.matrix(model$models[[1]]$training_properties$formula, data)[, -1, drop=FALSE]

  if(parallel == FALSE) {
  pb = progress::progress_bar$new(total = length(model$models), format = "[:bar] :percent :eta", width = round(getOption("width")/2))
  results_boot = list()

  for(b in 1:length(model$models)) {
    model_indv = model$models[[b]]
    model_indv <- check_model(model_indv)

    p_ret <- sapply (variable,function(v){

      results = getPDP(model = model_indv, v = v, data = data,
                       resolution.ice = resolution.ice, ice = ice, perm_data = perm_data, link = model_indv$loss$invlink, ...)
      results[sapply(results, is.null)] = NULL
      return(results)
    })

    pb$tick()
    results_boot[[b]] = p_ret
  }
  } else {
    if(is.logical(parallel)) {
      if(parallel) {
        parallel = parallel::detectCores() -1
      }
    }
    if(is.numeric(parallel)) {
      backend = parabar::start_backend(parallel)
      parabar::export(backend, ls(environment()), environment())
    }

    parabar::configure_bar(type = "modern", format = "[:bar] :percent :eta", width = round(getOption("width")/2))
    results_boot <- parabar::par_lapply(backend, 1:length(model$models), function(b) {
      model_indv = model$models[[b]]
      model_indv <- check_model(model_indv)

      p_ret <- sapply (variable,function(v){

        results = getPDP(model = model_indv, v = v, data = data,
                                resolution.ice = resolution.ice, ice = ice, perm_data = perm_data, link = model_indv$loss$invlink, ...)
        results[sapply(results, is.null)] = NULL
        return(results)
      })
      return(p_ret)
    })
    parabar::stop_backend(backend)
  }

  p_ret = lapply(1:length(results_boot[[1]]), function(j) {

    df_tmp <- data.frame(
      x = results_boot[[1]][[j]]$df[,1],
      mean = apply(sapply(1:length(results_boot), function(i) results_boot[[i]][[j]]$df[,2]), 1, mean),
      ci = 1.96*apply(sapply(1:length(results_boot), function(i) results_boot[[i]][[j]]$df[,2]), 1, stats::sd)
    )

    if(is.numeric(data[,results_boot[[1]][[j]]$v])){
      p <- ggplot2::ggplot(data=df_tmp, mapping = ggplot2::aes(x=x,y=y ))
      p <- p + ggplot2::geom_ribbon(ggplot2::aes(ymin = mean - ci, ymax = mean+ci), fill = "grey70")
      p <- p + ggplot2::geom_line(ggplot2::aes(y = mean))
      p <- p + ggplot2::ggtitle(label = results_boot[[1]][[j]]$label)
      p <- p + ggplot2::xlab(label = results_boot[[1]][[j]]$v)
      p <- p + ggplot2::ylab(label = as.character(model$models[[1]]$call$formula[[2]]))
      p <- p + ggplot2::geom_rug(sides = "b")
      p <- p + ggplot2::theme_bw()

    } else if (is.factor(data[,results_boot[[1]][[j]]$v])){
      p <- ggplot2::ggplot(data = df_tmp,mapping = ggplot2::aes(x = x,y = y),)
      p <- p + ggplot2::geom_bar(ggplot2::aes(y = mean), stat= "identity")
      p <- p + ggplot2::geom_errorbar(ggplot2::aes(ymin=mean-ci, ymax=mean+ci), width=.2,
                    position=ggplot2::position_dodge(.9))
      p <- p + ggplot2::theme_minimal()
      #p <- p + ggplot2::geom_text(ggplot2::aes(label=y), vjust=1.6)
      p <- p + ggplot2::ggtitle(label = results_boot[[1]][[j]]$label)
      p <- p + ggplot2::xlab(label = results_boot[[1]][[j]]$v)
      p <- p + ggplot2::ylab(label = as.character(model$models[[1]]$call$formula[[2]]))
      p <- p + ggplot2::xlab(results_boot[[1]][[j]]$v) + ggplot2::ylab(model$models[[1]]$call$formula[[2]])
      p <- p + ggplot2::theme_bw()
    }
    return(p)
  })


  p_ret = do.call(list, p_ret)
  if(plot) {
    if(model$models[[1]]$model_properties$output >1) do.call(gridExtra::grid.arrange, c(p_ret, nrow = ceiling(length(p_ret)/model$models[[1]]$model_properties$output)))
    else do.call(gridExtra::grid.arrange, c(p_ret, ncol = length(p_ret)))
  }
  if(!is.null(model$data$ylvls)) {
    names(p_ret) = paste0(model$data$ylvls, "_",names(p_ret))
  }
  return(invisible(p_ret))



}



getPDP = function(model, data, K, v, ice = FALSE, resolution.ice,  perm_data , link, ... ) {

  device = model$net$parameters[[1]]$device
  dtype = model$net$parameters[[1]]$dtype

  return(
    lapply(1:model$model_properties$output, function(n_output) {
      df_ice = NULL
      if(is.numeric(data[,v])){
        df <- data.frame(
          x = data[,v],
          y = sapply(seq_len(nrow(data)),function(i){
            perm_data[,v]<- perm_data[i,v]
            return(as.numeric(mean(link(model$net(torch::torch_tensor(perm_data,
                                                                      device = device,
                                                                      dtype = dtype))   )[,n_output,drop=FALSE] ))  )
          })
        )
        df <- df[order(df$x),]

        if(!is.null(model$data$ylvls)) {
          label = paste0("PDP - ", model$data$ylvls[n_output])
        } else {
          label = "PDP"
        }

        if(ice){
          perm_dat<-stats::model.matrix(model$training_properties$formula, data)[, -1, drop=FALSE]
          instances <- seq(from = min(perm_dat[,v]),
                           to = max(perm_dat[,v]),
                           length.out = resolution.ice + 1)
          #instances = sample(unique(perm_dat[,v]), resolution.ice)

          df_ice <- lapply(seq_len(length(instances)), function(i){
            perm_dat<-stats::model.matrix(model$training_properties$formula, data)[, -1, drop=FALSE]
            perm_dat[,v] <- instances[i]
            return(cbind(instances[i] ,as.numeric(link(model$net(torch::torch_tensor(perm_dat,
                                                                                     device = device,
                                                                                     dtype = dtype)))[,n_output,drop=FALSE] ), 1:nrow(perm_dat) ))
          })

          df_ice<- do.call(rbind, df_ice)
          colnames(df_ice) = c("x", "y", "group")
          df_ice = as.data.frame(df_ice)
          df_ice$group = as.factor(df_ice$group)

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
            return(stats::predict(model,perm_data, ...)[,n_output,drop=FALSE])
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

      }
      #return(p)
      return(list(df = df, label = label, v = v, df_ice = df_ice ))
    })
  )
}





#' Accumulated Local Effect Plot (ALE)
#'
#'
#' Performs an ALE for one or more features.
#'
#'
#' @param model a model created by \code{\link{dnn}}
#' @param variable variable as string for which the PDP should be done
#' @param data data on which ALE is performed on, if NULL training data will be used.
#' @param type ALE on which scale response or link, default is response
#' @param K number of neighborhoods original feature space gets divided into
#' @param ALE_type method on how the feature space is divided into neighborhoods.
#' @param plot plot ALE or not
#' @param parallel parallelize over bootstrap models or not
#' @param ... arguments passed to \code{\link{predict}}
#'
#' @details
#'
#' # Explanation
#'
#' Accumulated Local Effect plots (ALE) quantify how the predictions change when the features change. They are similar to partial dependency plots but are more robust to feature collinearity.
#'
#' # Mathematical details
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
#'
#'
#' @seealso \code{\link{PDP}}
#' @return A list of plots made with 'ggplot2' consisting of an individual plot for each defined variable.
#' @example /inst/examples/ALE-example.R
#' @export
ALE <- function(model,
                variable = NULL,
                data = NULL,
                type = "response",
                K = 10,
                ALE_type = c("equidistant", "quantile"),
                plot=TRUE,
                parallel = FALSE, ...) UseMethod("ALE")

#' @rdname ALE
#' @export
ALE.citodnn <- function(model,
                        variable = NULL,
                        data = NULL,
                        type = "response",
                        K = 10,
                        ALE_type = c("equidistant", "quantile"),
                        plot=TRUE,
                        parallel = FALSE, ...){
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package \"ggplot2\" must be installed to use this function.",
      call. = FALSE
    )
  }
  model <- check_model(model)
  ALE_type <- match.arg(ALE_type)
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

  is_categorical = sapply(data[, variable], is.factor )

  if(any(is_categorical)) {
    cat("Categorical features are not yet supported.\n")
    variable = variable[!is_categorical]
  }
  p_ret <- sapply (variable,function(v){
    results = getALE(model = model, v = v, ALE_type = ALE_type, data = data, K = K, type = type, ...)

    results[sapply(results, is.null)] = NULL

    return(results)
  })

  p_ret = lapply(p_ret, function(res) {
    p <- ggplot2::ggplot(data=res$df, mapping = ggplot2::aes(x = x,y = y))
    p <- p + ggplot2::geom_line()
    p <- p + ggplot2::ggtitle(label = res$label)
    p <- p + ggplot2::xlab(label = res$v)
    p <- p + ggplot2::ylab(label = "ALE")
    geom_df<- data.frame(x = res$data)
    p <- p + ggplot2::geom_rug(sides="b", data = geom_df,
                               mapping = ggplot2::aes(x = x),
                               inherit.aes = FALSE)
    p <- p + ggplot2::theme_bw()
    return(p)
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


#' @rdname ALE
#' @export
ALE.citodnnBootstrap <- function(model,
                                 variable = NULL,
                                 data = NULL,
                                 type = "response",
                                 K = 10,
                                 ALE_type = c("equidistant", "quantile"),
                                 plot=TRUE,
                                 parallel = FALSE,
                                 ...){
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package \"ggplot2\" must be installed to use this function.",
      call. = FALSE
    )
  }

  ALE_type <- match.arg(ALE_type)
  # parabar::configure_bar(ALE_type = "modern", format = "[:bar] :percent :eta", width = getOption("width")/2)
  ci <- NULL

  if(parallel == FALSE) {
    pb = progress::progress_bar$new(total = length(model$models), format = "[:bar] :percent :eta", width = round(getOption("width")/2))
    results_boot = list()

    for(b in 1:length(model$models)) {
      model_indv = model$models[[b]]
      model_indv <- check_model(model_indv)
      if(is.null(data)){
        data <- model$data$data
      }
      if(is.null(variable)) variable <- get_var_names(model_indv$training_properties$formula, data[1,])
      if(!any(variable %in% get_var_names(model_indv$training_properties$formula, data[1,]))){
        warning("unknown variable")
        return(NULL)
      }
      x <- NULL
      y <- NULL
      is_categorical = sapply(data[, variable], is.factor )
      if(any(is_categorical)) {
        # cat("Categorical features are not yet supported.\n")
        variable = variable[!is_categorical]
      }
      p_ret <- sapply (variable,function(v){
        results = getALE(model = model_indv, v = v, ALE_type = ALE_type,type = type, data = data, K = K, verbose = FALSE, ...)
        results[sapply(results, is.null)] = NULL
        return(results)
      })
      pb$tick()
      results_boot[[b]] = p_ret
    }
  } else {
    if(is.logical(parallel)) {
      if(parallel) {
        parallel = parallel::detectCores() -1
      }
    }
    if(is.numeric(parallel)) {
      backend = parabar::start_backend(parallel)
      parabar::export(backend, ls(environment()), environment())
    }

    parabar::configure_bar(type = "modern", format = "[:bar] :percent :eta", width = round(getOption("width")/2))
    results_boot <- parabar::par_lapply(backend, 1:length(model$models), function(b) {
      model_indv = model$models[[b]]
      model_indv <- check_model(model_indv)
      if(is.null(data)){
        data <- model$data$data
      }
      if(is.null(variable)) variable <- get_var_names(model_indv$training_properties$formula, data[1,])
      if(!any(variable %in% get_var_names(model_indv$training_properties$formula, data[1,]))){
        warning("unknown variable")
        return(NULL)
      }
      x <- NULL
      y <- NULL
      is_categorical = sapply(data[, variable], is.factor )
      if(any(is_categorical)) {
        # cat("Categorical features are not yet supported.\n")
        variable = variable[!is_categorical]
      }
      p_ret <- sapply (variable,function(v){
        results = getALE(model = model_indv, v = v, ALE_type = ALE_type, data = data, K = K, verbose = FALSE, ...)
        results[sapply(results, is.null)] = NULL
        return(results)
      })
      return(p_ret)
    })
    parabar::stop_backend(backend)

  }
  p_ret =
    lapply(1:length(results_boot[[1]]), function(j) {
      df_tmp = data.frame(
        x = results_boot[[1]][[j]]$df[,1],
        mean = apply(sapply(1:length(results_boot), function(i) results_boot[[i]][[j]]$df[,2]), 1, mean),
        ci = 1.96*apply(sapply(1:length(results_boot), function(i) results_boot[[i]][[j]]$df[,2]), 1, stats::sd)
      )
      p =
        ggplot2::ggplot(data =df_tmp, ggplot2::aes(x = x, y = mean)) +
        ggplot2::geom_ribbon(ggplot2::aes(ymin = mean - ci, ymax = mean+ci), fill = "grey70") +
        ggplot2::geom_line(ggplot2::aes(y = mean)) +
        ggplot2::ggtitle(label = results_boot[[1]][[j]]$label) +
        ggplot2::xlab(label = results_boot[[1]][[j]]$v) +
        ggplot2::ylab(label = "ALE") +
        ggplot2::theme_bw()
      return(p)
    })

  p_ret = do.call(list, p_ret)
  if(plot) {
    if(model$models[[1]]$model_properties$output >1) do.call(gridExtra::grid.arrange, c(p_ret, nrow = ceiling(length(p_ret)/model$models[[1]]$model_properties$output)))
    else do.call(gridExtra::grid.arrange, c(p_ret, ncol = length(p_ret)))
  }

  if(!is.null(model$data$ylvls)) {
    names(p_ret) = paste0(model$data$ylvls, "_",names(p_ret))
  }
  return(invisible(p_ret))
}



getALE = function(model, ALE_type, data, K, v, verbose = TRUE, type = "response", ...) {
  return(
    lapply(1:model$model_properties$output, function(n_output) {
      if ( ALE_type == "equidistant"){
        reduced_K <- FALSE
        repeat{
          borders <- seq(from = min(data[,v]),
                         to = max(data[,v]),
                         length.out = K+1)
          df <- data.frame(
            x = borders[1:K] + ((borders[2]-borders[1])/2),
            y = sapply(seq_len(length(borders))[-1], function(i){
              region_indizes <- which(data[,v]<= borders[i] &
                                        data[,v]>= borders[i-1])
              if(length(region_indizes)>0){
                perm_data <- data[region_indizes,]
                perm_data[,v] <- borders[i-1]
                lower_preds <- stats::predict(model, perm_data, type = type,...)[,n_output,drop=FALSE]
                perm_data[,v] <- borders[i]
                upper_preds <- stats::predict(model, perm_data, type = type,...)[,n_output,drop=FALSE]
                return(mean(upper_preds - lower_preds))
              }else{
                return(NA)
              }
            })
          )
          if(any(is.na(df$y))){
            reduced_K <- TRUE
            K <- K - 1
          }else{
            if(reduced_K){
              if(verbose) message(paste0("Number of Neighborhoods reduced to ",K))
            }
            break
          }
        }
      }else if ( ALE_type == "quantile"){

        quants <- stats::quantile(data[,v],probs = seq(0,1,1/K))
        groups <- lapply(c(2:(K+1)),function(i) return(which(data[,v] >= quants[i-1] & data[,v] < quants[i])))
        groups[[length(groups)]] <- c(groups[[length(groups)]],which.max(data[,v]))

        df <- data.frame (
          x = unlist(lapply(c(2:(K+1)), function(i)  return(unname((quants[i]+quants[i-1])/2)))),
          y = unlist(lapply(seq_len(length(groups)), function(i){
            perm_data <- data[groups[[i]],]
            perm_data[,v] <- quants[i]
            lower_preds <- stats::predict(model, perm_data,type = type, ...)[,n_output,drop=FALSE]
            perm_data[,v] <- quants[i+1]
            upper_preds <- stats::predict(model, perm_data,type = type, ...)[,n_output,drop=FALSE]
            return(mean(upper_preds - lower_preds))
          })))

      }
      for ( i in seq_len(nrow(df))[-1]){
        df$y[i]<- df$y[i-1]+df$y[i]
      }
      df$y <- df$y - mean(df$y)
      #if(!is.null(model$data$ylvls)) {
      label = paste0(v," \U2192 ", model$responses[n_output])
      # TODO model$data$responses[n_output]
      #} else {
      #  label = "ALE"
      #}
      return(list(df = df, label = label, data = data[,v], v = v))
    }))
}



get_importance<- function(model, n_permute= NULL, data = NULL, device = "cpu", out_of_bag = FALSE, ...){

  if(out_of_bag) {
    model$data$data = model$data$original$data[-model$data$indices,]
    model$data$X = model$data$original$X[-model$data$indices,]
    if(is.matrix(model$data$Y)) model$data$Y = model$data$original$Y[-model$data$indices,,drop=FALSE]
    if(is.vector(model$data$Y)) model$data$Y = model$data$original$Y[-model$data$indices]
  }

  if(is.null(n_permute)) n_permute <- ceiling(sqrt(nrow(model$data$data))*3)
  model<- check_model(model)
  softmax = FALSE

  if(is.function(model$loss$call)) {
    #warning("Importance is not supported for custom loss functions")
    return(NULL)
    }

  if(inherits(model$loss$call, "character")) {
    if(!any(model$loss$call  == c("softmax","mse", "mae"))){ return(NULL)}
    if(model$loss$call  == "softmax") {
      softmax = TRUE
    }
  } else {
    if(!any(model$loss$call$family == c("binomial")  )){ return(NULL)}
  }
  loss<- model$loss$loss

  true = model$data$Y

  if(inherits(model$loss$call, "character")) {
    true = torch::torch_tensor(model$data$Y)
  } else {
    if(model$loss$call$family  == c("binomial") ){
      mode(true) = "numeric"
      true = torch::torch_tensor(true)
    }
  }

  n_outputs = model$model_properties$output

  if(softmax) {
    n_outputs = 1
  }

  out = NULL

  for(n_prediction in 1:n_outputs) {

    if(n_outputs > 1) true_tmp = true[,n_prediction,drop=FALSE]
    else true_tmp = true

    if(!softmax) org_err <- as.numeric(loss( pred = torch::torch_tensor(stats::predict(model,model$data$data, type = "link", device = device, ...)[,n_prediction,drop=FALSE]) ,true = true_tmp)$mean())
    else org_err <- as.numeric(loss( pred = torch::torch_tensor(stats::predict(model,model$data$data, type = "link", device = device, ...)) ,true = true_tmp)$mean())

    importance <- data.frame(variable = get_var_names(model$training_properties$formula, model$data$data[1,]),
                             importance = c(0))

    for(i in seq_len(nrow(importance))){

      new_err <-c()
      if(n_permute < ((nrow(model$data$data)**2)-1)){
        for(k in seq_len(n_permute)){

          perm_preds <- c()

          perm_data <- model$data$data
          perm_data[, importance$variable[i]] <- perm_data[sample.int(n = nrow(perm_data),replace = FALSE),importance$variable[i]]

          if(!softmax) perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device, ...)[,n_prediction,drop=FALSE])
          else perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device, ...))

          new_err <- append(new_err, as.numeric(loss(pred = torch::torch_tensor(perm_preds),
                                                     true = true_tmp)$mean() ))


        }
      }else{
        for(j in seq_len(nrow(model$data$data))){
          perm_data <- model$data$data[j,]
          for(k in seq_len(nrow(model$data$data))[-j]){
            perm_data[i] <- model$data$data[k,i]
            if(!softmax) perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device, ...)[,n_prediction,drop=FALSE])
            else perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device, ...))
            true <- append(true_tmp, model$data$Y[j])
          }
        }
      }

      importance$importance[i] <- mean(new_err)/org_err

    }
    colnames(importance)[2] = paste0("importance_", n_prediction)
    if(n_prediction > 1) importance = importance[,2,drop=FALSE]
    out[[n_prediction]] = importance

  }

  return(do.call(cbind, out))
}





ACE = function(data, predict_f, model, epsilon = 0.1, obs_level = FALSE,interactions=TRUE,max_indices = NULL, ...) {

  x0 = data
  if(is.null(max_indices)) n = 1:ncol(x0)
  else n = max_indices
  f = function(x0) predict_f(model, x0, ...)
  h = epsilon*apply(data[,max_indices,drop=FALSE], 2, stats::sd)
  H = array(NA, c(nrow(x0), length(n), length(n)))
  hh = diag(h, length(n))
  f_x0 = f(x0)
  N = nrow(x0)
  if(length(n) > 1) {
    for (i in 1:(length(n)-1)) {
      i_idx = n[i]
      hi <- hh[, i]
      hi = matrix(hi, N, length(n), byrow = TRUE )
      x0_tmp = x0
      x0_tmp[,n] = x0_tmp[,n] + hi
      H[,i, i] =  (f(x0_tmp) - f_x0 )/h[i]
      if(interactions) {
        for (j in (i + 1):length(n)) {
          j_idx = n[j]
          hj = hh[, j]
          hj = matrix(hj, N, length(n), byrow = TRUE )
          x0_tmp_pp = x0_tmp_pn = x0_tmp_np = x0_tmp_nn = x0
          x0_tmp_pp[,n] = x0_tmp_pp[,n] + hi + hj
          x0_tmp_pn[,n] = x0_tmp_pn[,n] + hi - hj
          x0_tmp_np[,n] = x0_tmp_np[,n] - hi + hj
          x0_tmp_nn[,n] = x0_tmp_nn[,n] - hi - hj
          H[,i, j] = (f(x0_tmp_pp) - f(x0_tmp_pn) - f(x0_tmp_np) + f(x0_tmp_nn))/(4 * h[i]^2)
          H[,j, i] = H[,i, j]
        }
      }
    }
  }

  hi = hh[, length(n)]
  hi = matrix(hi, N, length(n), byrow = TRUE )
  x0_tmp = x0
  x0_tmp[,n] = x0_tmp[,n] + hi
  H[, length(n), length(n)] <-  ( f(x0_tmp) - f_x0 )/h[length(n)]
  effs = apply(H, 2:3, mean)
  abs = apply(H, 2:3, function(d) mean(abs(d)))
  sds = apply(H, 2:3, stats::sd)
  if(!obs_level) return(list(effs = effs, abs = abs, sds = sds))
  else return(H)
}


# conditionalEffects = function(object, ...) UseMethod("conditionalEffects")

#' Calculate average conditional effects
#'
#' @description
#' Average conditional effects calculate the local derivatives for each observation for each feature. They are similar to marginal effects. And the average of these conditional effects is an approximation of linear effects (see Pichler and Hartig, 2023 for more details). You can use this function to either calculate main effects (on the diagonal, take a look at the example) or interaction effects (off-diagonals) between features.
#'
#' To obtain uncertainties for these effects, enable the bootstrapping option in the `dnn(..)` function (see example).
#'
#' @param object object of class \code{citodnn}
#' @param interactions calculate interactions or not (computationally expensive)
#' @param epsilon difference used to calculate derivatives
#' @param device which device
#' @param indices of variables for which the ACE are calculated
#' @param data data which is used to calculate the ACE
#' @param type ACE on which scale (response or link)
#' @param ... additional arguments that are passed to the predict function
#'
#' @return an S3 object of class \code{"conditionalEffects"} is returned.
#' The list consists of the following attributes:
#' \item{result}{3-dimensional array with the raw results}
#' \item{mean}{Matrix, average conditional effects}
#' \item{abs}{Matrix, summed absolute conditional effects}
#' \item{sd}{Matrix, standard deviation of the conditional effects}
#'
#' @example /inst/examples/conditionalEffects-example.R
#' @references
#' Scholbeck, C. A., Casalicchio, G., Molnar, C., Bischl, B., & Heumann, C. (2022). Marginal effects for non-linear prediction functions. arXiv preprint arXiv:2201.08837.
#'
#' Pichler, M., & Hartig, F. (2023). Can predictive models be used for causal inference?. arXiv preprint arXiv:2306.10551.
#' @author Maximilian Pichler
#' @export
conditionalEffects = function(object, interactions=FALSE, epsilon = 0.1, device = c("cpu", "cuda", "mps"), indices = NULL, data = NULL, type = "response",...) UseMethod("conditionalEffects")

#' @rdname conditionalEffects
#' @export
conditionalEffects.citodnn =  function(object, interactions=FALSE, epsilon = 0.1, device = c("cpu", "cuda", "mps"), indices = NULL, data = NULL, type = "response",...) {
  if(is.null(data)) {
    data = object$data$data
    resp = object$data$Y
  }
  device <- match.arg(device)

  object = check_model(object)
  Y_name = as.character( object$call$formula[[2]] )
  data = data[,-which( colnames(data) %in% Y_name), drop=FALSE]
  # var_names = c(Y_name, colnames(data))

  out = NULL

  if(is.null(indices)) {
    vars = get_var_names(object$training_properties$formula, object$data$data[1,])
    indices = which(colnames(data) %in% vars[!sapply(data[,vars], is.factor)], arr.ind = TRUE)
  }

  for(n_prediction in 1:object$model_properties$output) {
    result = ACE(
      data = data,
      predict_f = function(model, newdata) {
        df = data.frame(newdata)
        colnames(df) = colnames(data)
        return(stats::predict(model, df, device = device, type = type, ...)[,n_prediction])
      },
      model = object, obs_level = TRUE,
      interactions=interactions,
      epsilon = epsilon,
      max_indices = indices
    )
    tmp = list()
    tmp$result = result
    tmp$mean = apply(result, 2:3, mean)
    colnames(tmp$mean) = colnames(data)[indices]
    rownames(tmp$mean) = colnames(data)[indices]
    tmp$abs = apply(result, 2:3, function(d) sum(abs(d)))
    tmp$sd = apply(result, 2:3, function(d) stats::sd(d))
    tmp$interactions = interactions
    out[[n_prediction]] = tmp
  }
  class(out) = "conditionalEffects"
  return(out)
}


#' @rdname conditionalEffects
#' @export
conditionalEffects.citodnnBootstrap = function(object, interactions=FALSE, epsilon = 0.1, device = c("cpu", "cuda", "mps"), indices = NULL, data = NULL, type = "response",...) {

  pb = progress::progress_bar$new(total = length(object$models), format = "[:bar] :percent :eta", width = round(getOption("width")/2))
  results_boot = list()

  for(b in 1:length(object$models)) {
    model_indv = object$models[[b]]
    condEffs = conditionalEffects(model_indv, interactions=interactions, epsilon = 0.1, device = c("cpu", "cuda", "mps"), indices = indices, data = data, type = type,...)
    pb$tick()
    results_boot[[b]] = condEffs
  }
  out = list()

  out$mean = lapply(1:length(results_boot[[1]]), function(i) apply(abind::abind(lapply(results_boot, function(r) r[[i]]$mean), along = 0L), 2:3, mean))
  out$se = lapply(1:length(results_boot[[1]]), function(i) apply(abind::abind(lapply(results_boot, function(r) r[[i]]$mean), along = 0L), 2:3, stats::sd))
  out$abs = lapply(1:length(results_boot[[1]]), function(i) apply(abind::abind(lapply(results_boot, function(r) r[[i]]$abs), along = 0L), 2:3, mean))
  out$abs_se = lapply(1:length(results_boot[[1]]), function(i) apply(abind::abind(lapply(results_boot, function(r) r[[i]]$abs), along = 0L), 2:3, stats::sd))
  out$sd = lapply(1:length(results_boot[[1]]), function(i) apply(abind::abind(lapply(results_boot, function(r) r[[i]]$sd), along = 0L), 2:3, mean))
  out$sd_se = lapply(1:length(results_boot[[1]]), function(i) apply(abind::abind(lapply(results_boot, function(r) r[[i]]$sd), along = 0L), 2:3, stats::sd))
  class(out) = "conditionalEffectsBootstrap"
  return(out)
}



#' Print average conditional effects
#'
#' @param x print ACE calculated by \code{\link{conditionalEffects}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#'
#' @return Matrix with average conditional effects
#'
#' @export
print.conditionalEffects = function(x, ...) {
  ACE = sapply(x, function(x) diag(x$mean))
  print(ACE)
  return(invisible(ACE))
}

#' @rdname print.conditionalEffects
#' @export
print.conditionalEffectsBootstrap = function(x, ...) {
  ACE = sapply(x$mean, function(X) diag(X))
  print(ACE)
  return(invisible(ACE))
}


