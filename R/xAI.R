#' Partial Dependence Plot (PDP)
#'
#' Calculates the Partial Dependency Plot for one feature, either numeric or categorical. Returns it as a plot.
#'
#'
#' @param model a model created by \code{\link{dnn}}
#' @param variable variable (as a string) for which the PDP should be computed. If none is supplied, it is computed for all variables.
#' @param data new data on which the PDP should be computed. If NULL, the PDP is computed on the training data.
#' @param ice if TRUE, the Individual Conditional Expectation (ICE) curves are also shown
#' @param resolution.ice resolution (number of grid points) at which the ICE curves are computed
#' @param plot whether to plot the PDP
#' @param parallel whether to parallelize over the bootstrap models
#' @param ... arguments passed to \code{\link{predict}}
#'
#'
#' @details
#'
#' # Description
#' Performs a partial dependence plot (PDP) estimation to analyze the relationship between a selected feature and the model's predictions.
#'
#' The PDP function estimates the partial function \eqn{\hat{f}_S}{}:
#'
#' \eqn{\hat{f}_S(x_S)=\frac{1}{n}\sum_{i=1}^n\hat{f}(x_S,x^{(i)}_{C})}{}
#'
#' using a Monte Carlo estimation, i.e. it computes the average prediction over the data while the selected feature is held fixed at a given value and the remaining features are kept at their observed values.
#'
#' For categorical features, every observation is set to each level of the feature in turn, the average prediction per level is calculated, and the result is shown as a bar plot.
#'
#' If `ice = TRUE`, the Individual Conditional Expectation (ICE) curves are also shown, with the PDP highlighted in yellow. Each ICE curve illustrates how the prediction for a single observation changes as the feature varies. ICE curves are computed on a value grid rather than at every observed feature value, and are not available for categorical features.
#'
#' @return A list of plots made with 'ggplot2', one for each selected variable.
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
  if(!is.null(model$loss$responses)) {
    names(p_ret) = paste0(model$loss$responses, "_",names(p_ret))
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
  if(!is.null(model$loss$responses)) {
    names(p_ret) = paste0(model$loss$responses, "_",names(p_ret))
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

        if(!is.null(model$loss$responses)) {
          label = paste0("PDP - ", model$loss$responses[n_output])
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

        if(!is.null(model$loss$responses)) {
          label = paste0("PDP - ", model$loss$responses[n_output])
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
#' @param variable variable (as a string) for which the ALE should be computed. If none is supplied, it is computed for all variables.
#' @param data data on which the ALE is computed; if NULL, the training data is used
#' @param type ALE on which scale response or link, default is response
#' @param analytical Analytical ALE based on conditional effects or not
#' @param center center ALE or not (only available for analytical ALE)
#' @param K number of neighborhoods the original feature space is divided into
#' @param ALE_type method by which the feature bins (neighborhoods) are created
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
                analytical = FALSE,
                center = FALSE,
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
                        analytical = TRUE,
                        center = FALSE,
                        K = 10,
                        ALE_type = c("quantile", "equidistant"),
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
  if(analytical) {
    predictions = stats::predict(model, data, type = type,...)
  }
  p_ret <- sapply (variable,function(v){
    if(!analytical) results = getALE(model = model, v = v, ALE_type = ALE_type, data = data, K = K, type = type,...)
    else results = getALEce(model = model, v = v, data = data, type = type, predictions = predictions, center = center,...)

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

  if(!is.null(model$loss$responses)) {
    names(p_ret) = paste0(model$loss$responses, "_",names(p_ret))
  }
  return(invisible(p_ret))
}


#' @rdname ALE
#' @export
ALE.citodnnBootstrap <- function(model,
                                 variable = NULL,
                                 data = NULL,
                                 type = "response",
                                 analytical = TRUE,
                                 center = FALSE,
                                 K = 10,
                                 ALE_type = c("quantile", "equidistant"),
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
      if(analytical) {
        predictions = stats::predict(model, data, type = type,...)
      }
      p_ret <- sapply (variable,function(v){
        if(!analytical) results = getALE(model = model_indv, v = v, ALE_type = ALE_type,type = type, data = data, K = K, verbose = FALSE, ...)
        else results = getALEce(model = model_indv, v = v, type = type, data = data, verbose = FALSE, predictions = predictions,center = center,...)
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
      if(analytical) {
        predictions = stats::predict(model, data, type = type,...)
      }
      p_ret <- sapply (variable,function(v){
        if(!analytical) results = getALE(model = model_indv, v = v, ALE_type = ALE_type,type = type, data = data, K = K, verbose = FALSE, ...)
        else results = getALEce(model = model_indv, v = v, type = type, data = data, verbose = FALSE, predictions = predictions, center = center,...)
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

  if(!is.null(model$loss$responses)) {
    names(p_ret) = paste0(model$loss$responses, "_",names(p_ret))
  }
  return(invisible(p_ret))
}




# ALE_ce =function(X, ce, predictions = NULL, center = FALSE) {
#   stopifnot(nrow(X) == nrow(ce), ncol(X) == ncol(ce))
#   vars = colnames(X)
#
#   ales = lapply(seq_len(ncol(X)), function(j) {
#     xj = X[, j]
#     gj = ce[, j]
#     ux = sort(unique(xj))
#     if (length(ux) == 1L) {
#       return(data.frame(
#         x   = ux,
#         ale = 0,
#         var = if (!is.null(vars)) vars[j] else j
#       ))
#     }
#     g_mean = vapply(ux, function(val) {
#       idx = which(xj == val)
#       mean(gj[idx], na.rm = TRUE)
#     }, numeric(1))
#
#     dx = diff(ux)
#     g_mid = (g_mean[-1] + g_mean[-length(g_mean)]) / 2
#     increments = g_mid * dx
#
#     ale_vals = c(0, cumsum(increments)) #+ mean(predictions)
#     if(!center) ale_vals = ale_vals - mean(ale_vals) + mean(predictions)
#     else ale_vals = ale_vals - mean(ale_vals)
#     data.frame(
#       x   = ux,
#       ale = ale_vals,
#       var = if (!is.null(vars)) vars[j] else j
#     )
#   })
#   do.call(rbind, ales)
# }

ale_weights <- function(xj, ux, weighted) {
  if (isFALSE(weighted)) return(rep(1, length(ux)))     # unweighted -> equal
  if (weighted == "frequency") {
    w <- as.numeric(table(factor(xj, levels = ux)))      # degenerate for continuous
  } else { # "density": meaningful for continuous, down-weights isolated tails
    dens <- stats::density(xj)
    w    <- stats::approx(dens$x, dens$y, xout = ux, rule = 2)$y
    w[is.na(w) | w < 0] <- 0
  }
  if (sum(w) == 0) w <- rep(1, length(ux))
  w / sum(w)
}

ale_curve_from_ce <- function(xj, gj, weighted = "density", trim = 0) {

  keep <- rep(TRUE, length(xj))
  if (trim > 0) {
    qs   <- stats::quantile(xj, probs = c(trim, 1 - trim), na.rm = TRUE)
    keep <- xj >= qs[1] & xj <= qs[2]
    xj   <- xj[keep]
    gj   <- gj[keep]
  }

  ux <- sort(unique(xj))
  if (length(ux) == 1L) {
    return(list(ux = ux, ale_vals = 0, w = 1, keep = keep))
  }

  # mean local derivative within each unique-x bin
  g_mean   <- vapply(ux, function(val) mean(gj[xj == val], na.rm = TRUE), numeric(1))

  # trapezoidal accumulation -> uncentered ALE curve on grid ux
  dx       <- diff(ux)
  g_mid    <- (g_mean[-1] + g_mean[-length(g_mean)]) / 2
  ale_vals <- c(0, cumsum(g_mid * dx))

  w <- ale_weights(xj, ux, weighted)

  list(ux = ux, ale_vals = ale_vals, w = w, keep = keep)
}




# getALEce = function(model, ALE_type, data, ce = NULL, v, verbose = TRUE, type = "response", center = FALSE,predictions = NULL,...) {
#   if(is.null(ce)) {
#     ce = model$conditional_effects
#   }
#
#   row_indices = ce$row_indices
#
#   if(is.null(predictions)) predictions = stats::predict(model, data[row_indices,,drop=FALSE], type = type,...)
#   return(
#     lapply(1:model$model_properties$output, function(n_output) {
#       ce_tmp = ce[[type]][[n_output]]$result
#       v_idx = which(v == rownames(ce[[type]][[n_output]]$mean), arr.ind = TRUE)
#       xj = data[row_indices, v]
#       gj = ce_tmp[, v_idx, v_idx]
#       ux = sort(unique(xj))
#       if (length(ux) == 1L) {
#       df = data.frame(
#           x   = ux,
#           y = 0,
#         )
#       }
#       g_mean = vapply(ux, function(val) {
#         idx = which(xj == val)
#         mean(gj[idx], na.rm = TRUE)
#       }, numeric(1))
#
#       dx = diff(ux)
#       g_mid = (g_mean[-1] + g_mean[-length(g_mean)]) / 2
#       increments = g_mid * dx
#
#       ale_vals = c(0, cumsum(increments)) #+ mean(predictions)
#       if(!center) ale_vals = ale_vals - mean(ale_vals) + mean(predictions[,n_output,drop=FALSE])
#       else ale_vals = ale_vals - mean(ale_vals)
#       df = data.frame(
#         x   = ux,
#         y = ale_vals
#       )
#
#       #if(!is.null(model$data$ylvls)) {
#       label = paste0(v," \U2192 ", model$loss$responses[n_output])
#       # TODO model$data$responses[n_output]
#       #} else {
#       #  label = "ALE"
#       #}
#       return(list(df = df, label = label, data = data[,v], v = v))
#     }))
#
# }

getALEce = function(model, ALE_type, data, ce = NULL, v, verbose = TRUE,
                    type = "response", center = FALSE, predictions = NULL,
                    trim = 0, ...) {
  if (is.null(ce)) ce = model$conditional_effects

  row_indices = ce$row_indices

  if (is.null(predictions)) predictions = stats::predict(model, data[row_indices, , drop = FALSE], type = type, ...)

  return(
    lapply(1:model$model_properties$output, function(n_output) {
      ce_tmp = ce[[type]][[n_output]]$result
      v_idx  = which(v == rownames(ce[[type]][[n_output]]$mean), arr.ind = TRUE)
      xj     = data[row_indices, v]
      gj     = ce_tmp[, v_idx, v_idx]

      curve = ale_curve_from_ce(xj, gj, weighted = FALSE, trim = trim)
      ux       = curve$ux
      ale_vals = curve$ale_vals

      if (length(ux) == 1L) {
        df = data.frame(x = ux, y = 0)
      } else {
        # centering referenced to the SAME (trimmed) subset used for the curve
        if (!center) {
          pred_sub = predictions[curve$keep, n_output, drop = FALSE]
          ale_vals = ale_vals - mean(ale_vals) + mean(pred_sub)
        } else {
          ale_vals = ale_vals - mean(ale_vals)
        }
        df = data.frame(x = ux, y = ale_vals)
      }

      label = paste0(v, " \U2192 ", model$loss$responses[n_output])
      # NOTE: data returned for the rug is the trimmed feature, matching the curve
      return(list(df = df, label = label, data = xj, v = v))
    })
  )
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
        reduced_K <- FALSE
        repeat{
          quants <- stats::quantile(data[,v],probs = seq(0,1,1/K))
          groups <- lapply(c(2:(K+1)),function(i) return(which(data[,v] >= quants[i-1] & data[,v] < quants[i])))
          groups[[length(groups)]] <- c(groups[[length(groups)]],which.max(data[,v]))

          lens = sapply(groups, length)
          if(!any(lens == 0)) {

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
            if(reduced_K){
              if(verbose) message(paste0("Number of Neighborhoods reduced to ",K))
            }
            break

          } else {
            K <- K - 1
            reduced_K <- TRUE
          }
        }
      }
      for ( i in seq_len(nrow(df))[-1]){
        df$y[i]<- df$y[i-1]+df$y[i]
      }
      df$y <- df$y - mean(df$y)
      #if(!is.null(model$data$ylvls)) {
      label = paste0(v," \U2192 ", model$loss$responses[n_output])
      # TODO model$data$responses[n_output]
      #} else {
      #  label = "ALE"
      #}
      return(list(df = df, label = label, data = data[,v], v = v))
    }))
}



get_importance<- function(model,
                          n_permute= NULL,
                          data = NULL,
                          type = c("response", "link"),
                          importance = c("permutation", "ce", "ale"),
                          device = "cpu",
                          out_of_bag = FALSE, ...){

  type = match.arg(type)
  importance = match.arg(importance)
  model = check_model(model)
  softmax = FALSE
  loss = model$loss
  if(inherits(loss, "cross-entropy loss")) softmax = TRUE
  n_outputs = model$model_properties$output
  if(softmax) n_outputs = 1

  if(!model$conditional_effects$any && importance %in% c("ce", "ale")) {
    importance = "permutation"
  }

  if(importance == "permutation") {

  if(out_of_bag) {
    model$data$data = model$data$original$data[-model$data$indices,]
    model$data$X = model$data$original$X[-model$data$indices,]
    if(is.matrix(model$data$Y)) model$data$Y = model$data$original$Y[-model$data$indices,,drop=FALSE]
    if(is.vector(model$data$Y)) model$data$Y = model$data$original$Y[-model$data$indices]
  }

  if(is.null(n_permute)) n_permute <- ceiling(sqrt(nrow(model$data$data))*3)




  # if(!inherits(loss, c("cross-entropy loss", "mean squared error loss", "mean absolute error loss"))) {
  #   return(NULL)
  # }


    true = model$data$Y

    if(softmax) true = torch::torch_tensor(true, dtype = torch::torch_long())$squeeze(2)
    else true = torch::torch_tensor(true, dtype = torch::torch_float32())


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

  } else if(importance == "ale") {

    return(get_importance_ale_var(model, data = data, type = type, weighted = "density"))

  } else {
    ce = model$conditional_effects[[type]]
    col_names = colnames(ce[[1]]$mean)
    ce = abind::abind(lapply(1:length(ce), function(i) ce[[i]]$result), along = -1)
    ce = ce**2
    if(n_outputs == 1) {
      ce = apply(ce, 2:4, sum)
      ce = apply(ce, 1, diag)
      if(is.vector(ce)) ce = matrix(ce, nrow = 1L)
      ce = t(ce)
      ce = colMeans(ce)
      imps = ce
      imps = imps*model$conditional_effects$vars
      out = data.frame(variable = col_names, importance_1 = imps)
    } else {
      out = data.frame(variable = col_names)
      for(i in 1:n_outputs) {
        ce_sub = ce[i,,,,drop=FALSE]
        ce_sub = abind::adrop(ce_sub, drop = 1)
        ce_sub = diag(apply(ce_sub, 2:3, mean))
        if(is.vector(ce_sub)) ce_sub = matrix(ce_sub, nrow = 1L)
        imps = t(ce_sub)
        imps = imps*model$conditional_effects$vars
        out[[paste0("importance_", i)]] = imps
      }
    }
    #imps = imps / sum(imps)
    return(out)
  }
}

get_importance_ale_var <- function(model,
                                   data = NULL,
                                   type = "response",
                                   weighted = "density",
                                   trim = 0) {

  if (is.null(model$conditional_effects) ||
      isFALSE(model$conditional_effects$any)) {
    stop("No conditional effects stored on the model; fit with conditional effects enabled.")
  }
  if (!isFALSE(weighted)) weighted <- match.arg(weighted, c("density", "frequency"))

  if (is.null(data)) data <- model$data$data

  ce       <- model$conditional_effects
  ce_scale <- ce[[type]]
  if (is.null(ce_scale)) stop(sprintf("No conditional effects stored for type = '%s'.", type))

  row_indices <- ce$row_indices
  n_output    <- model$model_properties$output
  var_names   <- rownames(ce_scale[[1]]$mean)

  out <- data.frame(variable = var_names, stringsAsFactors = FALSE)

  for (n in seq_len(n_output)) {

    ce_tmp <- ce_scale[[n]]$result          # obs x var x var

    imps <- vapply(var_names, function(v) {

      v_idx <- which(v == rownames(ce_scale[[n]]$mean))
      xj    <- data[row_indices, v]
      gj    <- ce_tmp[, v_idx, v_idx]        # per-obs df/dx_v (diagonal)

      curve <- ale_curve_from_ce(xj, gj, weighted = weighted, trim = trim)
      if (length(curve$ux) == 1L) return(0)

      # distribution-weighted variance of the (uncentered) curve;
      # centering is irrelevant: Var(ale - c) = Var(ale)
      w  <- curve$w
      mu <- sum(w * curve$ale_vals)
      sum(w * (curve$ale_vals - mu)^2)

    }, numeric(1))

    out[[paste0("importance_", n)]] <- as.numeric(imps)
  }

  rownames(out) <- NULL
  out
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
#' @param method Calculate the conditional effects analytically or via the finite difference
#' @param subsample subsample data to decrease computational runtime, must be either FALSE or in the range of `[0,1]`
#' @param interactions calculate interactions or not (computationally expensive)
#' @param epsilon difference used to calculate derivatives
#' @param device which device
#' @param indices of variables for which the ACE are calculated
#' @param data data which is used to calculate the ACE
#' @param batchsize batchsize
#' @param type ACE on which scale (response or link)
#' @param return_vars return variances of variables (internally required)
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
conditionalEffects.citodnn =  function(object,
                                       method = c("analytical", "finite"),
                                       subsample = FALSE,
                                       interactions=FALSE,
                                       epsilon = 0.1,
                                       device = c("cpu", "cuda", "mps"),
                                       indices = NULL,
                                       data = NULL,
                                       batchsize = NULL,
                                       type = "response",
                                       return_vars = FALSE,
                                       ...) {

  method = match.arg(method)

  if(method == "finite") {

    if(is.logical(subsample)) {
      indices_row = 1:nrow(object$data$X)
    } else {
      indices_row = sample.int(nrow(object$data$X), ceiling(subsample*nrow(object$data$X)))
    }

    if(is.null(data)) {
      data = object$data$data
      resp = object$data$Y
    }
    device <- match.arg(device)

    object = check_model(object)
    Y_name = as.character( object$call$formula[[2]] )
    data = data[indices_row,-which( colnames(data) %in% Y_name), drop=FALSE]
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

  } else {
   res =
    ACEanalytical(object,
                  interactions=interactions,
                  subsample = subsample,
                  device = device,
                  indices = indices,
                  data = data,
                  batchsize = batchsize,
                  type = type,
                  return_vars)
   out = list()
   if(type == "response") {
     out = res[[2]]
   } else {
     out = res[[1]]
   }

  }
  if(return_vars) out$vars = res$vars

  class(out) = "conditionalEffects"
  return(out)
}


#' @rdname conditionalEffects
#' @export
conditionalEffects.citodnnBootstrap = function(object, interactions=FALSE, epsilon = 0.1, device = c("cpu", "cuda", "mps"), indices = NULL, data = NULL, type = "response",...) {

  if(is.null(object$conditional_effects)) {

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
  } else {

  }
  return(out)
}

# interactions=FALSE
# subsample = FALSE
# device = "cpu"
# indices = NULL
# data = NULL
# batchsize = NULL
# type = "response"
# return_vars = FALSE

ACEanalytical = function(object,
                         interactions=FALSE,
                         subsample = FALSE,
                         device = c("cpu", "cuda", "mps"),
                         indices = NULL,
                         data = NULL,
                         batchsize = NULL,
                         type = "response",
                         return_vars = FALSE) {

  device = match.arg(device)
  object <- check_model(object)
  device <- check_device(device)
  object$net$to(device = device)
  object$loss$to(device = device)
  object$net$eval()

  Z = NULL

  if(is.null(data)){

    if(is.logical(subsample)) {
      indices_row = 1:nrow(object$data$X)
    } else {
      indices_row = sample.int(nrow(object$data$X), ceiling(subsample*nrow(object$data$X)))
    }

    sample_names <- rownames(object$data$X)[indices_row]
    X = torch::torch_tensor(object$data$X)[indices_row]
    if(!is.null(object$model_properties$embeddings)) {
      Z = torch::torch_tensor(object$data$Z, dtype = torch::torch_long())[indices_row]
    }
  } else {

    if(is.logical(subsample)) {
      indices_row = 1:nrow(data)
    } else {
      indices_row = sample.int(nrow(data), ceiling(subsample*nrow(object$data$X)))
    }

    sample_names <- rownames(data)
    if(is.data.frame(data)) X <- torch::torch_tensor(stats::model.matrix(stats::as.formula(stats::delete.response(object$call$formula)), data, xlev = object$data$xlvls)[,-1,drop=FALSE])[indices_row]
    else X <- torch::torch_tensor(stats::model.matrix(stats::as.formula(stats::delete.response(object$call$formula)), data.frame(data), xlev = object$data$xlvls)[,-1,drop=FALSE])[indices_row]

    if(!is.null(object$model_properties$embeddings)) {
      tmp = do.call(cbind, lapply(object$Z_formula, function(term) {
        if(!is.factor(data[[term]])) stop(paste0(term, " must be factor."))
        #TODO: check if newdata[[term]] has the same factor levels as object$data[[term]]
        data[[term]] |> as.integer()
      }) )
      Z = torch::torch_tensor(tmp, dtype = torch::torch_long())[indices_row]
    }
  }

  if(is.null(indices)) {
    if(!length(object$data$xlvls) == 0) {
      cat_vars_lvls = lapply(1:length(object$data$xlvls), function(i) sapply(object$data$xlvls[[i]], function(j) paste0(names(object$data$xlvls)[i],j, collapse = "")) ) |> unlist()
      cont_indices = which(!colnames(object$data$X) %in% cat_vars_lvls, arr.ind = TRUE)
      cat_indices = which(colnames(object$data$X) %in% cat_vars_lvls, arr.ind = TRUE)
    } else {
      cont_indices = 1:ncol(object$data$X)
      cat_indices = NULL
    }
  } else {
    tmp_indices = 1:ncol(object$data$X)
    cont_indices = indices
    cat_indices = which(!tmp_indices %in% indices, arr.ind = TRUE)
    if(length(cat_indices) == 0) {
      cat_indices = NULL
    }
  }

  if(length(cont_indices) == 0) {
    return(list(
      link = list(),
      response = list(),
      vars = 0,
      row_indices = 1,
      any = FALSE
    ))
  }

  if(is.null(batchsize)) batchsize = object$training_properties$batchsize
  #Y_torch <- object$loss$format_Y(object$data$Y)
  if(is.null(Z)) {
    dl <- get_data_loader(X, batch_size = batchsize, shuffle = FALSE)
  } else {
    dl <- get_data_loader(X, Z, batch_size = batchsize, shuffle = FALSE)
  }

  if(return_vars) {
    var_vars = as.numeric(X[,cont_indices]$var(1))
  }

  grads = list()
  grads_response = list()

  if(inherits(object$loss, c("mean squared error loss","gaussian loss","mean absolute error loss" ))) {
    skip_response = TRUE
  } else {
    skip_response = FALSE
  }
  coro::loop(for(b in dl) {

    Xb = b[[1]]$to(device = device, non_blocking = TRUE)
    #Yb = b[[2]]$to(device = device, non_blocking = TRUE)
    if(!is.null(Z)) { Zb = b[[2]]$to(device = device, non_blocking = TRUE) }

    if(!is.null(cat_indices)) {
      Xconst = torch::torch_zeros_like(Xb)
      Xconst[,cat_indices] =Xconst[,cat_indices] + Xb[,cat_indices]
      Xgrad = torch::torch_tensor(Xb[, cont_indices,drop=FALSE], requires_grad = TRUE, device = Xb$device, dtype = Xb$dtype)
      if(length(cont_indices) > 1) {Xconst[,cont_indices] = Xconst[,cont_indices] + Xgrad
      } else{ Xconst[,cont_indices] = Xconst[,cont_indices] + Xgrad[,1] }

    } else {
      Xconst = torch::torch_zeros_like(Xb)
      Xgrad = torch::torch_tensor(Xb[, cont_indices,drop=FALSE], requires_grad = TRUE, device = Xb$device, dtype = Xb$dtype)
      if(length(cont_indices) > 1) {Xconst[,cont_indices] = Xconst[,cont_indices] + Xgrad
      } else{ Xconst[,cont_indices] = Xconst[,cont_indices] + Xgrad[,1] }
    }

    pred =
      if(is.null(Z)) {
        object$net$forward(Xconst)
      } else {
        object$net$forward(Xconst, Zb)
      }
    # response
    pred_response = object$loss$invlink(pred)

    tmp_results =
      lapply(1:object$model_properties$output, function(R) {
        grads_response_inter = grads_inter = NULL

        grads_b = torch::autograd_grad(pred[,R,drop=F], Xgrad, grad_outputs = torch::torch_ones_like(Xgrad)/Xgrad$shape[2], retain_graph = TRUE, create_graph = TRUE)[[1]]
        if(skip_response) {
          grads_response_b = grads_b
        } else {
          grads_response_b = torch::autograd_grad(pred_response[,R,drop=F], Xgrad, grad_outputs = torch::torch_ones_like(Xgrad)/Xgrad$shape[2], retain_graph = TRUE, create_graph = TRUE)[[1]]
        }

        if(interactions) {

          if(type == "link") {
            grads_inter = lapply(1:nrow(grads_b), function(j) sapply(1:length(cont_indices), function(i) as.matrix(torch::autograd_grad(grads_b[j,i], Xgrad, retain_graph = TRUE)[[1]])[j,]))
            grads_inter = abind::abind(grads_inter, along = -1L)/2.0
            #grads_interactions = abind::abind(grads_interactions, grads_response_inter, along = 1)
          }

          if(type == "response") {
            if(skip_response & !is.null(grads_inter)) {
              grads_response_inter = grads_inter
            } else {
              grads_response_inter = lapply(1:nrow(grads_response_b), function(j) sapply(1:length(cont_indices), function(i) as.matrix(torch::autograd_grad(grads_response_b[j,i], Xgrad, retain_graph = TRUE)[[1]])[j,]))
              grads_response_inter = abind::abind(grads_response_inter, along = -1L)/2.0
              #grads_interactions_response = abind::abind(grads_interactions_response, grads_response_inter, along = 1)
            }
          }
        }

        return(list(as.matrix(grads_b), as.matrix(grads_response_b),grads_inter, grads_response_inter))
      })

    grads_b = lapply(1:object$model_properties$output, function(R) {

      grads_b_diag = lapply(1:nrow(tmp_results[[R]][[1]]), function(i) {

        if(interactions && (type == "link")) d = tmp_results[[R]][[3]][i,,]
        else d = matrix(NA, length(cont_indices), length(cont_indices))
        diag(d) = tmp_results[[R]][[1]][i,]
        d
      })
      return(abind::abind(grads_b_diag, along = -1))
    })

    grads_response_b = lapply(1:object$model_properties$output, function(R) {

      grads_response_b_diag = lapply(1:nrow(tmp_results[[R]][[2]]), function(i) {
        if(interactions && (type == "response")) d = tmp_results[[R]][[4]][i,,]
        else d = matrix(NA, length(cont_indices), length(cont_indices))
        diag(d) = tmp_results[[R]][[2]][i,]
        d
      })
      return(abind::abind(grads_response_b_diag, along = -1))
    })
    grads = append(grads, abind::abind(grads_b, along = -1) |> list())
    grads_response = append(grads_response, abind::abind(grads_response_b, along = -1) |> list())

  })

  ce = abind::abind(grads, along = 2L) |> asplit(MARGIN = 1L)
  ce_response = abind::abind(grads_response, along = 2L)|> asplit(MARGIN = 1L)

  out = list()
  for(n_prediction in 1:object$model_properties$output) {
    result = ce[[n_prediction]]
    tmp = list()
    tmp$result = result
    tmp$mean = apply(result, 2:3, mean)
    colnames(tmp$mean) = colnames(object$data$X)[cont_indices]
    rownames(tmp$mean) = colnames(object$data$X)[cont_indices]
    tmp$abs = apply(result, 2:3, function(d) sum(abs(d)))
    tmp$sd = apply(result, 2:3, function(d) stats::sd(d))
    tmp$interactions = interactions
    out[[n_prediction]] = tmp
  }

  out_response = list()
  for(n_prediction in 1:object$model_properties$output) {
    result = ce_response[[n_prediction]]
    tmp = list()
    tmp$result = result
    tmp$mean = apply(result, 2:3, mean)
    colnames(tmp$mean) = colnames(object$data$X)[cont_indices]
    rownames(tmp$mean) = colnames(object$data$X)[cont_indices]
    tmp$abs = apply(result, 2:3, function(d) sum(abs(d)))
    tmp$sd = apply(result, 2:3, function(d) stats::sd(d))
    tmp$interactions = interactions
    out_response[[n_prediction]] = tmp
  }

  out = list(out, out_response)

  if(return_vars) {
    out$vars = var_vars
  }

  out$row_indices = indices_row
  out$any = TRUE

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


