#' Partial Dependence Plot (PDP)
#'
#' Calculates the Partial Dependency Plot for one feature, either numeric or categorical. Returns it as a plot.
#'
#' @details
#'
#' Description
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
#' @param model a model created by \code{\link{dnn}}
#' @param variable variable as string for which the PDP should be done. If none is supplied it is done for all variables.
#' @param data specify new data PDP should be performed . If NULL, PDP is performed on the training data.
#' @param ice Individual Conditional Dependence will be shown if TRUE
#' @param resolution.ice resolution in which ice will be computed
#' @param plot plot PDP or not
#' @param parallel parallelize over bootstrap models or not
#' @param ... arguments passed to \code{\link{predict}}
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

  perm_data <- stats::model.matrix(model$training_properties$formula, data)

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
      p <- p + ggplot2::ylab(label = as.character(model$call$formula[2]))
      p <- p + ggplot2::geom_rug(sides = "b")
      if(ice) {
        p <- p + ggplot2::geom_line(data = res$df_ice, mapping = ggplot2::aes(x = x, y = y, group = group ))
        p <- p + ggplot2::geom_line(colour = "yellow", linewidth = 2, data=res$df, mapping = ggplot2::aes(x=x,y=y))
      }
    } else if (is.factor(data[,res$v])){
      p <- ggplot2::ggplot(data = res$df,mapping = ggplot2::aes(x = x,y = y),)
      p <- p + ggplot2::geom_bar(stat= "identity")
      p <- p + ggplot2::theme_minimal()
      p <- p + ggplot2::geom_text(ggplot2::aes(label=y), vjust=1.6)
      p <- p + ggplot2::ggtitle(label = res$label)
      p <- p + ggplot2::xlab(label = res$v)
      p <- p + ggplot2::ylab(label = as.character(model$call$formula[2]))
      p <- p + ggplot2::xlab(res$v) + ggplot2::ylab(model$call$formula[2])
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

  perm_data <- stats::model.matrix(model$models[[1]]$training_properties$formula, data)

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
  return(
    lapply(1:model$model_properties$output, function(n_output) {
      df_ice = NULL
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
