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
#' @references Pichler, M., & Hartig, F. (2023). Can predictive models be used for causal inference?. arXiv preprint arXiv:2306.10551.
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
  data = data[,-which( colnames(data) %in% Y_name)]
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
    condEffs = conditionalEffects(model_indv, interactions=FALSE, epsilon = 0.1, device = c("cpu", "cuda", "mps"), indices = NULL, data = NULL, type = "response",...)
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


