ACE = function(data, predict_f, model, epsilon = 0.1, obs_level = FALSE,interactions=TRUE,max_indices = NULL, ...) {

  x0 = data
  if(is.null(max_indices)) n = ncol(x0)
  else n = max_indices
  f = function(x0) predict_f(model, x0, ...)
  h = epsilon*apply(data, 2, sd)
  H = array(NA, c(nrow(x0), n, n))
  hh = diag(h, ncol(x0))
  f_x0 = f(x0)
  N = nrow(x0)
  for (i in 1:(n - 1)) {
    hi <- hh[, i]
    hi = matrix(hi, N, ncol(x0), byrow = TRUE )
    H[,i, i] =  (f(x0 + hi) - f_x0 )/h[i]
    if(interactions) {
      for (j in (i + 1):n) {
        hj = hh[, j]
        hj = matrix(hj, N, ncol(x0), byrow = TRUE )
        H[,i, j] = (f(x0 + hi + hj) - f(x0 + hi - hj) - f(x0 - hi + hj) + f(x0 - hi - hj))/(4 * h[i]^2)
        H[,j, i] = H[,i, j]
      }
    }
  }
  hi = hh[, n]
  hi = matrix(hi, N, ncol(x0), byrow = TRUE )
  H[, n, n] <-  ( f(x0 + hi) - f_x0 )/h[n]
  effs = apply(H, 2:3, mean)
  abs = apply(H, 2:3, function(d) mean(abs(d)))
  sds = apply(H, 2:3, sd)
  if(!obs_level) return(list(effs = effs, abs = abs, sds = sds))
  else return(H)
}


# conditionalEffects = function(object, ...) UseMethod("conditionalEffects")

#' Calculate average conditional effects
#'
#' @param object object of class nn.fit
#' @param interactions calculate interactions or not (computationally expensive)
#' @param epsilon difference used to calculate derivatives
#' @param device which device
#' @param indices calculation of effects until which column index
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
#' @author Maximilian Pichler
#' @export
conditionalEffects = function(object, interactions=FALSE, epsilon = 0.1, device = c("cpu", "cuda", "mps"), indices = NULL, data = NULL, type = "response",...) {
  if(is.null(data)) {
    data = object$data$data
    resp = object$data$Y
  }
  device <- match.arg(device)

  object = check_model(object)
  Y_name = as.character( object$call$formula[[2]] )
  data = data[,-which( colnames(data) %in% Y_name)]
  var_names = c(Y_name, colnames(data))

  out = NULL
  for(n in 1:object$model_properties$output) {
    result = ACE(
      data = data,
      predict_f = function(model, newdata) {
        df = data.frame(Y_name = 0, newdata)
        colnames(df) = var_names
        return(predict(model, df, device = device, type = type, ...)[,n])
      },
      model = object, obs_level = TRUE,
      interactions=interactions,
      epsilon = epsilon,
      max_indices = indices
    )
    tmp = list()
    tmp$result = result
    tmp$mean = apply(result, 2:3, mean)
    colnames(tmp$mean) = colnames(data)[1:ncol(tmp$mean)]
    rownames(tmp$mean) = colnames(data)[1:ncol(tmp$mean)]
    tmp$abs = apply(result, 2:3, function(d) sum(abs(d)))
    tmp$sd = apply(result, 2:3, function(d) sd(d))
    out[[n]] = tmp
  }
  class(out) = "conditionalEffects"
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



#' Hello World
#'
#' @export
hello_world = function() {
  print("Hi..")
}
