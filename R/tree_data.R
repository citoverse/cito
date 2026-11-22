#' Tree Data
#'
#' A simulated data set of 500 sites for trying out multimodal networks (\code{\link{mmn}}):
#' a binary image per site, three standardized environmental predictors, and three responses.
#'
#' @format ## `tree_data`
#' List with the following objects (500 observations)
#' \describe{
#'   \item{images}{Satellite images}
#'   \item{Env}{data.frame, Environment (tabulardata)}
#'   \item{Species}{Species 0 or Species 1}
#'   \item{Trees}{Number of Trees}
#'   \item{dbh}{DBH in cm}
#' }
"tree_data"
