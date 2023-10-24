#' Data Simulation for CNN
#'
#' @description
#'
#' generates images of rectangles and ellipsoids
#'
#' @param n number of images
#' @param size size of the (quadratic) images
#' @param p probability of generating a rectangle (1-p for ellipsoids)
#'
#' @details
#' This function generates simple data to demonstrate the usage of cnn().
#' The generated images are of centered rectangles and ellipsoids with random widths and heights.

#' @return array of dimension (n, 1, size, size)
#' @author Armin Schenk
#' @export
simulate_shapes <- function(n, size, p=0.5) {

  data <- array(0, dim = c(n,1,size,size))
  labels <- character(n)

  width <- sample(1:size, n, replace = TRUE)
  height <- sample(1:size, n, replace = TRUE)

  for(i in 1:n) {
    if(stats::rbinom(1,1,p)) {
      data[i,1,,] <- create_rectangle_matrix(size, width[i], height[i])
      labels[i] <- "rectangle"
    } else {
      data[i,1,,] <- create_ellipsoid_matrix(size, width[i], height[i])
      labels[i] <- "ellipsoid"
    }
  }

  return(list(data=data, labels=as.factor(labels)))
}

create_rectangle_matrix <- function(size, rectangle_width, rectangle_height) {
  matrix <- matrix(0, nrow = size, ncol = size)
  left <- (size - rectangle_width) %/% 2
  top <- (size - rectangle_height) %/% 2
  right <- left + rectangle_width
  bottom <- top + rectangle_height
  matrix[top:(bottom-1), left:(right-1)] <- 1
  return(matrix)
}

create_ellipsoid_matrix <- function(size, ellipsoid_width, ellipsoid_height) {
  matrix <- matrix(0, nrow = size, ncol = size)
  h <- size %/% 2
  k <- size %/% 2
  for (y in 1:size) {
    for (x in 1:size) {
      if (((x - h) / (ellipsoid_width / 2))^2 + ((y - k) / (ellipsoid_height / 2))^2 <= 1) {
        matrix[y, x] <- 1
      }
    }
  }
  return(matrix)
}
