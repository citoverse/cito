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

  width <- sample(round(size/2):size, n, replace = TRUE)
  height <- sample(round(size/2):size, n, replace = TRUE)

  center_y <- sample(floor(0.25*size):ceiling(0.75*size), n, replace = TRUE)
  center_x <- sample(floor(0.25*size):ceiling(0.75*size), n, replace = TRUE)

  for(i in 1:n) {
    if(stats::rbinom(1,1,p)) {
      data[i,1,,] <- create_rectangle_matrix(size, width[i], height[i], center_y[i], center_x[i])
      labels[i] <- "rectangle"
    } else {
      data[i,1,,] <- create_ellipsoid_matrix(size, width[i], height[i], center_y[i], center_x[i])
      labels[i] <- "ellipsoid"
    }
  }

  return(list(data=data, labels=as.factor(labels)))
}

create_rectangle_matrix <- function(size, rectangle_width, rectangle_height, center_y, center_x) {
  matrix <- matrix(0, nrow = size, ncol = size)
  left <- center_x - (rectangle_width %/% 2)
  top <- center_y - (rectangle_height %/% 2)
  right <- left + rectangle_width - 1
  bottom <- top + rectangle_height - 1
  left <- max(1, left)
  top <- max(1, top)
  right <- min(size, right)
  bottom <- min(size, bottom)
  matrix[top:bottom, left:right] <- 1
  return(matrix)
}

create_ellipsoid_matrix <- function(size, ellipsoid_width, ellipsoid_height, center_y, center_x) {
  matrix <- matrix(0, nrow = size, ncol = size)
  for (y in 1:size) {
    for (x in 1:size) {
      if (((x - center_x) / (ellipsoid_width / 2))^2 + ((y - center_y) / (ellipsoid_height / 2))^2 <= 1) {
        matrix[y, x] <- 1
      }
    }
  }
  return(matrix)
}
