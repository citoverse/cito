# Data Simulation for CNN

generates images of rectangles and ellipsoids

## Usage

``` r
simulate_shapes(n, size, channels = 1)
```

## Arguments

- n:

  number of images

- size:

  size of the (quadratic) images

- channels:

  number of channels the generated data has (in each channel a new
  rectangle/ellipsoid is created)

## Value

array of dimension (n, 1, size, size)

## Details

This function generates simple data to demonstrate the usage of cnn().
The generated images are of centered rectangles and ellipsoids with
random widths and heights.

## Author

Armin Schenk
