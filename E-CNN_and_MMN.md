---
title: "Convultions neural networks and Multi modal neural networks"
author: "Maximilian Pichler"
date: "2025-05-27"
abstract: "This vignette demonstrates the use of CITO for complex data (CNN) and a mixture of complex and tabular data (MMN). CNN can be used for various tasks, such as image classification ('Is this image a dog or a cat?') and ecological tasks, such as multi-deep species distribution models (deepSDM), where the input could be an environmental time series or remote sensing data and the goal is to predict species occurrences. MMN extends CNN by combining complex data, such as satellite images, and tabular data, such as environmental or spatial information. Furthermore, MMN can combine complex data of different dimensions and scales (e.g. LiDAR 3D inputs and coloured 2D optical satellite images). This vignette explains how to prepare the data and suggests an example project structure."
output:
 rmarkdown::html_vignette:
    toc: true
    toc_depth: 4
    html_document:
      toc: true
      theme: cerulean
vignette: >
  %\VignetteIndexEntry{Convultions neural networks and Multi modal neural networks}
  %\VignetteEncoding{UTF-8}
  %\VignetteEngine{knitr::rmarkdown}
editor_options:
  chunk_output_type: console
---



## Data Preparation of complex data

In Cito, the workflow for preparing complex data for Convolutional Neural Networks (CNNs) and Multimodal Neural Networks (MMNs) is the same. The only difference is that MMNs can process multiple data types simultaneously (see the 'MMN' section below).

Before we dive into the details, let's clarify how we represent different types of complex data in R using multidimensional arrays:

- Black/white image: $[height, width]$ with height/width being the number of pixels
- Colored images: $[3, height, width]$ 3 channels for the three colors
- LiDAR point clouds: $[x, y, z]$
- Environmental time series (can be also represented by an "image"): $[\text{time_steps}, \text{n_covariates}]$

Note: Cito currently supports up to three-dimensional inputs (excluding the sample dimension). Four-dimensional arrays, such as temporal RGB sequences $[3, height, width, time_steps]$, are not yet supported. Please contact us if you need support for 4D inputs.

### The format of the inputs we expect in cito

The `cnn()` and `mmn()` functions both expect their X argument to be a single array, with the first dimension indexing samples. The subsequent dimensions then correspond to the data structure of each sample. Specifically:

- Grayscale images: $[\text{n_samples}, height, width]$
- RGB images: $[\text{n_samples}, 3, height, width]$
- LiDAR point clouds: $[\text{n_samples}, x, y, z]$
- Time series data: $[\text{n_samples}, time_steps, n_covariates]$

It is crucial that the order of samples in X matches the order of observations in your response (target) vector (or matrix for multiple response) y.

Ultimately, the only requirement for cito is that it has these multidimensional arrays as inputs. Please note that there are several ways in which we can build them; the following workflow is just one example.


### 1. Preparing your data on disk
Although images can be saved in different formats, we recommend using formats for which an R function or R package is available to allow you to load the images into R.

Grayscale and RGB images should be saved as `.png` or `.jpeg`. Time series data can technically be interpreted as grayscale images and should therefore also be saved as `.png` or `.jpeg`.

However, LiDAR point clouds and/or other remote sensing data have more 'channels' (ofc, they do not have channels at all) than grayscale or RGB images and cannot therefore be saved as `.png` or `.jpeg`. Classical formats for saving such data are `.tiff` (GeoTiff) and `.nc` (netCDF).

We recommend saving each image individually to your hard drive and using a naming strategy that allows the observation ID to be inferred from the image name. Here is an example:

```
project/
├── data/
│   ├── RGBimages/
│   │   ├── 001-img.jpeg
│   │   ├── 002-img.jpeg
│   │   ├── 003-img.jpeg
│   │   ├── 004-img.jpeg
│   │   └── ...
│   ├── LiDAR/
│   │   ├── 001-LiDAR.tiff
│   │   ├── 002-LiDAR.tiff
│   │   ├── 003-LiDAR.tiff
│   │   ├── 004-LiDAR.tiff
│   │   └── ...
│   └── Response/
│       └── Y.csv
└── code/
    ├── 01-MMN.R
    └── 02-CNN.R
```


### 2. Load images into R

Before we can run/train cito, we must load the images into R, transform them into arrays, and concatenate the individual images of one input type into one array.

- Reading `.jpeg` files: Can be done either using the `imager` package via `as.array(imager::load.image("path-to-img.jpeg")))` or using the `torchvision` package (dependency of cito) via `torchvision::base_loader("path-to-img.jpeg")`
- Reading `.png` files: Can be done using the `torchvision` package (dependency of cito) via `torchvision::base_loader("path-to-img.jpeg")`
- Reading `.tiff` files:  `tiff::readTIFF("path-to-img.tiff")`
- Reading `.tiff` (GeoTIFF) files: `as.array(raster::brick("path-to-img.tiff"))`














