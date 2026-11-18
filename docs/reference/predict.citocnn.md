# Predict with a fitted CNN model

This function generates predictions from a Convolutional Neural Network
(CNN) model that was created using the [`cnn`](cnn.md) function.

## Usage

``` r
# S3 method for class 'citocnn'
predict(
  object,
  newdata = NULL,
  type = c("link", "response", "class"),
  device = NULL,
  batchsize = NULL,
  return_embeddings = FALSE,
  ...
)
```

## Arguments

- object:

  a model created by [`cnn`](cnn.md).

- newdata:

  A multidimensional array representing the new data for which
  predictions are to be made. The dimensions of `newdata` should match
  those of the training data, except for the first dimension which
  represents the number of samples. As an alternative, you can provide
  the relative or absolute path to the folder containing the images. In
  this case, unsigned integer images are divided by their maximum value
  (255 for 8 bit, 65535 for 16 bit), and floating point and signed
  integer TIFFs are passed unchanged. Windows of a raster are given with
  [`raster_patches`](raster_patches.md) (use its `like` argument, so
  that they carry the preprocessing of the training windows); they are
  not rescaled. If `NULL`, the function uses the data the model was
  trained on.

- type:

  A character string specifying the type of prediction to be made.
  Options are:

  - `"link"`: Scale of the linear predictor.

  - `"response"`: Scale of the response.

  - `"class"`: The predicted class labels (for classification tasks).

- device:

  Device to be used for making predictions. Options are "cpu", "cuda",
  and "mps". If `NULL`, the function uses the same device that was used
  when training the model. Default is `NULL`.

- batchsize:

  An integer specifying the number of samples to be processed at the
  same time. If `NULL`, the function uses the same batchsize that was
  used when training the model. Default is `NULL`.

- return_embeddings:

  Return embeddings instead of predictions

- ...:

  Additional arguments (currently not used).

## Value

A matrix of predictions. If `type` is `"class"`, a factor of predicted
class labels is returned.

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)

device <- ifelse(torch::cuda_is_available(), "cuda", "cpu")

## Data
shapes <- cito:::simulate_shapes(320, 28)
X <- shapes$data
Y <- shapes$labels

## Architecture
architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))

## Build and train network
cnn.fit <- cnn(X, Y, architecture, loss = "cross-entropy", epochs = 50,
               validation = 0.1, lr = 0.05, device = device)

## Get predictions of the validation set
valid <- cnn.fit$data$validation
predictions <- predict(cnn.fit, newdata = X[valid,,,,drop=FALSE], type="class")

## Classification accuracy
accuracy <- sum(predictions == Y[valid])/length(valid)

}
#> Loss at epoch 1: training: 0.696, validation: 0.700, lr: 0.05000

#> Loss at epoch 2: training: 0.694, validation: 0.697, lr: 0.05000
#> Loss at epoch 3: training: 0.693, validation: 0.695, lr: 0.05000
#> Loss at epoch 4: training: 0.692, validation: 0.693, lr: 0.05000
#> Loss at epoch 5: training: 0.691, validation: 0.692, lr: 0.05000
#> Loss at epoch 6: training: 0.690, validation: 0.692, lr: 0.05000
#> Loss at epoch 7: training: 0.690, validation: 0.692, lr: 0.05000
#> Loss at epoch 8: training: 0.689, validation: 0.690, lr: 0.05000
#> Loss at epoch 9: training: 0.689, validation: 0.689, lr: 0.05000
#> Loss at epoch 10: training: 0.688, validation: 0.686, lr: 0.05000
#> Loss at epoch 11: training: 0.688, validation: 0.687, lr: 0.05000
#> Loss at epoch 12: training: 0.686, validation: 0.686, lr: 0.05000
#> Loss at epoch 13: training: 0.685, validation: 0.684, lr: 0.05000
#> Loss at epoch 14: training: 0.685, validation: 0.680, lr: 0.05000
#> Loss at epoch 15: training: 0.683, validation: 0.680, lr: 0.05000
#> Loss at epoch 16: training: 0.681, validation: 0.678, lr: 0.05000
#> Loss at epoch 17: training: 0.681, validation: 0.676, lr: 0.05000
#> Loss at epoch 18: training: 0.679, validation: 0.672, lr: 0.05000
#> Loss at epoch 19: training: 0.678, validation: 0.673, lr: 0.05000
#> Loss at epoch 20: training: 0.676, validation: 0.668, lr: 0.05000
#> Loss at epoch 21: training: 0.673, validation: 0.662, lr: 0.05000
#> Loss at epoch 22: training: 0.671, validation: 0.665, lr: 0.05000
#> Loss at epoch 23: training: 0.669, validation: 0.660, lr: 0.05000
#> Loss at epoch 24: training: 0.678, validation: 0.658, lr: 0.05000
#> Loss at epoch 25: training: 0.669, validation: 0.655, lr: 0.05000
#> Loss at epoch 26: training: 0.661, validation: 0.664, lr: 0.05000
#> Loss at epoch 27: training: 0.664, validation: 0.662, lr: 0.05000
#> Loss at epoch 28: training: 0.667, validation: 0.651, lr: 0.05000
#> Loss at epoch 29: training: 0.669, validation: 0.650, lr: 0.05000
#> Loss at epoch 30: training: 0.657, validation: 0.647, lr: 0.05000
#> Loss at epoch 31: training: 0.656, validation: 0.640, lr: 0.05000
#> Loss at epoch 32: training: 0.658, validation: 0.641, lr: 0.05000
#> Loss at epoch 33: training: 0.661, validation: 0.645, lr: 0.05000
#> Loss at epoch 34: training: 0.648, validation: 0.637, lr: 0.05000
#> Loss at epoch 35: training: 0.656, validation: 0.646, lr: 0.05000
#> Loss at epoch 36: training: 0.657, validation: 0.637, lr: 0.05000
#> Loss at epoch 37: training: 0.648, validation: 0.645, lr: 0.05000
#> Loss at epoch 38: training: 0.646, validation: 0.647, lr: 0.05000
#> Loss at epoch 39: training: 0.648, validation: 0.632, lr: 0.05000
#> Loss at epoch 40: training: 0.645, validation: 0.646, lr: 0.05000
#> Loss at epoch 41: training: 0.642, validation: 0.628, lr: 0.05000
#> Loss at epoch 42: training: 0.638, validation: 0.646, lr: 0.05000
#> Loss at epoch 43: training: 0.632, validation: 0.623, lr: 0.05000
#> Loss at epoch 44: training: 0.633, validation: 0.644, lr: 0.05000
#> Loss at epoch 45: training: 0.635, validation: 0.651, lr: 0.05000
#> Loss at epoch 46: training: 0.656, validation: 0.629, lr: 0.05000
#> Loss at epoch 47: training: 0.639, validation: 0.623, lr: 0.05000
#> Loss at epoch 48: training: 0.653, validation: 0.625, lr: 0.05000
#> Loss at epoch 49: training: 0.629, validation: 0.662, lr: 0.05000
#> Loss at epoch 50: training: 0.617, validation: 0.646, lr: 0.05000
# }
```
