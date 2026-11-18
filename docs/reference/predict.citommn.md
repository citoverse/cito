# Predict with a fitted MMN model

This function generates predictions from a Multi-Modal Neural Network
(MMN) model that was created using the [`mmn`](mmn.md) function.

## Usage

``` r
# S3 method for class 'citommn'
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

  a model created by [`mmn`](mmn.md)

- newdata:

  A list containing the new data for which predictions are to be made.
  The dimensions of the elements in `newdata` should match those of the
  training data, except for the respective first dimensions which
  represents the number of samples. If `NULL`, the function uses the
  data the model was trained on.

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

  return the embeddings (output of the last hidden layer of the fusion
  network) instead of the predictions. Default is `FALSE`.

- ...:

  Additional arguments (currently not used).

## Value

A matrix of predictions. If `type` is `"class"`, a factor of predicted
class labels is returned.
