#' CNN
#' @export
cnn <- function(X,
                Y,
                layers = NULL, #Default einfuegen
                activation = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                               "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                               "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"),
                normalization = FALSE,
                lambda = 0.0,
                alpha = 0.5,
                dropout = 0.0,
                bias = TRUE,
                n_kernels = 10,
                kernel_size = list(conv = 3, pool = 2),
                stride = list(conv = 1, pool = kernel_size$'pool'),
                padding = 0,
                dilation = 1,
                loss = c("mse", "mae", "softmax", "cross-entropy", "gaussian", "binomial", "poisson"),
                optimizer = c("sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop"),
                lr = 0.01,
                validation = 0.0,
                batchsize = 32L,
                shuffle = TRUE,
                epochs = 100,
                early_stopping = NULL,
                lr_scheduler = NULL,
                custom_parameters = NULL,
                device = c("cpu", "cuda"),
                plot = TRUE,
                verbose = TRUE) {

  if(identical(activation, c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                              "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                              "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"))) activation <- "relu"

  #Data
  checkmate::assert(checkmate::checkArray(X, min.d = 3, max.d = 5))
  checkmate::assert(checkmate::checkFactor(Y), checkmate::checkNumeric(Y),
                    checkmate::checkMatrix(Y, mode = "numeric"), checkmate::checkMatrix(Y, mode = "logical"))

  #Architecture
  check_listable_parameter(activation, "S1[1,)")
  check_listable_parameter(normalization, "B1")
  check_listable_parameter(lambda, "R1[0,)")
  check_listable_parameter(alpha, "r1[0,1]")
  check_listable_parameter(dropout, "R1[0,1)")
  check_listable_parameter(bias, "B1")
  checkmate::qassert(n_kernels, "X1")
  check_listable_parameter(kernel_size, "X<=3[1,)")
  check_listable_parameter(stride, "X<=3[1,)")
  check_listable_parameter(padding, "X<=3[0,)")
  check_listable_parameter(dilation, "X<=3[1,)")

  #Training
  checkmate::qassert(lr, "R+[0,)")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(batchsize, "X1[1,)")
  checkmate::qassert(shuffle, "B1")
  checkmate::qassert(epochs, "X1[0,)")
  checkmate::qassert(early_stopping, c("0","X1[1,)"))
  checkmate::qassert(custom_parameters, c("0", "L+"))
  checkmate::qassert(plot, "B1")
  checkmate::qassert(verbose, "B1")


  if(!is.function(loss) & !inherits(loss,"family")) {
    loss <- match.arg(loss)
  }

  device <- match.arg(device)

  if(device == "cuda") {
    if(torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")
    } else {
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  } else {
    if(device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }

  n_samples <- dim(X)[1]
  input_dim <- dim(X)[-1]

  if(is.matrix(Y)) {
    y_dim <- ncol(Y)
    ylvls <- colnames(Y)
    if(is.logical(Y) | all(Y %in% c(0,1))) {
      #multilabel <- TRUE
      Y <- torch::torch_tensor(Y, dtype = torch::torch_long())
    } else {
      Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
    }
  } else if(is.factor(Y)) {
    y_dim <- length(levels(Y))
    ylvls <- levels(Y)
    Y <- torch::nnf_one_hot(torch::torch_tensor(Y, dtype=torch::torch_long()))
  } else {
    y_dim <- 1
    Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
  }

}
