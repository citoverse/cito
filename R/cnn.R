#' CNN
#' @export
cnn <- function(X,
                Y,
                layers = NULL, #Default einfuegen
                n_kernels = 10,
                kernel_size = list(conv = 3, pool = 2),
                stride = list(conv = 1, pool = NULL),
                padding = 0,
                dilation = 1,
                n_neurons = 10,
                bias = TRUE,
                activation = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                               "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                               "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"),
                normalization = FALSE,
                dropout = 0.0,
                lambda = 0.0,
                alpha = 0.5,
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
  check_listable_parameter(stride, c("X<=3[1,)","0"))
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
  input_shape <- dim(X)[-1]

  if(is.matrix(Y)) {
    y_dim <- ncol(Y)
    ylvls <- colnames(Y)
    if(is.logical(Y) | all(Y %in% c(0,1))) {
      #multilabel <- TRUE
      Y <- torch::torch_tensor(Y, dtype = torch::torch_long())
      y_dtype <- torch::torch_long()
    } else {
      Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
      y_dtype <- torch::torch_float32()
    }
  } else if(is.factor(Y)) {
    y_dim <- length(levels(Y))
    ylvls <- levels(Y)
    Y <- torch::nnf_one_hot(torch::torch_tensor(Y, dtype=torch::torch_long()))
    y_dtype <- torch::torch_long()
  } else {
    y_dim <- 1
    Y <- torch::torch_tensor(Y, dtype = torch::torch_float32())
    y_dtype <- torch::torch_float32()
  }

  loss_obj <- get_loss(loss)
  if(!is.null(loss_obj$parameter)) loss_obj$parameter <- list(scale = loss_obj$parameter)
  if(!is.null(custom_parameters)){
    if(!inherits(custom_parameters,"list")){
      warning("custom_parameters has to be list")
    } else {
      custom_parameters <- lapply(custom_parameters, function(x) torch::torch_tensor(x, requires_grad = TRUE, device = device))
      loss_obj$parameter <- append(loss_obj$parameter, unlist(custom_parameters))
    }
  }

  if(validation != 0) {
    valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
    train <- c(1:n_samples)[-valid]
    train_dl <- get_data_loader(X[train,], Y[train,], batch_size = batchsize, shuffle = shuffle, y_dtype = y_dtype)
    valid_dl <- get_data_loader(X[valid,], Y[valid,], batch_size = batchsize, shuffle = shuffle, y_dtype = y_dtype)
  } else {
    train_dl <- get_data_loader(X, Y, batch_size = batchsize, shuffle = shuffle, y_dtype = y_dtype)
    valid_dl <- NULL
  }

  layers <- fill_layer_parameters(layers = layers,
                                  input_dim = length(input_shape)-1,
                                  n_kernels = n_kernels,
                                  kernel_size = kernel_size,
                                  stride = stride,
                                  padding = padding,
                                  dilation = dilation,
                                  n_neurons = n_neurons,
                                  bias = bias,
                                  activation = activation,
                                  normalization = normalization,
                                  dropout = dropout)

  net <- build_cnn(input_shape = input_shape,
                   output = y_dim,
                   layers = layers)

  return(net)
}




