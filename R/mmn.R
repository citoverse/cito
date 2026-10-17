#' Train a Multi-Modal Neural Network (MMN)
#'
#' This function trains a Multi-Modal Neural Network (MMN) which consists of a combination of DNNs and CNNs.
#'
#' @param formula A formula object specifying the model structure. See examples for more information.
#' @param dataList A list containing the data for training the model. The list should contain all variables used in the formula.
#' @param fusion_hidden A numeric vector specifying the number of units (nodes) in each hidden layer of the fusion network. The length of this vector determines the number of hidden layers created, with each element specifying the number of units in the corresponding layer.
#' @param fusion_activation A character vector specifying the activation function(s) applied after each hidden layer in the fusion network. If a single character string is provided, the same activation function will be applied to all hidden layers. Alternatively, a character vector of the same length as \code{fusion_hidden} can be provided to apply different activation functions to each layer. Available options include: \code{"relu"}, \code{"leaky_relu"}, \code{"tanh"}, \code{"elu"}, \code{"rrelu"}, \code{"prelu"}, \code{"softplus"}, \code{"celu"}, \code{"selu"}, \code{"gelu"}, \code{"relu6"}, \code{"sigmoid"}, \code{"softsign"}, \code{"hardtanh"}, \code{"tanhshrink"}, \code{"softshrink"}, \code{"hardshrink"}, \code{"log_sigmoid"}.
#' @param fusion_bias A logical value or a vector indicating whether to include bias terms in each layer of the fusion network. If a single logical value is provided, it will apply to all layers. To specify bias inclusion for each layer individually, provide a logical vector of length \code{length(fusion_hidden) + 1}, where each element corresponds to a hidden layer, and the final element controls whether a bias term is added to the output layer.
#' @param fusion_dropout The dropout rate(s) to apply to each hidden layer in the fusion network. This can be a single numeric value (between 0 and 1) to apply the same dropout rate to all hidden layers, or a numeric vector of length \code{length(fusion_hidden)} to set different dropout rates for each layer individually. The dropout rate is not applied to the output layer.
#' @param loss The loss function to be used. Options include "mse", "mae", "cross-entropy", "bernoulli", "gaussian", "binomial", "poisson", "nbinom", "mvp", "multinomial", and "clogit". You can also specify your own loss function. See Details for more information. Default is "mse".
#' @param custom_parameters Parameters for the custom loss function. See the vignette for an example. Default is NULL.
#' @param optimizer The optimizer to be used. Options include "sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop", and "ignite_adam". See \code{\link{config_optimizer}} for further adjustments to the optimizer. Default is "sgd".
#' @param lr Learning rate for the optimizer. Default is 0.01.
#' @param lr_scheduler Learning rate scheduler. See \code{\link{config_lr_scheduler}} for creating a learning rate scheduler. Default is NULL.
#' @param alpha Alpha value for L1/L2 regularization. Default is 0.5.
#' @param lambda Lambda value for L1/L2 regularization. Default is 0.0.
#' @param validation Proportion of the data to be used for validation. Alternatively, a vector containing the indices of the validation samples can be provided. Default is 0.0.
#' @param batchsize Batch size for training. If NULL, batchsize is 10% of the training data. Default is NULL.
#' @param shuffle Whether to shuffle the data before each epoch. Default is TRUE.
#' @param data_augmentation A list of functions used for data augmentation. Elements must be either functions or strings corresponding to inbuilt data augmentation functions. See details for more information.
#' @param epochs Number of epochs to train the model.
#' @param weights weights or other values (can be also a matrix) that the likelihood has access to Default is 100.
#' @param early_stopping Number of epochs with no improvement after which training will be stopped. Default is Inf.
#' @param burnin Number of epochs after which the training stops if the loss is still above the baseloss. Default is Inf.
#' @param baseloss Baseloss used for burnin and plot. If NULL, the baseloss corresponds to intercept only models. Default is NULL.
#' @param device Device to be used for training. Options are "cpu", "cuda", and "mps". Default is "cpu".
#' @param plot Whether to plot the training progress. Default is TRUE.
#' @param verbose Whether to print detailed training progress. Default is TRUE.
#'
#' @return An S3 object of class \code{"citommn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_module". Originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call.}
#' \item{loss}{An object of class "nn_module". Contains all relevant information for the loss function, e.g. parameters and a function (format_Y) that transforms target data.}
#' \item{data}{A list. Contains the data used for the training of the model.}
#' \item{model_properties}{A list of properties, that define the architecture of the model.}
#' \item{training_properties}{A list of all training hyperparameters used the last time the model was trained.}
#' \item{losses}{A data.frame containing training and validation losses of each epoch.}
#' \item{best_epoch_net_state_dict}{Serialized state dict of net from the best training epoch.}
#' \item{best_epoch_loss_state_dict}{Serialized state dict of loss from the best training epoch.}
#' \item{last_epoch_net_state_dict}{Serialized state dict of net from the last training epoch.}
#' \item{last_epoch_loss_state_dict}{Serialized state dict of loss from the last training epoch.}
#' \item{use_model_epoch}{String, either "best" or "last". Determines whether the parameters (e.g. weights, biases) from the best or the last training epoch are used (e.g. for prediction).}
#' \item{loaded_model_epoch}{String, shows from which training epoch the parameters are currently loaded in \code{net} and \code{loss}.}
#'
#' @details
#'
#' # Details:
#'
#' Also check \code{\link{dnn}} and \code{\link{cnn}} for details to common arguments.
#'
#' # MMN architecture:
#'
#' ![](MMN.png "MMN architecture")
#'
#' The MMN combines multiple CNNs and DNNs. This allows the model to process data in different formats (e.g., DNN+CNN for tabular data and images, or CNN+CNN for images with different spatial resolutions).
#' The architecture of the MMN is defined by the arguments \code{formula}, \code{fusion_hidden}, \code{fusion_activation}, \code{fusion_bias} and \code{fusion_dropout}:
#' \itemize{
#'    \item \code{formula} specifies the architecture of the individual networks as well as their respective inputs, and the target data of the MMN (which specifies the shape of the output layer).
#'    \item \code{fusion_hidden}, \code{fusion_activation}, \code{fusion_bias} and \code{fusion_dropout} define the architecture of the DNN that fuses the outputs of the individual networks. See \code{\link{dnn}} for details.
#' }
#'
#' **`mmn(Y ~ dnn(X=tabular_data1) + dnn(~., data=tabular_data2) + cnn(X=image_data), dataList=mmn_data, ...)`**
#'
#' In this example, **Y** (left side of ~) is the target data of the MMN. On the right side of ~ you can specify as many DNNs and CNNs as required.
#' The specification works exactly as in \code{\link{dnn}} and \code{\link{cnn}} with the following restrictions:
#'
#' \itemize{
#'    \item Only specify arguments that relate to the architecture and input data of the network (bold arguments mandatory):
#'        \itemize{
#'            \item dnn(): **formula**, **data**, hidden, activation, bias, dropout, (**X**, alternatively to formula and data)
#'            \item cnn(): **X**, **architecture**
#'        }
#'    \item Arguments relating to the training (e.g. loss, lr, epochs, ...) have to be passed to mmn() instead.
#'    \item The names of the data variables (in this example: Y, tabular_data1, tabular_data2, image_data) must be available in \code{dataList} (named list).
#' }
#'
#' @import checkmate
#' @example /inst/examples/mmn-example.R
#' @author Armin Schenk
#' @seealso \code{\link{predict.citommn}}, \code{\link{print.citommn}}, \code{\link{summary.citommn}}, \code{\link{coef.citommn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#' @export
mmn <- function(formula,
                dataList = NULL,
                fusion_hidden = c(50L, 50L),
                fusion_activation = "relu",
                fusion_bias = TRUE,
                fusion_dropout = 0.0,
                loss = c("mse", "mae", "cross-entropy", "bernoulli", "gaussian", "binomial", "poisson", "mvp", "nbinom", "multinomial", "clogit", "softmax"),
                custom_parameters = NULL,
                optimizer = c("sgd","adam","adadelta", "adagrad", "rmsprop", "rprop", "ignite_adam"),
                lr = 0.01,
                lr_scheduler = NULL,
                alpha = 0.5,
                lambda = 0.0,
                validation = 0.0,
                batchsize = NULL,
                shuffle = TRUE,
                data_augmentation = NULL,
                epochs = 100,
                weights = NULL,
                early_stopping = Inf,
                burnin = Inf,
                baseloss = NULL,
                device = c("cpu", "cuda", "mps"),
                plot = TRUE,
                verbose = TRUE) {

  # symbols in the formula that are not in dataList (e.g. the architecture object) are
  # resolved where the formula was written, parent.frame() would be wrong whenever mmn()
  # is called indirectly (do.call, a wrapper function, lapply, ...)
  caller_env <- environment(formula)
  if(is.null(caller_env)) caller_env <- parent.frame()
  checkmate::assert(checkmate::checkList(dataList), checkmate::checkNull(dataList))
  checkmate::qassert(custom_parameters, c("0", "L+"))
  checkmate::qassert(lr, "N1(0,)")
  checkmate::qassert(alpha, "N1[0,1]")
  checkmate::qassert(lambda, "N1[0,)")
  checkmate::qassert(validation, c("N1[0,1)","X>1[1,)"))
  checkmate::qassert(batchsize, c("0", "X1[1,)"))
  checkmate::qassert(shuffle, "B1")
  checkmate::qassert(data_augmentation, c("0", "L+"))
  checkmate::qassert(epochs, "X1[0,)")
  checkmate::qassert(early_stopping, "N1[1,]")
  checkmate::qassert(burnin, "N1[1,]")
  checkmate::qassert(baseloss, c("0", "N1"))
  checkmate::qassert(device, "S+[3,)")
  checkmate::qassert(plot, "B1")
  checkmate::qassert(verbose, "B1")

  device <- match.arg(device)

  # if(!is.function(loss) & !inherits(loss,"family")) {
  #   loss <- match.arg(loss)
  #
  #   if((device == "mps") & (loss %in% c("poisson", "nbinom", "multinomial"))) {
  #     message("`poisson`, `nbinom`, and `multinomial` are not yet supported for `device=mps`, switching to `device=cpu`")
  #     device = "cpu"
  #   }
  # }
  #
  # if(inherits(loss,"family")) {
  #   if((device == "mps") & (loss$family %in% c("poisson", "nbinom"))) {
  #     message("`poisson` or `nbinom` are not yet supported for `device=mps`, switching to `device=cpu`")
  #     device = "cpu"
  #   }
  # }

  device_old <- device
  device <- check_device(device)

  formula <- stats::terms(formula)
  check_mmn_formula(formula)

  if(!inherits(formula[[2]], "name")) stop(paste0("In '", deparse1(formula), "': Left side of ~ (target data) has to be assigned to a name that exists in dataList. E.g., mmn(some_name ~ dnn(...) + cnn(...), dataList=list(some_name=your_data, ...), ...)"))
  if(!as.character(formula[[2]]) %in% names(dataList)) stop(paste0("In '", deparse1(formula), "': Couldn't find '", as.character(formula[[2]]), "' in names of dataList."))
  Y <- eval(formula[[2]], envir = dataList)

  if(is.character(loss)) loss <- match.arg(loss)
  if(!is.null(weights))
    weights <- matrix(weights, nrow = if(is.null(nrow(Y))) length(Y) else nrow(Y))
  loss_obj <- get_loss(loss, Y, custom_parameters, baseloss, weights)
  assert_weights_supported(loss, weights)
  baseloss <- loss_obj$baseloss

  X <- format_input_data(formula[[3]], dataList)

  from_folder = FALSE
  if(any(sapply(X, is.character))) from_folder = TRUE

  X <- lapply(X, function(x) {
    if(is.character(x)) x = list.files(x, full.names = TRUE)
    return(x)
    })

  Y <- loss_obj$format_Y(Y)
  # a batchsize of 0 gives a dataloader of length Inf that yields empty batches forever
  if(is.null(batchsize)) batchsize = max(1L, round(0.1*dim(Y)[1]))

  if(!is.null(data_augmentation)) data_augmentation <- check_data_augmentation(data_augmentation)

  if(length(validation) == 1 && validation == 0) {
    valid <- NULL
  } else {
    n_samples <- dim(Y)[1]
    if(length(validation) > 1) {
      if(any(validation>n_samples)) stop("Validation indices mustn't exceed the number of samples.")
      valid <- validation
    } else {
      valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
    }
  }

  weights_torch <- if(is.null(weights)) NULL else torch::torch_tensor(weights)
  dls <- build_loaders(X, Y, weights = weights_torch, valid = valid, batch_size = batchsize, shuffle = shuffle,
                       from_folder = from_folder, data_augmentation = data_augmentation)
  train_dl <- dls$train_dl
  valid_dl <- dls$valid_dl

  model_properties <- list()
  model_properties$subModules <- get_model_properties(formula[[3]], dataList, caller_env)
  model_properties$fusion <- list(output = loss_obj$y_dim,
                                  hidden = fusion_hidden,
                                  activation = fusion_activation,
                                  bias = fusion_bias,
                                  dropout = fusion_dropout)
  class(model_properties) <- "citommn_properties"

  net <- build_mmn(model_properties)

  training_properties <- list(optimizer = optimizer,
                              lr = lr,
                              lr_scheduler = lr_scheduler,
                              alpha = alpha,
                              lambda = lambda,
                              validation = validation,
                              batchsize = batchsize,
                              shuffle = shuffle,
                              data_augmentation = data_augmentation,
                              epochs = epochs, #redundant?
                              early_stopping = early_stopping,
                              burnin = burnin,
                              baseloss = baseloss,
                              device = device_old,
                              plot = plot,
                              verbose = verbose,
                              has_weights = !is.null(weights))

  out <- list()
  class(out) <- "citommn"
  out$net <- net
  out$version = "1.2"
  out$call <- match.call()
  out$call$formula <- formula
  out$loss <- loss_obj
  out$data <- list(dataList=dataList)
  if(length(validation) > 1 || validation != 0) out$data <- append(out$data, list(validation = valid))
  out$model_properties <- model_properties
  out$training_properties <- training_properties

  out <- train_model(model = out, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl)

  return(out)
}

#' Predict with a fitted MMN model
#'
#' This function generates predictions from a Multi-Modal Neural Network (MMN) model that was created using the \code{\link{mmn}} function.
#'
#' @param object a model created by \code{\link{mmn}}
#' @param newdata A list containing the new data for which predictions are to be made. The dimensions of the elements in \code{newdata} should match those of the training data, except for the respective first dimensions which represents the number of samples. If \code{NULL}, the function uses the data the model was trained on.
#' @param type A character string specifying the type of prediction to be made. Options are:
#' \itemize{
#'   \item \code{"link"}: Scale of the linear predictor.
#'   \item \code{"response"}: Scale of the response.
#'   \item \code{"class"}: The predicted class labels (for classification tasks).
#' }
#' @param device Device to be used for making predictions. Options are "cpu", "cuda", and "mps". If \code{NULL}, the function uses the same device that was used when training the model. Default is \code{NULL}.
#' @param batchsize An integer specifying the number of samples to be processed at the same time. If \code{NULL}, the function uses the same batchsize that was used when training the model. Default is \code{NULL}.
#' @param return_embeddings return the embeddings (output of the last hidden layer of the fusion network) instead of the predictions. Default is \code{FALSE}.
#' @param ... Additional arguments (currently not used).
#' @return A matrix of predictions. If \code{type} is \code{"class"}, a factor of predicted class labels is returned.
#'
#' @export
predict.citommn <- function(object,
                            newdata = NULL,
                            type=c("link", "response", "class"),
                            device = NULL,
                            batchsize = NULL,
                            return_embeddings = FALSE,
                            ...) {

  checkmate::assert(checkmate::checkNull(newdata),
                    checkmate::checkList(newdata))

  object <- check_model(object)

  type <- match.arg(type)

  if(is.null(device)) device <- object$training_properties$device
  device <- check_device(device)

  object$net$to(device = device)
  object$loss$to(device = device)

  if(is.null(batchsize)) batchsize <- object$training_properties$batchsize


  if(type %in% c("response", "class")) {
    link <- object$loss$invlink
  }else{
    link = function(a) a
  }

  # format_input_data() returns tensors, which carry no dimnames, so the sample names have to be
  # taken first. Without them metadata[rownames(pred), ] silently yields nothing downstream and an
  # accuracy comes back as NA rather than as an error.
  sample_names <- NULL
  for(x in if(is.null(newdata)) object$data$dataList else newdata) {
    nm <- if(is.character(x) && length(x) == 1L && dir.exists(x)) basename(list.files(x))
          else if(!is.null(dim(x))) dimnames(x)[[1]]
          else names(x)
    if(!is.null(nm)) {
      sample_names <- nm
      break
    }
  }

  if(is.null(newdata)){
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = object$data$dataList)
  } else {
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = newdata)
    # the training data is routinely dropped from a fitted MMN to keep the saved object small,
    # so the per-stream shape check is skipped when it is gone rather than failing the predict
    old <- try(format_input_data(object$call$formula[[3]], object$data$dataList), silent = TRUE)
    check_newdata_shape(newdata, if(inherits(old, "try-error")) NULL else old)
  }

  from_folder = FALSE
  if(any(sapply(newdata, is.character))) from_folder = TRUE

  newdata <- lapply(newdata, function(x) {
    if(is.character(x)) x = list.files(x, full.names = TRUE)
    return(x)
  })

  nt <- if(is.null(object$training_properties$num_threads)) 4L else object$training_properties$num_threads
  dl <- do.call(get_data_loader, append(newdata, list(batch_size = batchsize, shuffle = FALSE,
                                                      from_folder = from_folder, num_threads = nt)))

  pred <- NULL
  coro::loop(for(b in dl) {
    b <- lapply(b, function(x) x$to(device=device, non_blocking= TRUE))
    if(is.null(pred)) pred <- torch::as_array(link(object$net(b, return_embeddings = return_embeddings))$to(device="cpu"))
    else pred <- rbind(pred, torch::as_array(link(object$net(b, return_embeddings = return_embeddings))$to(device="cpu")))
  })

  if(!is.null(sample_names) && length(sample_names) == nrow(pred)) rownames(pred) <- sample_names

  if(!return_embeddings) {

    if(!is.null(object$loss$responses)) {
      colnames(pred) <- object$loss$responses
      if(type == "class") pred <- factor(apply(pred, 1, function(x) object$loss$responses[which.max(x)]), levels = object$loss$responses)
    }
  }

  return(pred)
}

#' Print a fitted MMN model
#'
#' This function prints the architecture of a Multi-Modal Neural Network (MMN) model created using the \code{\link{mmn}} function.
#'
#' @param x A model created by \code{\link{mmn}}.
#' @param ... Additional arguments (currently not used).
#' @return The original model object \code{x}, returned invisibly.
#' @export
print.citommn <- function(x, ...){
  x <- check_model(x)
  print(x$call)
  print(x$net)
  return(invisible(x))
}

#' Summarize a fitted MMN model
#'
#' Reports the permutation importance of each data stream: one stream is shuffled across the batch
#' while the others are left intact, and the increase in loss is what that stream contributed.
#'
#' @param object A model created by \code{\link{mmn}}.
#' @param n_permute Number of permutations per data stream. Default is 10.
#' @param device Device on which to evaluate. Defaults to the device the model was trained on.
#' @param ... Additional arguments (currently not used).
#'
#' @details
#' This is the stream-level analogue of permutation importance for a DNN, and is the interpretive
#' tool a multi-modal model admits, since per-feature effects are not comparable across modalities
#' of different kinds. The standard deviation across permutations is reported beside the estimate,
#' because a stream that contributes little can come out negative by chance.
#'
#' @return An object of class \code{"summary.citommn"} holding one row per data stream with the
#'   increase in loss, its standard deviation across permutations, and the share of the total.
#' @export
summary.citommn <- function(object, n_permute = 10L, device = NULL, ...){
  object <- check_model(object)
  out <- list(call = object$call,
              importance = modality_importance(object, n_permute = n_permute, device = device))
  class(out) <- "summary.citommn"
  return(out)
}

#' @rdname summary.citommn
#' @param x a summary of a fitted MMN
#' @export
print.summary.citommn <- function(x, ...){
  cat("Summary of Multi-Modal Neural Network\n\n")
  cli::cli_h3(cli::style_bold("Permutation importance of the data streams"))
  m <- as.matrix(x$importance[, c("loss_increase", "sd", "share"), drop = FALSE])
  rownames(m) <- x$importance$modality
  colnames(m) <- c("Loss increase", "SD", "Share")
  print(round(m, 4))
  cat(sprintf("\nBaseline loss %.4f, %d permutations per stream. A stream that contributes little\ncan come out negative by chance; compare the estimate against its SD before reading it.\n",
              x$importance$baseline_loss[1], x$importance$n_permute[1]))
  invisible(x)
}

# Permuting one data stream across the batch and refitting nothing: the increase in loss is what
# that stream contributed. This is the stream-level analogue of permutation importance for a DNN,
# and the only interpretive handle an MMN has, since per-pixel effects are not defined across
# modalities of different kinds.
modality_importance <- function(object, n_permute = 10L, device = NULL, batchsize = NULL) {
  checkmate::qassert(n_permute, "X1[1,)")
  object <- check_model(object)
  if(is.null(device)) device <- object$training_properties$device
  device <- check_device(device)
  if(is.null(batchsize)) batchsize <- object$training_properties$batchsize

  inputs <- format_input_data(object$call$formula[[3]], object$data$dataList)
  Y <- object$loss$format_Y(eval(object$call$formula[[2]], envir = object$data$dataList))
  n <- dim(Y)[1]
  labels <- modality_labels(object$call$formula[[3]])
  if(length(labels) != length(inputs)) labels <- paste0("stream_", seq_along(inputs))

  object$net$to(device = device)
  object$loss$to(device = device)
  run_loss <- function(ins) {
    dl <- do.call(get_data_loader, append(lapply(ins, to_input_tensor), 
                                          list(Y, batch_size = batchsize, shuffle = FALSE)))
    total <- 0
    count <- 0
    coro::loop(for(b in dl) {
      b <- lapply(b, function(z) z$to(device = device, non_blocking = TRUE))
      out <- object$net(b[-length(b)])
      l <- object$loss(out, b[[length(b)]])$mean()
      total <- total + as.numeric(l)*dim(b[[length(b)]])[1]
      count <- count + dim(b[[length(b)]])[1]
    })
    total/count
  }

  base <- run_loss(inputs)
  draws <- lapply(seq_along(inputs), function(j) {
    vapply(seq_len(n_permute), function(k) {
      perm <- inputs
      perm[[j]] <- permute_rows(perm[[j]], sample.int(n))
      run_loss(perm)
    }, numeric(1)) - base
  })
  inc <- vapply(draws, mean, numeric(1))
  # a stream that contributes little can come out negative by chance, so the spread across
  # permutations is reported beside the estimate rather than left for the reader to assume away
  sd_inc <- vapply(draws, stats::sd, numeric(1))

  share <- if(all(inc <= 0)) rep(NA_real_, length(inc)) else pmax(inc, 0)/sum(pmax(inc, 0))
  data.frame(modality = labels, loss_increase = inc, sd = sd_inc, share = share,
             baseline_loss = base, n_permute = n_permute, stringsAsFactors = FALSE)
}

# the submodule calls in formula order, so that the table names the streams the way the user wrote them
modality_labels <- function(term) {
  if(length(term) > 1 && term[[1]] == "+")
    return(c(modality_labels(term[[2]]), modality_labels(term[[3]])))
  deparse1(term)
}

permute_rows <- function(x, idx) {
  if(is.character(x)) return(x[idx])
  if(is.null(dim(x))) return(x[idx])
  abind::asub(x, idx, dims = 1, drop = FALSE)
}

to_input_tensor <- function(x) {
  if(is.character(x)) return(list.files(x, full.names = TRUE))
  if(inherits(x, "torch_tensor")) return(x)
  torch::torch_tensor(x, dtype = torch::torch_float32())
}

#' Plot a fitted MMN model
#'
#' Visualises either the permutation importance of the data streams or the architecture of a
#' Multi-Modal Neural Network.
#'
#' @param x a model created by \code{\link{mmn}}
#' @param plot_type \code{"importance"} draws the permutation importance of each data stream,
#'   \code{"arch"} the architecture: one row per data stream with its submodel, the concatenation,
#'   the fusion network and the output. \code{plot()} on a single \code{\link{create_architecture}}
#'   object still draws that submodel's layers in full.
#' @param n_permute number of permutations per data stream, only used for \code{"importance"}
#' @param device device on which to evaluate, defaults to the device the model was trained on
#' @param ... passed to \code{\link[graphics]{barplot}}, only used for \code{"importance"}
#'
#' @return invisibly, the importance table for \code{"importance"} and \code{NULL} otherwise
#' @export
plot.citommn <- function(x, plot_type = c("importance", "arch"), n_permute = 10L,
                         device = NULL, ...) {
  plot_type <- match.arg(plot_type)
  if(plot_type == "arch") {
    x <- check_model(x)
    plot_mmn_architecture(x)
    return(invisible(NULL))
  }
  imp <- modality_importance(x, n_permute = n_permute, device = device)
  labels <- vapply(imp$modality, short_stream_label, character(1), USE.NAMES = FALSE)
  op <- graphics::par(mar = c(5.1, max(4.1, 0.6*max(nchar(labels)) + 1.5), 4.1, 2.1))
  on.exit(graphics::par(op))
  # the whiskers reach past the bars, and a stream can come out negative, so neither end is given
  xlim <- range(0, imp$loss_increase + imp$sd, imp$loss_increase - imp$sd)
  mids <- graphics::barplot(imp$loss_increase, names.arg = labels, horiz = TRUE, las = 1,
                            xlim = xlim, xlab = "Increase in loss when the stream is permuted",
                            main = "Data stream importance", ...)
  graphics::arrows(imp$loss_increase - imp$sd, mids, imp$loss_increase + imp$sd, mids,
                   code = 3, angle = 90, length = 0.05)
  return(invisible(imp))
}

# the deparsed submodule call is far too wide for an axis label, so the stream is named by the
# data it reads, which is what distinguishes the streams from one another anyway
short_stream_label <- function(term) {
  nm <- stream_data_name(term)
  if(is.null(nm)) return(term)
  paste0(as.character(str2lang(term)[[1]]), "(", nm, ")")
}

stream_data_name <- function(term) {
  cl <- str2lang(term)
  fn <- as.character(cl[[1]])
  if(!fn %in% c("cnn", "dnn")) return(NULL)
  cl <- match.call(switch(fn, cnn = cnn, dnn = dnn), cl)
  arg <- if(fn == "cnn") cl$X else cl$data
  if(is.null(arg)) NULL else deparse1(arg)
}

# A multi-modal model is a graph, not a stack, so the drawing that helps is the streams side by
# side feeding one fusion head; plot(architecture) still draws a single submodel's layers.
plot_mmn_architecture <- function(x) {
  props <- x$model_properties
  subs <- props$subModules
  n <- length(subs)
  terms <- modality_labels(x$call$formula[[3]])
  if(length(terms) != n) terms <- paste0("stream_", seq_len(n))
  widths <- submodule_widths(x$net, subs)

  data_box <- lapply(seq_len(n), function(i) {
    nm <- stream_data_name(terms[i])
    c(if(is.null(nm)) terms[i] else nm,
      if(inherits(subs[[i]], "citocnn_properties")) paste(subs[[i]]$input, collapse = " x ")
      else paste0(subs[[i]]$input, " features"))
  })
  sub_box <- lapply(seq_len(n), function(i)
    c(if(inherits(subs[[i]], "citocnn_properties")) "CNN" else "DNN", stream_layers(subs[[i]]),
      if(!is.na(widths[i])) paste0("output: ", widths[i])))
  fusion_box <- c("Fusion", if(!anyNA(widths)) paste0("concatenated: ", sum(widths)),
                  paste0("hidden ", paste(props$fusion$hidden, collapse = ", ")),
                  props$fusion$activation[1])
  out_box <- c("Output", paste0(props$fusion$output,
                                if(props$fusion$output > 1) " responses" else " response"),
               deparse1(x$call$loss))
  boxes <- c(data_box, sub_box, list(fusion_box), list(out_box))
  xl <- c(rep(0.01, n), rep(0.21, n), 0.60, 0.87)
  xr <- c(rep(0.17, n), rep(0.52, n), 0.82, 0.99)

  op <- graphics::par(mar = c(0.5, 0.5, 2.5, 0.5))
  on.exit(graphics::par(op))
  graphics::plot.new()
  graphics::title(main = "Multi-modal architecture")

  # the boxes are sized from their own rendered text, so a deep CNN does not spill out of its
  # rectangle; strheight of a single line measures ink, not the line box, so the block is measured
  h <- 1/n
  pad <- graphics::strheight("Ag")
  bh <- vapply(boxes, function(b) graphics::strheight(paste(b, collapse = "\n")), numeric(1)) + pad
  lim <- c(rep(0.86*h, 2*n), 0.9, 0.9)
  cex <- min(1, lim/bh, 0.92*(xr - xl)/vapply(boxes, function(b) max(graphics::strwidth(b)), numeric(1)))
  bh <- bh*cex/2

  ys <- 1 - (seq_len(n) - 0.5)*h
  yc <- c(ys, ys, 0.5, 0.5)
  cols <- c(rep("orange", n),
            vapply(subs, function(p) if(inherits(p, "citocnn_properties")) "lightblue" else "lightgreen",
                   character(1)), "lightyellow", "pink")
  for(k in seq_along(boxes)) {
    graphics::rect(xl[k], yc[k] - bh[k], xr[k], yc[k] + bh[k], col = cols[k])
    graphics::text((xl[k] + xr[k])/2, yc[k], paste(boxes[[k]], collapse = "\n"), cex = cex)
  }
  for(i in seq_len(n)) {
    graphics::arrows(0.17, ys[i], 0.21, ys[i], length = 0.08)
    graphics::arrows(0.52, ys[i], 0.60, 0.5, length = 0.08)
  }
  graphics::arrows(0.82, 0.5, 0.87, 0.5, length = 0.08)
}

# the exact width each stream contributes to the fusion input, taken from the built module rather
# than recomputed from the layer list, which would have to redo every shape calculation
submodule_widths <- function(net, props) {
  vapply(seq_along(props), function(j) {
    if(!is.null(props[[j]]$custom$output_dim)) return(as.integer(props[[j]]$custom$output_dim))
    w <- try(dim(net$subModules[[j]](torch::torch_rand(c(2L, props[[j]]$input))))[2], silent = TRUE)
    if(inherits(w, "try-error")) NA_integer_ else as.integer(w)
  }, integer(1))
}

stream_layers <- function(p) {
  if(!is.null(p$custom)) return("custom module")
  if(!inherits(p, "citocnn_properties"))
    return(c(paste0("hidden ", paste(p$hidden, collapse = ", ")), p$activation[1]))
  vapply(p$architecture, function(l) {
    switch(class(l)[1],
           transfer = paste0("transfer(", l$name, ")"),
           conv     = paste0("conv(", l$n_kernels, ", ", paste(l$kernel_size, collapse = "x"), ")"),
           maxPool  = paste0("maxPool(", paste(l$kernel_size, collapse = "x"), ")"),
           avgPool  = paste0("avgPool(", paste(l$kernel_size, collapse = "x"), ")"),
           linear   = paste0("linear(", l$n_neurons, ")"),
           class(l)[1])
  }, character(1))
}

#' Retrieve parameters of a fitted MMN model
#'
#' This function returns the list of parameters (weights and biases) and buffers (e.g. running mean and variance of batch normalization layers) currently in use by the neural network model created using the \code{\link{mmn}} function.
#'
#' @param object A model created by \code{\link{mmn}}.
#' @param ... Additional arguments (currently not used).
#' @return A list with two components:
#' \itemize{
#'   \item \code{net_parameters}: A list of the model's weights and biases for the currently used model epoch.
#'   \item \code{net_buffers}: A list of buffers (e.g., running statistics) for the currently used model epoch.
#'   \item \code{loss_parameters}: A list of the loss function's parameters for the currently used model epoch.
#' }
#' @export
coef.citommn <- function(object,...){
  object <- check_model(object)
  out <- list()
  out$net_parameters <- lapply(object$net$parameters, function(x) torch::as_array(x$to(device = "cpu")))
  if(!is.null(object$net$buffers)) out$net_buffers <- lapply(object$net$buffers, function(x) torch::as_array(x$to(device = "cpu")))
  if(!is.null(object$loss$parameters)) out$loss_parameters <- lapply(object$loss$parameters, function(x) torch::as_array(x$to(device = "cpu")))
  return(out)
}

# Without this a wrong image size surfaces as a torch matmul error from several frames down, and
# streams of differing length are silently truncated to the shortest.
check_newdata_shape <- function(new, old) {
  n <- vapply(new, function(x) if(is.character(x)) length(list.files(x)) else
                               if(is.null(dim(x))) length(x) else dim(x)[1], numeric(1))
  if(length(unique(n)) > 1L)
    stop(paste0("The data streams have different numbers of samples (", paste(n, collapse = ", "),
                "). Every stream must describe the same samples, in the same order."), call. = FALSE)
  if(is.null(old) || length(old) != length(new)) return(invisible(TRUE))
  for(i in seq_along(new)) {
    if(is.character(new[[i]]) || is.character(old[[i]])) next
    d_new <- dim(new[[i]])
    d_old <- dim(old[[i]])
    if(is.null(d_new) || is.null(d_old) || length(d_new) != length(d_old)) next
    if(!identical(d_new[-1], d_old[-1]))
      stop(paste0("Data stream ", i, " has shape ", paste(d_new[-1], collapse = "x"),
                  " per sample, but the model was fitted on ", paste(d_old[-1], collapse = "x"),
                  "."), call. = FALSE)
  }
  invisible(TRUE)
}

format_input_data <- function(formula, dataList) {

  inner <- function(term, input_data) {
    if(term[[1]] == "+") {
      input_data <- inner(term[[2]], input_data)
      input_data <- inner(term[[3]], input_data)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)
      X <- NULL
      formula <- NULL
      data <- NULL
      if(!is.null(call$X)) {
        if(!inherits(call$X, "name")) stop(paste0("In '", deparse1(term), "': X has to be assigned to a name that exists in dataList. E.g., mmn(Y ~ dnn(X=some_name, ...) + cnn(...), dataList=list(Y=Y, some_name=your_data, ...), ...)"))
        if(!as.character(call$X) %in% names(dataList)) stop(paste0("In '", deparse1(term), "': Couldn't find '", as.character(call$X), "' in names of dataList."))
        X <- eval(call$X, envir = dataList)
      } else if(!is.null(call$formula)) {
        formula <- stats::formula(call$formula)
        if(is.null(call$data)) stop(paste0("In '", deparse1(term), "': When using 'formula', 'data' has to be specified."))
        if(!inherits(call$data, "name")) stop(paste0("In '", deparse1(term), "': data has to be assigned to a name that exists in dataList. E.g., mmn(Y ~ dnn(~ A + e(B), data = some_name, ...) + cnn(...), dataList=list(Y=Y, some_name=your_data, ...), ...)"))
        if(!as.character(call$data) %in% names(dataList)) stop(paste0("In '", deparse1(term), "': Couldn't find '", as.character(call$data), "' in names of dataList."))
        data <- data.frame(eval(call$data, envir = dataList))
      }
      temp <- get_X_Y(formula, X, NULL, data)
      input_data <- append(input_data, list(torch::torch_tensor(temp$X, dtype = torch::torch_float32())))
      if (!is.null(temp$Z)) input_data <- append(input_data, list(torch::torch_tensor(temp$Z, dtype = torch::torch_long())))
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      if(!inherits(call$X, "name")) stop(paste0("In '", deparse1(term), "': X has to be assigned to a name that exists in dataList. E.g., mmn(Y ~ cnn(X=some_name, ...) + dnn(...), dataList=list(Y=Y, some_name=your_data, ...), ...)"))
      if(!as.character(call$X) %in% names(dataList)) stop(paste0("In '", deparse1(term), "': Couldn't find '", as.character(call$X), "' in names of dataList."))
      if(is.character(eval(call$X, envir = dataList))) input_data <- append(input_data, list(eval(call$X, envir = dataList)))
      else input_data <- append(input_data, list(torch::torch_tensor(eval(call$X, envir = dataList), dtype = torch::torch_float32())))
    }
    return(input_data)
  }

  return(inner(formula, list()))
}

get_model_properties <- function(formula, dataList, caller_env = parent.frame()) {
  eval_env <- list2env(dataList, parent = caller_env)
  inner <- function(term, model_properties) {
    if(term[[1]] == "+") {
      model_properties <- inner(term[[2]], model_properties)
      model_properties <- inner(term[[3]], model_properties)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)
      model_properties <- append(model_properties, list(eval(call, envir = eval_env)))
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      model_properties <- append(model_properties, list(eval(call, envir = eval_env)))
    }
    return(model_properties)
  }

  return(inner(formula, list()))
}

check_mmn_formula <- function(formula) {

  inner <- function(term) {
    if(length(term) == 1) {
      stop(paste0("In '", deparse1(formula), "': 'formula' in mmn() must not contain symbols outside of dnn() or cnn() structures. Therefore, '", deparse1(term), "' is not allowed."))
    }
    if(term[[1]] == "+") {
      inner(term[[2]])
      inner(term[[3]])
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)
      if(is.null(call$formula) & is.null(call$X)) stop(paste0("In '", deparse1(term), "': Either 'formula' or 'X' must be specified."))
      if(!is.null(call$Y)) stop(paste0("In '", deparse1(term), "': 'Y' must not be specified."))
      if(!is.null(call$formula)) {
        if(call$formula[[1]] != "~") stop(paste0("In '", deparse1(term), "': ~ missing in 'formula'."))
        if(length(call$formula) != 2) stop(paste0("In '", deparse1(term), "': response (left side of ~) of 'formula' must not be specified."))
      }
      args <- names(formals(dnn))
      allowed_args <- c("formula", "data", "hidden", "activation", "bias", "dropout", "X", "custom")
      for(arg in setdiff(args, allowed_args)) {
        if(!is.null(call[[arg]])) warning(paste0("In '", deparse1(term), "': Specifying '", arg, "' here has no effect. When using dnn() within mmn() only the following arguments are important: ", paste0(allowed_args, collapse = ", ")))
      }
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      if(!is.null(call$Y)) stop(paste0("In '", deparse1(term), "': 'Y' must not be specified."))
      if(is.null(call$X)) stop(paste0("In '", deparse1(term), "': 'X' must be specified."))
      if(is.null(call$architecture) & is.null(call$custom)) stop(paste0("In '", deparse1(term), "': either 'architecture' or 'custom' must be specified."))
      args <- names(formals(cnn))
      allowed_args <- c("X", "architecture", "custom")
      for(arg in setdiff(args, allowed_args)) {
        if(!is.null(call[[arg]])) warning(paste0("In '", deparse1(term), "': Specifying '", arg, "' here has no effect. When using cnn() within mmn() only the following arguments are important: ", paste0(allowed_args, collapse = ", ")))
      }
    } else {
      stop(paste0("In '", deparse1(formula), "': Symbol not supported: ", term[[1]]))
    }
  }

  if(formula[[1]] != "~") stop(paste0("Incorrect formula '", deparse1(formula), "': ~ missing."))
  if(length(formula) == 2) stop(paste0("Incorrect formula '", deparse1(formula), "': response (left side of ~) missing."))
  inner(formula[[3]])
}

