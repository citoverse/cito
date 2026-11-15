build_model <- function(object) {
  if(inherits(object, "citodnn")) {
    net <- build_dnn(object$model_properties)
  } else if(inherits(object, "citodnn_properties")) {
    net <- build_dnn(object)
  } else if(inherits(object, "citocnn")) {
    net <- build_cnn(object$model_properties)
  } else if(inherits(object, "citocnn_properties")) {
    net <- build_cnn(object)
  } else if(inherits(object, "citommn")) {
    net <- build_mmn(object$model_properties)
  } else if(inherits(object, "citommn_properties")) {
    net <- build_mmn(object)
  } else if(inherits(object, "citotnn")) {
    net <- build_tnn(object$model_properties)
  } else if(inherits(object, "citotnn_properties")) {
    net <- build_tnn(object)
  } else {
    stop("Object not of class citodnn, citodnn_properties, citocnn, citocnn_properties, citommn or citommn_properties")
  }
  return(net)
}


# builds a cito-compatible net from a user-provided nn_module
# the adapter provides the classname, the has_embeddings field, and the (x, z, return_embeddings) signature
# that the training loop, the predict methods, mmn() and the xAI functions rely on
build_custom = function(model_properties, classname) {
  custom = model_properties$custom

  if(!is.null(custom$serialized)) {
    inner = torch::torch_load(custom$serialized) # user passed an instantiated module
  } else {
    # the generator has the same formals as the initialize method, if it accepts
    # input/output/embeddings, they are filled in automatically (unless provided in args),
    # so that custom modules don't have to hard code e.g. the number of predictors
    args = custom$args
    supported = names(formals(custom$module))
    output = model_properties$output
    if(is.null(output)) output = custom$output_dim # within mmn() there is no output, submodules are sized via output_dim
    if("input" %in% supported && is.null(args$input)) args$input = model_properties$input
    if("output" %in% supported && is.null(args$output)) args$output = output
    if("embeddings" %in% supported && is.null(args$embeddings)) args$embeddings = model_properties$embeddings$dims
    inner = do.call(custom$module, args)
  }

  has_embeddings = !is.null(model_properties$embeddings)
  pass_re = "return_embeddings" %in% names(formals(inner))

  self = NULL
  adapter = torch::nn_module(
    classname = classname,
    initialize = function() {
      self$custom = inner
      self$has_embeddings = has_embeddings
    },
    forward = function(x, z = NULL, return_embeddings = FALSE) {
      if(return_embeddings && !pass_re) stop("return_embeddings is not supported for this custom module. Add a return_embeddings argument to the forward method of your custom module.")
      args = list(x)
      if(has_embeddings) args = append(args, list(z))
      if(pass_re) args = append(args, list(return_embeddings = return_embeddings))
      return(do.call(self$custom, args))
    }
  )
  return(adapter())
}


build_dnn = function(model_properties) {
  if(!is.null(model_properties$custom)) return(build_custom(model_properties, classname = "DNN"))

  input = model_properties$input
  output = model_properties$output
  hidden = model_properties$hidden
  activation = model_properties$activation
  bias = model_properties$bias
  dropout = model_properties$dropout
  embeddings = model_properties$embeddings

  layers = list()
  if(is.null(hidden)) {
    if(is.null(embeddings)) layers[[1]] = torch::nn_linear(input, out_features = output, bias = bias)
    else layers[[1]] = torch::nn_linear(input+sum(sapply(embeddings$args, function(a) a$dim)), out_features = output, bias = bias)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden)+1 != length(bias)) bias = rep(bias, (length(hidden)+1))
    if(length(hidden) != length(dropout)) dropout = rep(dropout,length(hidden))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        if(is.null(embeddings)) layers[[1]] = torch::nn_linear(input, out_features = hidden[1], bias = bias[1])
        else layers[[1]] = torch::nn_linear(input+sum(sapply(embeddings$args, function(a) a$dim)), out_features = hidden[1], bias = bias[1])
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i])
      }
      counter = counter + 1
      #layers[[counter]] = torch::nn_batch_norm1d(num_features = hidden[[i]])
      #counter = counter + 1
      layers[[counter]]<- get_activation_layer(activation[i])

      counter = counter+1
      if(dropout[i]>0) {
        layers[[counter]] = torch::nn_dropout(dropout[i])
        counter = counter+1
      }
    }

    if(!is.null(output)) layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = output, bias = bias[i+1])
  }
  self = NULL
  if(!is.null(embeddings)) {
    net_embed <- torch::nn_module(
      classname = "DNN",
      initialize = function() {
        for(i in 1:length(embeddings$dims)) {
          self[[paste0("e_", i)]] = torch::nn_embedding(embeddings$dims[i], embeddings$args[[i]]$dim  )
        }
        for(i in 1:length(layers)) {
          self[[paste0("l_",i)]] = layers[[i]]
        }
        self$has_embeddings = TRUE
      },
      forward = function(input_hidden, input_embeddings, return_embeddings = FALSE) {
        n_em = length(embeddings$dims)
        embeds =
          lapply(1:n_em, function(j) {
          return( torch::torch_squeeze( self[[paste0("e_", j)]](input_embeddings[,j,drop=FALSE]) ,2))
        })

        x = torch::torch_cat(c(embeds, input_hidden ), 2L)
        for(i in 1:length(layers)) {
          x = self[[paste0("l_",i)]](x)
          if(return_embeddings) {
            if(i == (length(layers)-1)) {
              return(x)
            }
          }
        }
        return(x)
      }
    )
    net = net_embed()

    # set weights if provided by function
    for(i in 1:length(embeddings$dims)) {
      if(!is.null(embeddings$args[[i]]$weights)) net[[paste0("e_", i)]]$weight$set_data( torch::torch_tensor(embeddings$args[[i]]$weights,
                                                                                                             dtype = net[[paste0("e_", i)]]$weight$dtype))
    }

    # turn-off gradient if desired
    # set weights if provided by function
    for(i in 1:length(embeddings$dims)) {
      if(!(embeddings$args[[i]]$train)) net[[paste0("e_", i)]]$weight$requires_grad = FALSE
    }

  } else {
    create_dnn <- torch::nn_module(
      classname = "DNN",
      initialize = function() {
        for(i in 1:length(layers)) {
          self[[paste0("l_",i)]] = layers[[i]]
        }
        self$has_embeddings = FALSE
      },
      forward = function(x, return_embeddings = FALSE) {
        for(i in 1:length(layers)) {
          x = self[[paste0("l_",i)]](x)
          if(return_embeddings) {
            if(i == (length(layers)-1)) {
              return(x)
            }
          }
        }
        return(x)
      }
    )
    net <- create_dnn()
  }
  return(net)
}


build_cnn <- function(model_properties) {
  if(!is.null(model_properties$custom)) return(build_custom(model_properties, classname = "CNN"))

  input_shape = model_properties$input
  output_shape = model_properties$output
  architecture = model_properties$architecture

  input_dim <- length(input_shape) - 1
  net_layers = list()
  counter <- 1
  flattened <- FALSE
  transfer <- FALSE
  flattened_counter = 0L
  for(layer in architecture) {
    if(inherits(layer, "transfer")) {
      if(input_dim != 2) stop("The pretrained models only work for 2D convolutions.")

      if(input_shape[1] != 3) layer$rgb <- FALSE

      transfer_model <- get_pretrained_model(layer$name, layer$pretrained, layer$rgb)

      if(!layer$rgb) replace_first_conv_layer(transfer_model, input_shape[1])

      if(layer$freeze) transfer_model <- freeze_weights(transfer_model)

      transfer <- TRUE
      input_shape <- get_transfer_output_shape(layer$name)
    } else if(inherits(layer, "conv")) {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_conv1d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]),
                                      torch::nn_conv2d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]),
                                      torch::nn_conv3d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]))
      counter <- counter+1

      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = layer[["n_kernels"]],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = layer[["dilation"]])

      if(layer[["normalization"]]) {
        net_layers[[counter]] <- switch(input_dim,
                                        torch::nn_batch_norm1d(input_shape[1]),
                                        torch::nn_batch_norm2d(input_shape[1]),
                                        torch::nn_batch_norm3d(input_shape[1]))
        counter <- counter+1
      }

      net_layers[[counter]] <- get_activation_layer(layer[["activation"]])
      counter <- counter+1

      if(layer[["dropout"]] > 0) {
        net_layers[[counter]] <- switch(input_dim,
                                        torch::nn_dropout(layer[["dropout"]]),
                                        torch::nn_dropout2d(layer[["dropout"]]),
                                        torch::nn_dropout3d(layer[["dropout"]]))
        counter <- counter+1
      }

    } else if(inherits(layer, "maxPool")) {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_max_pool1d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]),
                                      torch::nn_max_pool2d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]),
                                      torch::nn_max_pool3d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]))
      counter <- counter+1

      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = layer[["dilation"]])

    } else if(inherits(layer, "avgPool")) {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_avg_pool1d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]),
                                      torch::nn_avg_pool2d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]),
                                      torch::nn_avg_pool3d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]))
      counter <- counter+1

      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = rep(1, input_dim))

    } else if(inherits(layer, "linear")) {
      if(!flattened) {
        net_layers[[counter]] <- torch::nn_flatten()
        flattened_counter = counter
        counter <- counter+1
        input_shape <- prod(input_shape)
        flattened <- T
      }

      net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, out_features = layer[["n_neurons"]], bias = layer[["bias"]])
      input_shape <- layer[["n_neurons"]]
      counter <- counter+1

      if(layer[["normalization"]]) {
        net_layers[[counter]] <- torch::nn_batch_norm1d(layer[["n_neurons"]])

        counter <- counter+1
      }

      net_layers[[counter]] <- get_activation_layer(layer[["activation"]])
      counter <- counter+1

      if(layer[["dropout"]] > 0) {
        net_layers[[counter]] <- torch::nn_dropout(layer[["dropout"]])
        counter <- counter+1
      }
    }
  }

  if(!flattened) {
    net_layers[[counter]] <- torch::nn_flatten()
    flattened_counter = counter
    counter <- counter+1
    input_shape <- prod(input_shape)
  }

  if(!is.null(output_shape)) net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, out_features = output_shape)

  self = NULL
  create_cnn <- torch::nn_module(
    classname = "CNN",
    initialize = function() {
      for(i in 1:length(net_layers)) {
        self[[paste0("l_",i)]] = net_layers[[i]]
      }
      self$flattened_counter = flattened_counter
    },
    forward = function(x, return_embeddings = FALSE) {
      for(i in 1:length(net_layers)) {
        x = self[[paste0("l_",i)]](x)
        if(return_embeddings) {
          if(i == self$flattened_counter) return(x)
        }
      }
      return(x)
    }
  )
  net <- create_cnn()

  if(transfer) {
    net <- replace_classifier(transfer_model, net)
  }

  return(net)
}

build_mmn <- function(model_properties) {
  self = NULL
  create_mmn <- torch::nn_module(
    classname = "MMN",
    initialize = function(subModules, fusion) {
      self$subModules <- torch::nn_module_list(subModules)
      self$fusion <- fusion
    },
    forward = function(input, return_embeddings = FALSE) {
      i <- 1
      fusion_input <- list()
      for(j in 1:length(self$subModules)) {
        if(inherits(self$subModules[[j]], "DNN") && self$subModules[[j]]$has_embeddings) {
          fusion_input <- append(fusion_input, list(self$subModules[[j]](input[[i]], input[[i+1]])))
          i <- i + 2
        } else {
          fusion_input <- append(fusion_input, list(self$subModules[[j]](input[[i]])))
          i <- i + 1
        }
      }
      return(self$fusion(torch::torch_cat(fusion_input, dim = 2L), return_embeddings = return_embeddings))
    }
  )

  subModules <- lapply(model_properties$subModules, build_model)

  fusion_input <- 0
  for(i in 1:length(subModules)) {
    if(!is.null(model_properties$subModules[[i]]$custom)) {
      # output_dim is enforced by check_custom(in_mmn = TRUE), no dummy forward necessary
      # (a dummy forward could fail for custom modules that expect non-float inputs)
      fusion_input <- fusion_input + model_properties$subModules[[i]]$custom$output_dim
    } else if(inherits(subModules[[i]], "DNN") && subModules[[i]]$has_embeddings) {
      tempX <- torch::torch_rand(c(2, model_properties$subModules[[i]]$input))
      tempZ <- lapply(model_properties$subModules[[i]]$embeddings$dims, function(x) {
        return(torch::torch_randint(1,x+1,c(2,1), dtype = torch::torch_long()))
      })
      tempZ <- torch::torch_cat(tempZ,2)
      fusion_input <- fusion_input + dim(subModules[[i]](tempX, tempZ))[2]
    } else {
      temp <- torch::torch_rand(c(2, model_properties$subModules[[i]]$input))
      fusion_input <- fusion_input + dim(subModules[[i]](temp))[2]
    }
  }

  model_properties$fusion$input <- fusion_input

  fusion <- build_dnn(model_properties$fusion)

  net <- create_mmn(subModules, fusion)

  return(net)
}





TNNEncoderLayer <- torch::nn_module(
  "TNNEncoderLayer",

  initialize = function(d_model, nhead, dim_feedforward, dropout) {
    self$attn_dropout = dropout
    self$nhead = as.integer(nhead)
    self$head_dim = as.integer(d_model %/% nhead)
    self$in_proj_weight = torch::nn_parameter(torch::torch_empty(3L * d_model, d_model))
    self$in_proj_bias = torch::nn_parameter(torch::torch_zeros(3L * d_model))
    torch::nn_init_xavier_uniform_(self$in_proj_weight)
    self$out_proj = torch::nn_linear(d_model, d_model)
    self$linear1 = torch::nn_linear(d_model, dim_feedforward)
    self$linear2 = torch::nn_linear(dim_feedforward, d_model)
    self$norm1 = torch::nn_layer_norm(d_model)
    self$norm2 = torch::nn_layer_norm(d_model)
    self$drop1 = torch::nn_dropout(dropout)
    self$drop2 = torch::nn_dropout(dropout)
    self$drop_ff = torch::nn_dropout(dropout)
  },

  # post-norm, as nn_transformer_encoder_layer(norm_first = FALSE); keep: [B, 1, T, T], TRUE = may attend
  forward = function(x, keep) {
    b = x$size(1)
    n = x$size(2)
    qkv = torch::torch_chunk(torch::nnf_linear(x, self$in_proj_weight, self$in_proj_bias), 3L, dim = 3L)
    heads = lapply(qkv, function(z) z$view(c(b, n, self$nhead, self$head_dim))$transpose(2, 3))
    o = torch::torch_scaled_dot_product_attention(heads[[1]], heads[[2]], heads[[3]], attn_mask = keep,
                                                  dropout_p = if(self$training) self$attn_dropout else 0)
    o = self$out_proj(o$transpose(2, 3)$reshape(c(b, n, self$nhead * self$head_dim)))
    x = self$norm1(x + self$drop1(o))
    self$norm2(x + self$drop2(self$linear2(self$drop_ff(torch::nnf_relu(self$linear1(x))))))
  }
)

# Token order: modalities, column tokens (group by group), register, cls. Batch layout of the inputs:
# modality streams, then value and known (observed) per token group, then, if the response is
# tokenized, the fixed validation mask.
TNN <- torch::nn_module(
  "TNN",

  initialize = function(subModules, projections, has_embeddings, props) {
    d = props$emb_dim
    self$subModules = torch::nn_module_list(subModules)
    self$projections = torch::nn_module_list(projections)
    self$has_embeddings = has_embeddings
    self$n_modalities = length(subModules)
    self$n_modal_inputs = length(subModules) + sum(has_embeddings)
    self$group_sizes = vapply(props$groups, function(g) length(g$columns), integer(1))
    self$response = props$response
    self$use_register = props$register
    self$use_cls = props$cls
    if(length(props$groups) > 0) {
      G = length(props$groups)
      self$col_emb = torch::nn_embedding(sum(self$group_sizes), d)
      self$val_w = torch::nn_parameter(torch::torch_empty(G, d)$normal_(0, 1))
      self$val_b = torch::nn_parameter(torch::torch_zeros(G, d))
      self$unknown = torch::nn_parameter(torch::torch_empty(G, d)$normal_(0, 1))
    }
    if(props$register) self$register = torch::nn_parameter(torch::torch_zeros(1L, 1L, d)$normal_(0, 0.02))
    if(props$cls) {
      self$cls = torch::nn_parameter(torch::torch_zeros(1L, 1L, d)$normal_(0, 0.02))
      self$head = torch::nn_linear(d, props$y_dim)
    }
    self$layers = torch::nn_module_list(lapply(seq_len(props$layers), function(i)
      TNNEncoderLayer(d, props$heads, props$ff_dim, props$dropout)))
    if(!is.null(props$response)) {
      S = self$group_sizes[props$response]
      bound = 1 / sqrt(d)
      self$out_w = torch::nn_parameter(torch::torch_empty(S, d)$uniform_(-bound, bound))
      self$out_b = torch::nn_parameter(torch::torch_empty(S)$uniform_(-bound, bound))
      self$always = torch::nn_buffer(torch::torch_tensor(props$masking$always, dtype = torch::torch_bool()))
      self$random = torch::nn_buffer(torch::torch_tensor(props$masking$random, dtype = torch::torch_bool()))
      self$never = torch::nn_buffer(torch::torch_tensor(props$masking$never, dtype = torch::torch_bool()))
    }
    self$keep = torch::nn_buffer(torch::torch_tensor(props$attention, dtype = torch::torch_bool()))
  },

  modality_tokens = function(modal) {
    if(self$n_modalities == 0) return(NULL)
    i = 1L
    tokens = vector("list", self$n_modalities)
    for(j in seq_len(self$n_modalities)) {
      if(self$has_embeddings[j]) {
        h = self$subModules[[j]](modal[[i]], modal[[i + 1L]])
        i = i + 2L
      } else {
        h = self$subModules[[j]](modal[[i]])
        i = i + 1L
      }
      tokens[[j]] = self$projections[[j]](h)$unsqueeze(2)
    }
    torch::torch_cat(tokens, dim = 2L)
  },

  # values: list of [B, K] float with 0 where unknown (a NA would give NaN gradients through v * w);
  # known: list of [B, K] bool. Unknown tokens are never keys.
  encode = function(mtok, values, known) {
    B = if(is.null(mtok)) values[[1]]$size(1) else mtok$size(1)
    dev = self$keep$device
    parts = list()
    pads = list()
    if(!is.null(mtok)) {
      parts = list(mtok)
      pads = list(torch::torch_zeros(c(B, self$n_modalities), dtype = torch::torch_bool(), device = dev))
    }
    offset = 0L
    for(g in seq_along(values)) {
      K = self$group_sizes[g]
      ids = torch::torch_arange(offset + 1L, offset + K, dtype = torch::torch_long(), device = dev)
      offset = offset + K
      v = values[[g]]$unsqueeze(3) * self$val_w[g, ] + self$val_b[g, ]
      v = torch::torch_where(known[[g]]$unsqueeze(3), v, self$unknown[g, ])
      parts[[length(parts) + 1L]] = self$col_emb(ids)$unsqueeze(1L) + v
      pads[[length(pads) + 1L]] = known[[g]]$logical_not()
    }
    for(extra in c("register", "cls")) {
      if(!isTRUE(self[[paste0("use_", extra)]])) next
      parts[[length(parts) + 1L]] = self[[extra]]$expand(c(B, 1L, -1L))
      pads[[length(pads) + 1L]] = torch::torch_zeros(c(B, 1L), dtype = torch::torch_bool(), device = dev)
    }
    x = torch::torch_cat(parts, dim = 2L)
    keep = self$keep$unsqueeze(1L)$logical_and(torch::torch_cat(pads, dim = 2L)$logical_not()$unsqueeze(2L))$unsqueeze(2L)
    for(i in seq_along(self$layers)) x = self$layers[[i]](x, keep)
    x
  },

  response_logits = function(x) {
    start = self$n_modalities + sum(self$group_sizes[seq_len(self$response - 1L)]) + 1L
    h = x[, start:(start + self$group_sizes[self$response] - 1L), ]
    (h * self$out_w$unsqueeze(1L))$sum(dim = 3L) + self$out_b$unsqueeze(1L)
  },

  split_input = function(input) {
    m = self$n_modal_inputs
    G = length(self$group_sizes)
    list(modal = input[seq_len(m)],
         values = lapply(seq_len(G), function(g) input[[m + 2L * g - 1L]]),
         known = lapply(seq_len(G), function(g) input[[m + 2L * g]]$to(dtype = torch::torch_bool())))
  },

  # Training draws the hidden cells per minibatch here, because the input depends on them: a hidden
  # cell enters as unknown. Evaluation (cito's validation loop) uses the fixed mask passed as last input.
  forward = function(input, return_embeddings = FALSE) {
    s = self$split_input(input)
    mtok = self$modality_tokens(s$modal)
    if(is.null(self$response)) {
      x = self$encode(mtok, s$values, s$known)
      h = x[, x$size(2), ]
      return(if(return_embeddings) h else self$head(h))
    }
    r = self$response
    obs = s$known[[r]]
    hidden = if(self$training) {
      p = torch::torch_rand(c(obs$size(1), 1L), device = obs$device)
      rnd = torch::torch_rand(obs$shape, device = obs$device) < p
      self$always$unsqueeze(1L)$logical_or(self$random$unsqueeze(1L)$logical_and(rnd))
    } else {
      input[[length(input)]]$to(dtype = torch::torch_bool())
    }
    hidden = hidden$logical_and(obs)
    s$known[[r]] = obs$logical_and(hidden$logical_not())$logical_and(self$never$logical_not()$unsqueeze(1L))
    list(pred = self$response_logits(self$encode(mtok, s$values, s$known)), hidden = hidden)
  }
)

build_tnn <- function(model_properties) {
  d = model_properties$emb_dim
  subModules = list()
  projections = list()
  has_embeddings = logical(0)
  for(p in model_properties$subModules) {
    if(is.null(p$custom)) {
      p$output = d
      net = build_model(p)
      proj = torch::nn_identity()
    } else {
      net = build_model(p)
      proj = if(p$custom$output_dim == d) torch::nn_identity() else torch::nn_linear(p$custom$output_dim, d)
    }
    subModules = c(subModules, list(net))
    projections = c(projections, list(proj))
    has_embeddings = c(has_embeddings, inherits(net, "DNN") && isTRUE(net$has_embeddings))
  }
  TNN(subModules, projections, has_embeddings, model_properties)
}
