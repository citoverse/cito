get_data_loader_rnn<- function(X, Y, lag, batch_size, shuffle = TRUE, x_dtype, y_dtype){
  torch.dataset <- torch::dataset(
    name = "dataset",
    initialize = function(X, Y, lag) {
      self$lag <- lag
      self$X <- torch::torch_tensor(X[c(1:nrow(X)),,drop = FALSE])
      self$Y <- torch::torch_tensor(Y[c((lag+1):nrow(Y)),,drop = FALSE])
    },

    .getitem = function(index) {
      list(
        x = self$X[index:(self$lag+index),],
        y = self$Y[index,]
      )

    },
    .length = function() {
      self$Y$size()[[1]]
    }
  )

  ds <- torch.dataset(X = X, Y = Y, lag = lag)
  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)

  return(dl)
}


build_rnn <- function(type, num_layers, input_size, output_size,  hidden_size, bias, dropout){

  rnn_net <- torch::nn_module(
    initialize = function(type= type, input_size = input_size,
                          output_size = output_size, hidden_size = hidden_size,
                          num_layers = num_layers, dropout = dropout, bias = bias) {

      self$type <- type
      self$num_layers <- num_layers

      self$rnn <- if (self$type == "gru") {
        torch::nn_gru(
          input_size = input_size,
          hidden_size = hidden_size,
          num_layers = num_layers,
          dropout = dropout,
          bias = bias,
          batch_first = TRUE
        )
      } else {
        torch::nn_lstm(
          input_size = input_size,
          hidden_size = hidden_size,
          num_layers = num_layers,
          dropout = dropout,
          bias = bias,
          batch_first = TRUE
        )
      }

      self$out <- torch::nn_linear(hidden_size, output_size, bias = bias)

    },

    forward = function(x) {

      x <- self$rnn(x)[[1]]
      x <- x[ , dim(x)[2], ]
      return(self$out(x))
    }

  )

  net<- rnn_net(type= type, input_size = input_size,
                output_size = output_size, hidden_size = hidden_size,
                num_layers = num_layers, dropout = dropout, bias = bias)
  return(net)
}
