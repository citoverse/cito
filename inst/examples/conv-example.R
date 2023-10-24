\donttest{
  if(torch::torch_is_installed()){
    library(cito)

    # A convolutional layer where all available parameters are assigned
    # No value will be overwritten by 'create_architecture()'
    layer1 <- conv(10, 3, 1, 0, 1, TRUE, "relu", FALSE, 0.5)

    # A convolutional layer where only the activation function is assigned
    # n_kernels, kernel_size, stride, padding, dilation, bias,
    # normalization and dropout are filled with the defaults
    # passed to the 'create_architecture()' function
    layer2 <- conv(activation="selu")
  }
}
