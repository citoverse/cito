\donttest{
  if(torch::torch_is_installed()){
    library(cito)

    # A maximum pooling layer where all available parameters are assigned
    # No value will be overwritten by 'create_architecture()'
    layer1 <- maxPool(3, 1, 0, 1)

    # A maximum pooling layer where only the kernel size is assigned
    # stride, padding and dilation are filled with the defaults
    # passed to the 'create_architecture()' function
    layer2 <- maxPool(kernel_size=4)
  }
}
