\donttest{
if(torch::torch_is_installed()){
library(cito)

# A average pooling layer where all available parameters are assigned
# No value will be overwritten by 'create_architecture()'
layer1 <- avgPool(3, 1, 0)

# A average pooling layer where only the kernel size is assigned
# stride and padding are filled with the defaults
# passed to the 'create_architecture()' function
layer2 <- avgPool(kernel_size=4)
}
}
