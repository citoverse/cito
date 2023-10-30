\donttest{
if(torch::torch_is_installed()){
library(cito)

# A linear layer where all available parameters are assigned
# No value will be overwritten by 'create_architecture()'
layer1 <- linear(100, TRUE, "relu", FALSE, 0.5)

# A linear layer where only the activation function is assigned
# n_neurons, bias, normalization and dropout are filled with the defaults
# passed to the 'create_architecture()' function
layer2 <- linear(activation="selu")
}
}
