\donttest{
if(torch::torch_is_installed()){
library(cito)

# Convolutional layers with different n_kernels and kernel_sizes
c1 <- conv(n_kernels = 8, kernel_size = 5)
c2 <- conv(n_kernels = 16, kernel_size = 3)

# Linear layer
l <- linear(n_neurons = 100)

# MaxPooling layer
mP <- maxPool(kernel_size = 2)

# Create the architecture by using the created layers
# Change the defaults with which the not assigned layer parameters will be filled e.g.
# change default dropout to different values for linear and convolutional layer
# only change the default normalization for linear layers
# change default activation of both linear and convolutional layers to 'selu'
architecture <- create_architecture(c1, c1, mP, c2, c2, mP, l,
                                    default_dropout = list(linear=0.6, conv=0.4),
                                    default_normalization = list(linear=TRUE),
                                    default_activation = "selu")

# See how the finished CNN would look like for specific input and output shapes
print(architecture, c(3,128,128), 10)

# To use predefined architectures  use the transfer() layer
alexnet <- transfer("alexnet")

# No other linear layers are used after the transfer layer:
# The cnn() function will replace the linear classifier of the
# alexnet architecture with a single linear output layer
architecture <- create_architecture(alexnet)
print(architecture, c(3,128,128), 10)

# Some linear layers are used after the transfer layer:
# The cnn() function will replace the linear classifier of the alexnet architecture
# with the specified linear layers + an output layer that matches the output dimensions
architecture <- create_architecture(alexnet, linear(300), linear(100))
print(architecture, c(3,128,128), 10)
}
}
