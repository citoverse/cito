\donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)

## Data
shapes <- simulate_shapes(320, 28)
X <- shapes$data
Y <- shapes$labels

## Architecture
architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))

## Build and train network
cnn.fit <- cnn(X, Y, architecture, loss = "softmax", epochs = 50, validation = 0.1, lr = 0.05)

# Structure of Neural Network
print(cnn.fit)
}
}
