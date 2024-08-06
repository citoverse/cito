\donttest{
if(torch::torch_is_installed()){
library(cito)

device <- ifelse(torch::cuda_is_available(), "cuda", "cpu")

set.seed(222)

## Data
shapes <- cito:::simulate_shapes(320, 28)
X <- shapes$data
Y <- shapes$labels

## Architecture
architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))

## Build and train network
cnn.fit <- cnn(X, Y, architecture, loss = "softmax", epochs = 50, validation = 0.1, lr = 0.05, device=device)

# Weights of neural network
coef(cnn.fit)
}
}
