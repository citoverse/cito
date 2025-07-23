\donttest{
if(torch::torch_is_installed()){
library(cito)

# Example workflow in cito

device <- ifelse(torch::cuda_is_available(), "cuda", "cpu")

## Simulated data
shapes <- cito:::simulate_shapes(n=320, size=50, channels=3)
X_cnn <- shapes$data
X_dnn <- matrix(runif(320*3),320,3)
Y <- (as.integer(shapes$labels)-1)*2 + 0.5*X_dnn[,1] + 0.3*X_dnn[,2] - 0.8*X_dnn[,3]

data <- list(Y=Y, X_cnn=X_cnn, X_dnn=X_dnn)

## Architecture of the CNN
architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))

## Build and train network
mmn.fit <- mmn(Y ~ dnn(~., data=X_dnn, hidden = c(100,100,100), activation = "relu") + cnn(X=X_cnn, architecture = architecture),
               dataList = data, loss = "mse", epochs = 50, validation = 0.1, lr = 0.05, device=device)

## If the loss is still decreasing you can continue training for additional epochs:
mmn.fit <- continue_training(mmn.fit, epochs = 50)
}
}
