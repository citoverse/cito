\dontrun{
library(cito)

# create optimizer object
opt <- config_optimizer(type = "adagrad", lr_decay = 1e-04, weight_decay = 0.1 )

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris, optimizer = opt)

}
