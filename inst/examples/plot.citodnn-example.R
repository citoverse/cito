\donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)
validation_set<- sample(c(1:nrow(datasets::iris)),25)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])

# accumulated local effects
plot(nn.fit)

# architecture
if(requireNamespace("ggraph", quietly = TRUE)) plot(nn.fit, plot_type = "arch")
}
}
