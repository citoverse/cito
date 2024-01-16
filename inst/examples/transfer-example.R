\donttest{
if(torch::torch_is_installed()){
library(cito)

# Creates a "transfer" "citolayer" object that later tells the cnn() function that
# the alexnet architecture and its pretrained weights should be used, but none
# of the weights are frozen
alexnet <- transfer(name="alexnet", pretrained=TRUE, freeze=FALSE)

# Creates a "transfer" "citolayer" object that later tells the cnn() function that
# the resnet18 architecture and its pretrained weights should be used.
# Also all weights except from the linear layer at the end are frozen (and
# therefore not changed during training)
resnet18 <- transfer(name="resnet18", pretrained=TRUE, freeze=TRUE)
}
}
