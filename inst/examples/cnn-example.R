\donttest{
if(torch::torch_is_installed()){
library(cito)

# Example workflow in cito

device <- ifelse(torch::cuda_is_available(), "cuda", "cpu")

## Data
### We generate our own data:
### 320 images (3x50x50) of either rectangles or ellipsoids
shapes <- cito:::simulate_shapes(n=320, size=50, channels=3)
X <- shapes$data
Y <- shapes$labels

## Architecture
### Declare the architecture of the CNN
### Note that the output layer is added automatically by cnn()
architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))

## Build and train network
### cross-entropy (softmax link) for classification
cnn.fit <- cnn(X, Y, architecture, loss = "cross-entropy",
               epochs = 2, validation = 0.1, lr = 0.05, device=device)

## The training loss is below the baseline loss but at the end of the
## training the loss was still decreasing, so continue training for another 5
## epochs
cnn.fit <- continue_training(cnn.fit, epochs = 5)

# Structure of Neural Network
print(cnn.fit)

# Plot Neural Network
plot(cnn.fit)

## Convergence can be tested via the analyze_training function
analyze_training(cnn.fit)

## Predictions, and the output of the last hidden layer (embeddings)
predict(cnn.fit, newdata = X[1:5,,,,drop = FALSE], type = "class")
dim(predict(cnn.fit, newdata = X[1:5,,,,drop = FALSE], return_embeddings = TRUE))

## Data augmentation: random 90 degree rotations and flips of every image,
## drawn anew in each epoch
cnn.aug <- cnn(X, Y, architecture, loss = "cross-entropy", epochs = 2,
               data_augmentation = list("rotate90", "flip"), device = device)

## Images on disk: X is a folder (jpeg, png or tiff), read batch by batch
## during training, so the data do not have to fit into memory
if(requireNamespace("tiff", quietly = TRUE)) {
  folder <- file.path(tempdir(), "shapes")
  dir.create(folder, showWarnings = FALSE)
  for(i in 1:50) {
    tiff::writeTIFF(aperm(X[i,,,], c(2, 3, 1)), file.path(folder, sprintf("%03d.tiff", i)))
  }
  # files are matched to Y in the order of list.files(folder)
  cnn.disk <- cnn(folder, Y[1:50], architecture, loss = "cross-entropy",
                  epochs = 1, device = device)
  predict(cnn.disk, newdata = folder, type = "class")[1:5]
  unlink(folder, recursive = TRUE)
}

## Windows cut from one large georeferenced raster around each observation
## are an input as well, see ?raster_patches

## Transfer learning
### With the transfer() function we can use predefined architectures with pretrained weights
transfer_architecture <- create_architecture(transfer("resnet18"))
resnet <- cnn(X, Y, transfer_architecture, loss = "cross-entropy",
              epochs = 1, validation = 0.1, lr = 0.05, device=device)
print(resnet)
plot(resnet)
}
}
