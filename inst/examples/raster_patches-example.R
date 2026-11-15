\donttest{
if(torch::torch_is_installed()){
library(cito)

# a small georeferenced raster stands in for a large satellite scene
f = system.file("extdata", "raster.tif", package = "cito")
coords = cbind(x = runif(200, 600100, 600900), y = runif(200, 5300100, 5300700))
X = raster_patches(f, coords, extent = 200)
X
Y = matrix(rnorm(200), ncol = 1)

nn.fit = cnn(X = X, Y = Y, loss = "mse", epochs = 2L, batchsize = 32L,
             architecture = create_architecture(conv(4), maxPool(), linear(10)))
predict(nn.fit, newdata = X[1:5])
}
}
