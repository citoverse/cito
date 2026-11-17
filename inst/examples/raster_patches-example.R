\donttest{
if(torch::torch_is_installed() && cito:::cito_decoder_available()[["tiff"]]){
library(cito)

# a small georeferenced raster stands in for a large satellite scene
f = system.file("extdata", "raster.tif", package = "cito")
coords = cbind(x = runif(200, 600100, 600900), y = runif(200, 5300100, 5300700))
# log-transform, then standardize every band with the mean and sd over the windows;
# missing pixels become 0 (the mean) and are flagged in an extra channel
X = raster_patches(f, coords, extent = 200, transform = function(x) torch::torch_log1p(x),
                   standardize = TRUE, na_channel = TRUE)
X
# the first four windows as the network sees them: [windows, channels, height, width]
dim(as.array(X[1:4]))
Y = matrix(rnorm(200), ncol = 1)

nn.fit = cnn(X = X, Y = Y, loss = "mse", epochs = 2L, batchsize = 32L,
             architecture = create_architecture(conv(4), maxPool(), linear(10)))

# new sites, with the preprocessing of the training windows
new_coords = cbind(x = runif(5, 600100, 600900), y = runif(5, 5300100, 5300700))
predict(nn.fit, newdata = raster_patches(coords = new_coords, like = X))
}
}
