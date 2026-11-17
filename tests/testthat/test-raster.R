source("utils.R")

# Fixtures (written with GDAL): 100 x 80 pixels, 10 m, origin 600000 / 5300800, three bands, every
# value encodes its pixel, v = 10000 * band + 100 * row + col with 0-based row and col.
raster_file = function(name) if(name == "raster.tif") system.file("extdata", "raster.tif", package = "cito") else
  test_path("fixtures", "raster", name)

expected = function(x0, y0, size, value = function(b, r, c) 10000 * b + 100 * r + c, fill = 0, nodata = NULL) {
  out = array(fill, c(length(x0), 3, size, size))
  for(i in seq_along(x0)) for(b in 1:3) for(r in 0:(size - 1)) for(c in 0:(size - 1)) {
    row = y0[i] + r
    col = x0[i] + c
    if(row < 0 || row > 79 || col < 0 || col > 99) next
    v = value(b, row, col)
    out[i, b, r + 1, c + 1] = if(!is.null(nodata) && v == nodata) fill else v
  }
  out
}

# window corners in pixels -> centre coordinates of a 10 x 10 pixel (100 m) window
centres = function(x0, y0) cbind(600000 + 10 * (x0 + 5), 5300800 - 10 * (y0 + 5))

read_all = function(X, index = seq_len(length(X))) {
  ds = cito:::cito_dataset(X, data_augmentation = NULL, num_threads = 2L)
  torch::as_array(ds$.getbatch(index)[[1]])
}

x0 = c(10L, 0L, 90L, -4L, 95L, 200L, 47L)
y0 = c(20L, 0L, 70L, -3L, 75L, 10L, 33L)

test_that("windows are exact for every layout, at the edges and outside", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")

  cases = list(
    list(f = "raster.tif", value = function(b, r, c) 10000 * b + 100 * r + c, nodata = NULL),
    list(f = "strips_band_float.tif", value = function(b, r, c) 10000 * b + 100 * r + c, nodata = 10203),
    # GDAL wrote NoData = 0 into the integer fixtures; only the int16 one actually contains a 0
    list(f = "tiled_int16.tif", value = function(b, r, c) 10000 * b + 100 * r + c - 20000, nodata = 0),
    list(f = "point.tif", value = function(b, r, c) 10000 * b + 100 * r + c, nodata = NULL))
  for(cs in cases) {
    X = suppressWarnings(raster_patches(raster_file(cs$f), centres(x0, y0), extent = 100, fill = -1))
    expect_equal(X$size, c(10L, 10L))
    expect_equal(X$info$geotransform, c(600000, 10, 0, 5300800, 0, -10), info = cs$f)
    got = read_all(X)
    expect_equal(got, expected(x0, y0, 10, cs$value, fill = -1, nodata = cs$nodata), info = cs$f)
  }
})

test_that("center and scale are applied per band, fill is left alone", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  X = suppressWarnings(raster_patches(raster_file("raster.tif"), centres(x0, y0), extent = 100, center = c(1, 2, 3), scale = c(10, 100, 1000), fill = -1))
  e = expected(x0, y0, 10, fill = -1)
  for(b in 1:3) e[, b, , ][e[, b, , ] != -1] = (e[, b, , ][e[, b, , ] != -1] - b) / 10^b
  expect_equal(read_all(X), e, tolerance = 1e-6)
})

test_that("prefetched batches are the batches that were asked for, shuffled or not", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  set.seed(1)
  n = 50L
  xs = sample(-3:95, n, TRUE)
  ys = sample(-3:75, n, TRUE)
  X = raster_patches(raster_file("raster.tif"), centres(xs, ys), extent = 100)
  e = expected(xs, ys, 10)
  for(shuffle in c(TRUE, FALSE)) {
    dl = cito:::get_data_loader(X, torch::torch_tensor(matrix(seq_len(n))), batch_size = 8L, shuffle = shuffle,
                                from_folder = TRUE, num_threads = 2L)
    for(epoch in 1:2) {
      seen = integer(0)
      coro::loop(for(b in dl) {
        idx = as.integer(torch::as_array(b[[2]]))
        seen = c(seen, idx)
        expect_equal(torch::as_array(b[[1]]), e[idx, , , , drop = FALSE])
      })
      expect_setequal(seen, seq_len(n))
      if(!shuffle) expect_equal(seen, seq_len(n))
    }
  }
})

test_that("a TIFF without georeferencing is addressed in pixels", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  skip_if_not_installed("tiff")
  f = tempfile(fileext = ".tif")
  img = matrix(seq_len(30 * 20), 20, 30, byrow = TRUE)
  tiff::writeTIFF(img / 65535, f, bits.per.sample = 16L)
  X = raster_patches(f, cbind(c(5, 15), c(5, 10)), extent = 4)
  got = read_all(X)
  expect_equal(got[1, 1, , ], img[4:7, 4:7])
  expect_equal(got[2, 1, , ], img[9:12, 14:17])
  # a file that changed its dimensions since raster_patches() is refused, not read at stale offsets
  cito:::raster_close(f)
  tiff::writeTIFF(img[1:10, 1:10] / 65535, f, bits.per.sample = 16L)
  expect_error(read_all(X), "has changed")
  cito:::raster_close(f)
})

test_that("cnn and mmn train, predict and survive saveRDS on raster inputs", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  set.seed(2)
  n = 60L
  coords = cbind(runif(n, 600100, 600900), runif(n, 5300100, 5300700))
  X = raster_patches(raster_file("raster.tif"), coords, extent = 100, center = 20000, scale = 10000)
  Y = matrix(rnorm(n), ncol = 1)
  arch = create_architecture(conv(2), maxPool(), linear(5))
  fit = cnn(X = X, Y = Y, loss = "mse", architecture = arch, epochs = 2L, batchsize = 16L, plot = FALSE, verbose = FALSE)
  p = predict(fit, newdata = X[1:7])
  expect_equal(dim(p), c(7L, 1L))
  expect_equal(predict(fit, newdata = X[1:7]), p)
  expect_error(continue_training(fit, epochs = 1L, verbose = FALSE), NA)
  f = tempfile(fileext = ".rds")
  saveRDS(fit, f)
  expect_equal(predict(readRDS(f), newdata = X[1:7]), p, tolerance = 1e-6)
  named = raster_patches(coords = `rownames<-`(coords[1:3, ], c("a", "b", "c")), like = X)
  expect_equal(rownames(predict(fit, newdata = named)), c("a", "b", "c"))
  # new sites without like =: other preprocessing warns, another window shape is refused
  expect_warning(predict(fit, newdata = raster_patches(raster_file("raster.tif"), coords[1:3, ], extent = 100)), "like")
  expect_warning(predict(fit, newdata = raster_patches(coords = coords[1:3, ], like = X)), NA)
  expect_error(predict(fit, newdata = raster_patches(raster_file("raster.tif"), coords[1:3, ], extent = 80, center = 20000, scale = 10000)), "windows of")
  expect_error(continue_training(fit, X = X[1:30], Y = Y[1:30, , drop = FALSE], epochs = 1L, verbose = FALSE), NA)
  expect_error(continue_training(fit, X = raster_patches(raster_file("raster.tif"), coords[1:30, ], extent = 80, center = 20000, scale = 10000),
                                 Y = Y[1:30, , drop = FALSE], epochs = 1L, verbose = FALSE), "windows of")
  expect_error(cnn(X = X[1:40], Y = Y, architecture = arch, epochs = 1L, plot = FALSE, verbose = FALSE), "equal number")
  expect_error(cnn(X = X, Y = Y, architecture = arch, epochs = 1L, num_workers = 2L, plot = FALSE, verbose = FALSE), NA)

  env = data.frame(a = rnorm(n))
  m = mmn(Y ~ cnn(X = X, architecture = arch) + dnn(~a, data = env),
          dataList = list(Y = Y, X = X, env = env), loss = "mse", epochs = 2L, batchsize = 16L, plot = FALSE, verbose = FALSE)
  pm = predict(m, newdata = list(X = X[1:5], env = env[1:5, , drop = FALSE]))
  expect_equal(dim(pm), c(5L, 1L))
})

test_that("windows entirely outside the raster are flagged, a zero scale is refused", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  f = raster_file("raster.tif")
  expect_warning(raster_patches(f, cbind(c(11.5, 600500), c(48.9, 5300400)), extent = 100), "1 of 2 windows")
  expect_warning(raster_patches(f, centres(c(-9L, 99L), c(-9L, 79L)), extent = 100), NA)
  expect_error(raster_patches(f, centres(10L, 10L), extent = 100, scale = 0), "scale")
})

test_that("raster_patches() refuses inputs that would put constants or non-finite values into the network", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  f = raster_file("raster.tif")
  xy = centres(c(10L, 20L), c(10L, 20L))
  expect_error(raster_patches(f, xy, extent = 100, center = NA_real_), "center")
  expect_error(raster_patches(f, xy, extent = 100, fill = NaN), "fill")
  expect_error(raster_patches(f, cbind(Inf, 5300400), extent = 100), "finite")
  expect_error(raster_patches(f, xy, extent = 1e7), "map units")
  expect_error(raster_patches(f, xy, extent = 100, transform = function(x) torch::torch_cat(list(x, x), 1L)), "transform")
  X = raster_patches(f, xy, extent = 100)
  expect_error(X[c(1, NA)], "NA")
  expect_error(raster_patches(f, xy, extent = 50, like = X), "drop raster, extent")
  # a UTM coordinate on a degree-like grid lies far outside: a warning, not an integer overflow
  expect_warning(raster_patches(f, cbind(1e15, 5300400), extent = 100), "1 of 1 windows")
})

test_that("float32 NoData is matched at float precision, and window positions round half up", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  # written by GDAL with NoData -3.4e38, which float32 cannot hold exactly; the cell at column 1, row 0 is NoData
  X = raster_patches(raster_file("nodata_float32.tif"), cbind(1.5, 2.5), extent = 1)
  expect_true(is.nan(torch::as_array(cito:::raw_windows(X, 1L))[1, 1, 1, 1]))
  # sites on consecutive pixel centres with an even window move the window by one pixel each
  X = raster_patches(raster_file("raster.tif"), cbind(600405 + 10 * 0:15, 5300405), extent = 40)
  expect_equal(diff(cito:::raster_offsets(X, 1:16)$x0), rep(1L, 15))
})

test_that("loaders share one reader per raster", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  X = raster_patches(raster_file("raster.tif"), centres(c(10L, 20L, 30L), c(10L, 20L, 30L)), extent = 100, scale = 7)
  before = length(ls(cito:::raster_readers))
  for(i in 1:20) read_all(X)
  expect_lte(length(ls(cito:::raster_readers)), before + 1L)
  expect_equal(read_all(X), read_all(X))
  # the cache keeps at most 16 readers per process
  for(k in 1:20) cito:::raster_reader(X, k)
  expect_lte(length(ls(cito:::raster_readers)), 16L)
})

test_that("transform, standardize, fill and the missing-pixel channel are applied in that order", {
  skip_if_no_torch()
  skip_if_not(isTRUE(cito:::cito_decoder_available()[3]), "cito built without libtiff")
  f = raster_file("raster.tif")
  inside = centres(c(10L, 40L, 70L), c(10L, 30L, 50L))
  raw = read_all(suppressWarnings(raster_patches(f, inside, extent = 100)))
  X = raster_patches(f, inside, extent = 100, transform = function(x) torch::torch_log(x), standardize = TRUE)
  l = log(raw)
  m = apply(l, 2, mean)
  s = apply(l, 2, stats::sd)
  expect_equal(X$center, unname(m), tolerance = 1e-5)
  expect_equal(X$scale, unname(s), tolerance = 1e-4)
  got = read_all(X)
  for(b in 1:3) expect_equal(got[, b, , ], (l[, b, , ] - m[b]) / s[b], tolerance = 1e-4)
  # a window half outside: the outside pixels are fill and flagged in the extra channel
  X = suppressWarnings(raster_patches(f, centres(-5L, 10L), extent = 100, center = 10000, scale = 100, fill = -9, na_channel = TRUE))
  got = read_all(X)
  expect_equal(as.array(X), got)
  expect_equal(dim(got), c(1, 4, 10, 10))
  expect_true(all(got[1, 1:3, , 1:5] == -9))
  expect_true(all(got[1, 4, , 1:5] == 1) && all(got[1, 4, , 6:10] == 0))
  expect_equal(got[1, 1, 1, 6], (10000 * 1 + 100 * 10 + 0 - 10000) / 100)
  # like: new coordinates with the preprocessing of the training windows
  Xn = raster_patches(coords = centres(20L, 20L), like = X)
  expect_equal(Xn$center, X$center)
  expect_true(Xn$na_channel)
  expect_error(raster_patches(f, inside, extent = 100, center = 1, standardize = TRUE), "either")
  expect_error(raster_patches(f, inside, extent = 100, center = c(1, 2)), "per channel")
  # a transform that changes the channels sizes center/scale and the network
  X = raster_patches(f, inside, extent = 100, transform = function(x) x[, 1:2, , ] / x[, 3:3, , ], standardize = TRUE)
  expect_equal(length(X$center), 2)
  fit = cnn(X = X, Y = matrix(rnorm(3)), epochs = 1L, batchsize = 3L, plot = FALSE, verbose = FALSE,
            architecture = create_architecture(conv(2), linear(4)))
  expect_equal(fit$model_properties$input[1], 2)
})
