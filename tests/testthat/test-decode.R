library(cito)
source("utils.R")

# On a CRAN Linux build all three libraries are present, so the compiled decoder is the path
# that actually runs. Nothing else in the suite asserts that it agrees with the R readers it
# replaced, and a disagreement there changes the number of input channels a model is fitted with.

GREY_TRNS_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAAAAABX3VL4AAAAAnRSTlMAAHaTzTgAAAAOSURBVHicY+ASYRDRAAAA2ABbxl+YxwAAAABJRU5ErkJggg=="

# no base64 dependency in the package, and the fixture is 94 bytes
base64enc_decode = function(x) {
  alpha = strsplit("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", "")[[1]]
  chars = strsplit(sub("=+$", "", x), "")[[1]]
  bits = unlist(lapply(match(chars, alpha) - 1L, function(v) as.integer(intToBits(v))[6:1]))
  n = length(bits) %/% 8L
  as.raw(vapply(seq_len(n), function(i) sum(bits[((i-1L)*8L+1L):(i*8L)] * 2L^(7:0)), numeric(1)))
}

make_fixtures = function(dir) {
  set.seed(42)
  rgb = array(runif(24*20*3), dim = c(24, 20, 3))
  grey = matrix(runif(24*20), 24, 20)
  writeables = list(
    "rgb.jpg"       = function(f) jpeg::writeJPEG(rgb, f, quality = 1),
    "grey.jpg"      = function(f) jpeg::writeJPEG(grey, f, quality = 1),
    "rgb.png"       = function(f) png::writePNG(rgb, f),
    "grey.png"      = function(f) png::writePNG(grey, f),
    "greyalpha.png" = function(f) png::writePNG(array(c(grey, matrix(1, 24, 20)), dim = c(24, 20, 2)), f),
    "rgba.png"      = function(f) png::writePNG(array(c(rgb, matrix(1, 24, 20)), dim = c(24, 20, 4)), f),
    "rgb8.tif"      = function(f) tiff::writeTIFF(rgb, f, bits.per.sample = 8L),
    "rgb16.tif"     = function(f) tiff::writeTIFF(rgb, f, bits.per.sample = 16L),
    "grey16.tif"    = function(f) tiff::writeTIFF(grey, f, bits.per.sample = 16L)
  )
  for(nm in names(writeables)) writeables[[nm]](file.path(dir, nm))
  # a greyscale PNG carrying a tRNS chunk decodes to grey+alpha, and must not then be
  # expanded to RGB. png::writePNG cannot write tRNS, so the file is stored as bytes.
  writeBin(base64enc_decode(GREY_TRNS_B64), file.path(dir, "grey_trns.png"))
  sort(list.files(dir, full.names = TRUE))
}

reference = function(path) {
  if(grepl("\\.tiff?$", path)) {
    a = suppressWarnings(tiff::readTIFF(path))
    if(length(dim(a)) == 2) dim(a) = c(dim(a), 1L)
  } else {
    a = torchvision::base_loader(path)
  }
  torch::torch_tensor(a)$permute(c(3, 1, 2))
}

testthat::test_that("the compiled decoder matches the R readers", {
  skip_if_no_torch()
  testthat::skip_if(!any(cito:::cito_decoder_available()), "built without any image library")

  dir = file.path(tempdir(), "cito-decode")
  dir.create(dir, showWarnings = FALSE)
  on.exit(unlink(dir, recursive = TRUE))
  files = make_fixtures(dir)

  for(f in files) {
    res = cito:::cito_decode_batch(f, 1L)
    if(!isTRUE(res$ok)) next                       # format not compiled in, the R reader has it
    dt = if(res$bits == 8L) "uint8" else "int16"
    t = torch::torch_tensor_from_buffer(res$data, shape = res$dim, dtype = dt)$to(dtype = torch::torch_float32())
    if(res$bits == 16L) t = torch::torch_where(t < 0, t + 65536, t)
    t = t$permute(c(1, 4, 2, 3))/(2^res$bits - 1)
    ref = reference(f)
    testthat::expect_equal(dim(t)[-1], dim(ref), info = basename(f))
    testthat::expect_lt(as.numeric(torch::torch_max(torch::torch_abs(t[1, , , ] - ref))), 1e-5)
  }
})

testthat::test_that("the compiled path and the R fallback agree exactly", {
  skip_if_no_torch()
  testthat::skip_if(!any(cito:::cito_decoder_available()), "built without any image library")

  dir = file.path(tempdir(), "cito-decode-equiv")
  dir.create(dir, showWarnings = FALSE)
  on.exit(unlink(dir, recursive = TRUE))
  files = make_fixtures(dir)
  # a file with no extension: the compiled path sniffs magic bytes, so the R path must too
  file.copy(file.path(dir, "rgb.png"), file.path(dir, "noextension"))
  files = sort(list.files(dir, full.names = TRUE))

  y = torch::torch_tensor(matrix(1))
  read_with = function(path, compiled) {
    ds = cito:::cito_dataset(path, y, data_augmentation = NULL)
    if(!compiled) ds$decoder = c(jpeg = FALSE, png = FALSE, tiff = FALSE)
    ds$.getbatch(1L)[[1]]
  }
  for(f in files) {
    a = read_with(f, TRUE)
    b = read_with(f, FALSE)
    # a difference here means the fitted architecture depends on which libraries were installed
    testthat::expect_identical(dim(a), dim(b), info = basename(f))
    testthat::expect_lt(as.numeric(torch::torch_max(torch::torch_abs(a - b))), 1e-6)
  }
})

testthat::test_that("the decoder reports per-file errors and refuses layouts it cannot read", {
  skip_if_no_torch()
  testthat::skip_if(!any(cito:::cito_decoder_available()), "built without any image library")

  dir = file.path(tempdir(), "cito-decode-err")
  dir.create(dir, showWarnings = FALSE)
  on.exit(unlink(dir, recursive = TRUE))
  writeLines("not an image", file.path(dir, "notes.txt"))
  writeBin(as.raw(c(0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3, 4)), file.path(dir, "truncated.jpg"))

  testthat::expect_false(cito:::cito_decode_batch(file.path(dir, "missing.jpg"), 1L)$ok)
  testthat::expect_false(cito:::cito_decode_batch(file.path(dir, "notes.txt"), 1L)$ok)
  testthat::expect_false(cito:::cito_decode_batch(file.path(dir, "truncated.jpg"), 1L)$ok)

  # a float TIFF is a layout the compiled path hands back rather than guessing at
  tiff::writeTIFF(array(runif(8*8*3), dim = c(8, 8, 3)), file.path(dir, "f.tif"), bits.per.sample = 32L)
  testthat::expect_false(cito:::cito_decode_batch(file.path(dir, "f.tif"), 1L)$ok)
})

testthat::test_that("cnn() reads a folder whichever decoder is used", {
  skip_if_no_torch()
  dir = file.path(tempdir(), "cito-decode-cnn")
  dir.create(dir, showWarnings = FALSE)
  on.exit(unlink(dir, recursive = TRUE))
  set.seed(1)
  for(i in 1:8) jpeg::writeJPEG(array(runif(16*16*3), dim = c(16, 16, 3)),
                                file.path(dir, sprintf("i%02d.JPG", i)), quality = 1)
  arch = create_architecture(conv(2), maxPool(), linear(2))
  testthat::expect_error({
    m = cnn(X = dir, Y = runif(8), architecture = arch, loss = "mse", epochs = 1L,
            batchsize = 4L, plot = FALSE, verbose = FALSE)
    .p = predict(m, newdata = dir)
  }, NA)
})


# The layouts remote-sensing data actually arrives in. These are files rather than generated
# fixtures because writing them needs GDAL, which cito does not depend on. Between them they
# cover the cases that cost the most if they regress: a cloud-optimised GeoTIFF, whose overviews
# are extra directories and must not be mistaken for pages; float32, which is what per-pixel
# embedding products such as TESSERA distribute; signed integers, which elevation models use;
# band-interleaved planes; a genuine multi-page file; and subsampled YCbCr, whose scanlines are
# shorter than width times samples and once caused a heap over-read.
eo_fixture = function(name) testthat::test_path("fixtures", name)

testthat::test_that("the remote-sensing layouts load, and both paths agree where both can read", {
  skip_if_no_torch()
  testthat::skip_if(!any(cito:::cito_decoder_available()), "built without any image library")

  y = torch::torch_tensor(matrix(1))
  read_with = function(path, compiled) {
    ds = cito:::cito_dataset(path, y, data_augmentation = NULL)
    if(!compiled) ds$decoder = c(jpeg = FALSE, png = FALSE, tiff = FALSE)
    ds$.getbatch(1L)[[1]]
  }

  # shape and sample format the compiled path must produce
  expected = list(cog_u16.tif = c(4L, 32L, 32L), float32.tif = c(4L, 32L, 32L),
                  dem_int16.tif = c(1L, 32L, 32L), separate_u16.tif = c(4L, 32L, 32L))
  for(nm in names(expected)) {
    t = read_with(eo_fixture(nm), TRUE)
    testthat::expect_identical(dim(t)[-1], expected[[nm]], info = nm)
    testthat::expect_true(all(is.finite(as.numeric(t$reshape(-1)))), info = nm)
  }

  # a COG must not be mistaken for a multi-page file, and both paths must read it the same
  for(nm in c("cog_u16.tif", "float32.tif", "multipage.tif", "masked.tif",
              "palette.tif", "miniswhite.tif")) {
    a = read_with(eo_fixture(nm), TRUE)
    b = read_with(eo_fixture(nm), FALSE)
    testthat::expect_identical(dim(a), dim(b), info = nm)
    testthat::expect_lt(as.numeric(torch::torch_max(torch::torch_abs(a - b))), 1e-6)
  }

  # float32 and signed samples are physical quantities and must not be rescaled to [0, 1]
  f32 = read_with(eo_fixture("float32.tif"), TRUE)
  testthat::expect_lt(as.numeric(f32$min()), -1)
  dem = read_with(eo_fixture("dem_int16.tif"), TRUE)
  testthat::expect_lt(as.numeric(dem$min()), -100)
  testthat::expect_gt(as.numeric(dem$max()), 1000)

  # an internal mask is a 1-bit directory flagged FILETYPE_MASK. Counting it as a page would
  # refuse essentially every modern COG, and readTIFF(all = TRUE) chokes on the 1-bit depth.
  testthat::expect_identical(dim(read_with(eo_fixture("masked.tif"), TRUE))[-1], c(1L, 32L, 32L))

  # a palette image expands to RGB through its colormap, and MINISWHITE returns the stored
  # samples rather than inverting them, because that is what GDAL and readTIFF both do
  testthat::expect_identical(dim(read_with(eo_fixture("palette.tif"), TRUE))[-1], c(3L, 32L, 32L))
  testthat::expect_identical(dim(read_with(eo_fixture("miniswhite.tif"), TRUE))[-1], c(1L, 32L, 32L))

  # subsampled YCbCr is handed back rather than copied past the end of its scanline buffer
  testthat::expect_true(isTRUE(cito:::cito_decode_batch(eo_fixture("ycbcr.tif"), 1L)$fallback))
  # a genuine multi-page file goes to the R reader, which stacks the pages
  testthat::expect_true(isTRUE(cito:::cito_decode_batch(eo_fixture("multipage.tif"), 1L)$fallback))
})


testthat::test_that("a float32 batch survives garbage collection", {
  skip_if_no_torch()
  testthat::skip_if(!any(cito:::cito_decoder_available()), "built without any image library")

  # torch_tensor_from_buffer() borrows the raw vector, and $to(dtype) copies only when the
  # dtype changes -- for a float32 file it does not, so without an explicit clone the tensor
  # ends up aliasing memory R has collected, and the batch fills with NaN and +/-Inf.
  f = eo_fixture("float32.tif")
  ds = cito:::cito_dataset(rep(f, 4L), torch::torch_tensor(matrix(runif(4))), data_augmentation = NULL)
  b = ds$.getbatch(1:4)[[1]]
  before = as.numeric(torch::torch_max(torch::torch_abs(b)))
  invisible(lapply(1:100, function(i) runif(50000)))
  gc()
  x = as.array(b)
  testthat::expect_false(any(is.nan(x)))
  testthat::expect_true(all(is.finite(x)))
  testthat::expect_equal(max(abs(x)), before, tolerance = 1e-6)
})

testthat::test_that("signed and unsigned samples are never mixed in one batch", {
  skip_if_no_torch()
  testthat::skip_if(!any(cito:::cito_decoder_available()), "built without any image library")

  # a signed elevation model and an unsigned reflectance band in one folder is ordinary, and
  # reading them under one interpretation rescales half the batch. Which half would depend on
  # the shuffle, so the same file would differ between epochs.
  # same geometry and bit depth, differing only in sample format, so that the sample-format
  # check is what decides and not the geometry check
  u16 = file.path(tempdir(), "refl_u16.tif")
  on.exit(unlink(u16))
  tiff::writeTIFF(matrix(seq(0, 1, length.out = 32*32), 32, 32), u16, bits.per.sample = 16L)
  res = cito:::cito_decode_batch(c(eo_fixture("dem_int16.tif"), u16), 1L)
  testthat::expect_false(isTRUE(res$ok))
  testthat::expect_true(isTRUE(res$fallback))
})

testthat::test_that("layouts the R reader gets wrong are refused, not handed on", {
  skip_if_no_torch()
  y = torch::torch_tensor(matrix(1))
  # readTIFF returns different pixels on repeated reads of a subsampled YCbCr file, sometimes
  # +/-Inf. The compiled path refuses it; the fallback must refuse it too rather than inherit it.
  ds = cito:::cito_dataset(eo_fixture("ycbcr.tif"), y, data_augmentation = NULL)
  ds$decoder = c(jpeg = FALSE, png = FALSE, tiff = FALSE)
  testthat::expect_error(ds$.getbatch(1L), "YCbCr")
})
