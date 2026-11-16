#' Image patches cut from one large raster
#'
#' Describes, for every observation, a square window of a large GeoTIFF centered on the
#' observation's coordinates. The result is passed as \code{X} to \code{\link{cnn}} (or to a
#' \code{cnn()} term of \code{\link{mmn}}) in place of an array of images or a folder. The
#' windows are never extracted to disk: each batch is read from the raster when it is needed,
#' and the next batch is read in the background while the current one trains.
#'
#' @param raster Path to a GeoTIFF. Any number of bands; 8, 16 and 32 bit integer and 32 and
#'   64 bit float samples.
#' @param coords Matrix or data.frame with two columns, x and y, giving the center of each
#'   window in the coordinate reference system of \code{raster}. For a TIFF without
#'   georeferencing, x and y are pixel units from the top-left corner.
#' @param extent Side length of the window in map units, rounded to whole pixels at the
#'   resolution of \code{raster}.
#' @param center,scale \code{NULL}, or one value per channel (or a single value for all channels).
#'   Every channel is transformed as \code{(value - center) / scale}, after \code{transform}.
#' @param standardize Logical. \code{TRUE} estimates \code{center} and \code{scale} as the mean and
#'   standard deviation of every channel over all windows of \code{coords} (after \code{transform},
#'   without missing pixels). This reads every window once.
#' @param transform \code{NULL} or a function applied to every batch before \code{center} and
#'   \code{scale}: it receives a float tensor \code{[windows, bands, height, width]} in the units of
#'   the raster, with \code{NaN} for missing pixels, and returns a tensor with the same first
#'   dimension, e.g. \code{function(x) torch::torch_log1p(x)} or a band ratio. Channels and spatial
#'   size may change.
#' @param fill Value for missing pixels (outside the raster, NoData, \code{NaN}, and non-finite results of
#'   \code{transform}), in the units after \code{center} and \code{scale}: with \code{standardize = TRUE},
#'   the default 0 imputes the mean.
#' @param na_channel Logical. \code{TRUE} appends one channel that is 1 where any band of the pixel is
#'   missing and 0 otherwise, so that the network can tell an imputed pixel from an observed one.
#' @param like A \code{citoraster} whose raster, extent and preprocessing are reused for new
#'   \code{coords}, e.g. to predict at new sites with the \code{center} and \code{scale} estimated on
#'   the training windows. \code{raster}, \code{extent} and the preprocessing arguments are then ignored.
#'
#' @details
#' Reading speed depends almost entirely on how the raster is stored. A tiled GeoTIFF reads
#' a window from the few tiles it overlaps; a raster stored in strips (the GeoTIFF default)
#' decompresses every full-width row the window touches, which is two to three orders of
#' magnitude slower for large rasters. Write rasters for training as tiled files, e.g.
#' \code{terra::writeRaster(r, "x.tif", gdal = c("TILED=YES", "COMPRESS=ZSTD", "PREDICTOR=2"))}.
#' The number of reader threads is \code{num_threads} of \code{\link{cnn}}.
#'
#' Every batch is preprocessed in this order: the raw values are read (missing pixels as \code{NaN}),
#' \code{transform} is applied, then \code{(value - center) / scale}, then missing pixels are set to
#' \code{fill} and the \code{na_channel} is appended. Unlike images read from a folder, values are not
#' rescaled to the unit interval: rasters carry physical quantities, so normalization is left to these
#' arguments. Preprocessing is part of the \code{citoraster}, so a model fitted on it predicts on the same
#' scale; for new sites pass \code{like}.
#'
#' @return An object of class \code{citoraster}, which can be subset by observation with \code{[}.
#' @example /inst/examples/raster_patches-example.R
#' @export
raster_patches = function(raster, coords, extent, center = NULL, scale = NULL, standardize = FALSE,
                          transform = NULL, fill = 0, na_channel = FALSE, like = NULL) {
  coords = as.matrix(coords)
  if(ncol(coords) != 2L || !is.numeric(coords) || anyNA(coords)) stop("coords must have two numeric columns, x and y, without NA.", call. = FALSE)
  if(!is.null(like)) {
    checkmate::assert_class(like, "citoraster")
    like$coords = coords
    warn_outside(like)
    return(like)
  }
  checkmate::qassert(raster, "S1")
  checkmate::assert_file_exists(raster)
  checkmate::qassert(extent, "N1(0,)")
  checkmate::qassert(fill, "n1")
  checkmate::qassert(center, c("0", "n+()"))
  checkmate::qassert(scale, c("0", "n+()"))
  checkmate::qassert(standardize, "B1")
  checkmate::qassert(na_channel, "B1")
  checkmate::assert_function(transform, null.ok = TRUE)
  if(any(scale == 0)) stop("scale must not be 0.", call. = FALSE)
  if(standardize && (!is.null(center) || !is.null(scale))) stop("Give either center and scale, or standardize = TRUE.", call. = FALSE)
  if(!isTRUE(cito_decoder_available()[3])) stop("cito was installed without libtiff, which raster_patches() needs.", call. = FALSE)
  info = cito_raster_info(normalizePath(raster))
  gt = info$geotransform
  if(gt[3] != 0 || gt[5] != 0) stop(raster, " is rotated; only north-up rasters are supported.", call. = FALSE)
  size = round(extent / abs(gt[c(2, 6)]))
  if(any(size < 1)) stop("extent is smaller than one pixel of ", raster, call. = FALSE)
  if(!info$tiled && as.numeric(info$width) * info$height > 4e6)
    warning(raster, " is stored in strips, so every window decompresses full-width rows. ",
            "A tiled copy reads windows up to 1000x faster, see ?raster_patches.", call. = FALSE)
  out = structure(list(raster = normalizePath(raster), coords = coords, size = as.integer(size), center = NULL, scale = NULL,
                       transform = transform, fill = fill, na_channel = na_channel, info = info),
                  class = "citoraster")
  warn_outside(out)
  channels = dim(raw_windows(out, 1L, transform = TRUE))[2]
  per_channel = function(v, default) {
    if(is.null(v)) return(rep(default, channels))
    if(length(v) == 1L) return(rep(v, channels))
    if(length(v) != channels) stop("center and scale need one value per channel (", channels, ") or a single value.", call. = FALSE)
    v
  }
  if(standardize) {
    st = channel_stats(out, channels)
    center = st$mean
    scale = st$sd
  }
  out$center = per_channel(center, 0)
  out$scale = per_channel(scale, 1)
  out
}

# a coordinate system mismatch puts every window outside, and the model then trains on constants
warn_outside = function(x) {
  o = raster_offsets(x, seq_len(nrow(x$coords)))
  outside = o$x0 + x$size[1] <= 0 | o$x0 >= x$info$width | o$y0 + x$size[2] <= 0 | o$y0 >= x$info$height
  if(any(outside)) warning(sum(outside), " of ", length(outside), " windows lie entirely outside ", basename(x$raster),
                           " and consist of fill only. Are coords in the coordinate system of the raster?", call. = FALSE)
}

# windows as read, missing pixels NaN, optionally transformed
raw_windows = function(x, index, transform = FALSE, nthreads = 1L) {
  rd = raster_reader(x, nthreads)
  o = raster_offsets(x, index)
  buf = cito_raster_fetch(rd, cito_raster_submit(rd, o$x0, o$y0, x$size[1], x$size[2]))
  t = torch::torch_tensor_from_buffer(buf, shape = c(length(index), x$info$bands, x$size[2], x$size[1]), dtype = "float")$clone()
  if(transform && !is.null(x$transform)) t = x$transform(t)
  t
}

channel_stats = function(x, channels) {
  s = s2 = k = numeric(channels)
  for(idx in split(seq_len(length(x)), ceiling(seq_len(length(x)) / 256))) {
    t = raw_windows(x, idx, transform = TRUE)$transpose(1, 2)$reshape(c(channels, -1L))$to(dtype = torch::torch_float64())
    ok = torch::torch_isfinite(t)
    t = t$masked_fill(ok$logical_not(), 0)
    s = s + as.numeric(t$sum(2))
    s2 = s2 + as.numeric((t * t)$sum(2))
    k = k + as.numeric(ok$sum(2))
  }
  if(any(k < 2)) stop("standardize = TRUE: channel(s) ", paste(which(k < 2), collapse = ", "), " have fewer than two observed pixels.", call. = FALSE)
  m = s / k
  sd = sqrt(pmax(s2 / k - m^2, 0) * k / (k - 1))
  if(any(sd == 0)) {
    warning("standardize = TRUE: channel(s) ", paste(which(sd == 0), collapse = ", "), " are constant; their scale is set to 1.", call. = FALSE)
    sd[sd == 0] = 1
  }
  list(mean = m, sd = sd)
}

# transform, center/scale, fill and the missing-pixel channel, for a batch read with NaN for missing
preprocess_windows = function(x, t) {
  if(!is.null(x$transform)) t = x$transform(t)
  C = t$size(2)
  t = (t - torch::torch_tensor(x$center)$view(c(1L, C, 1L, 1L))) / torch::torch_tensor(x$scale)$view(c(1L, C, 1L, 1L))
  miss = torch::torch_isfinite(t)$logical_not()
  t = t$masked_fill(miss, x$fill)
  if(x$na_channel) t = torch::torch_cat(list(t, miss$any(dim = 2L, keepdim = TRUE)$to(dtype = t$dtype)), 2L)
  t
}

#' @export
length.citoraster = function(x) nrow(x$coords)

#' @export
`[.citoraster` = function(x, i, ..., drop = TRUE) {
  x$coords = x$coords[i, , drop = FALSE]
  x
}

#' @export
print.citoraster = function(x, ...) {
  cat(sprintf("%d windows of %d x %d pixels x %d bands from %s\n", length(x), x$size[1], x$size[2],
              x$info$bands, basename(x$raster)))
  cat(sprintf("  preprocessing: %s(value - center) / scale, missing pixels = %g%s\n", if(is.null(x$transform)) "" else "transform, then ",
              x$fill, if(x$na_channel) ", plus a missing-pixel channel" else ""))
  invisible(x)
}

# One reader (a worker thread plus one TIFF handle per reader thread) per raster and setting, shared
# by every loader: a reader per loader held them open until gc(), 125 descriptors after 150 predict() calls.
raster_readers = new.env()

raster_reader = function(x, nthreads) {
  key = paste(x$raster, file.mtime(x$raster), nthreads)
  if(is.null(raster_readers[[key]]))
    raster_readers[[key]] = cito_raster_open(x$raster, nthreads, rep(0, x$info$bands), rep(1, x$info$bands), NaN)
  raster_readers[[key]]
}

# top-left pixel of every window
raster_offsets = function(x, index) {
  gt = x$info$geotransform
  xy = x$coords[index, , drop = FALSE]
  list(x0 = as.integer(round((xy[, 1] - gt[1]) / gt[2] - x$size[1] / 2)),
       y0 = as.integer(round((xy[, 2] - gt[4]) / gt[6] - x$size[2] / 2)))
}

lazy_input = function(x) is.character(x) || inherits(x, "citoraster")

expand_input = function(x) if(is.character(x)) list.files(x, full.names = TRUE) else x

# Hands the dataset the order of the coming epoch, so that it can read the next batch while the
# current one trains. Shuffles with torch_randperm(), as torch's own sampler does, so a seed set
# with torch_manual_seed() gives the same order with and without a raster input.
order_sampler = torch::sampler(
  "cito_order_sampler",
  initialize = function(data_source, shuffle) {
    self$data_source = data_source
    self$shuffle = shuffle
  },
  .length = function() length(self$data_source),
  .iter_batch = function(batch_size) {
    n = length(self$data_source)
    o = if(self$shuffle) torch::as_array(torch::torch_randperm(n)$add(1L, 1L)$to(dtype = torch::torch_int())) else seq_len(n)
    self$data_source$order = o
    i = 1L
    function() {
      if(i > n) return(coro::exhausted())
      out = o[i:min(n, i + batch_size - 1L)]
      i <<- i + batch_size
      out
    }
  },
  .iter = function() self$.iter_batch(1L)
)
