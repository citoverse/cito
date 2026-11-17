#' Image patches cut from one large raster
#'
#' Describes, for every observation, a square window of a large GeoTIFF centered on the
#' observation's coordinates. The result is passed as \code{X} to \code{\link{cnn}} (or to a
#' \code{cnn()} term of \code{\link{mmn}}) in place of an array of images or a folder. The
#' windows are never extracted to disk: each batch is read from the raster when it is needed,
#' and with libtiff 4.5 or newer the next batch is read in the background while the current one trains.
#'
#' @param raster Path to a GeoTIFF. Any number of bands; 8, 16 and 32 bit integer and 32 and
#'   64 bit float samples. Values are read as stored and converted to 32 bit float: integers above
#'   2^24 are rounded, and 64 bit values beyond the float range become missing.
#' @param coords Matrix or data.frame with two columns, x and y, giving the center of each
#'   window in the coordinate reference system of \code{raster} (which is not checked; a warning
#'   is given when windows lie entirely outside the raster). For a TIFF without georeferencing, x and
#'   y are pixel units from the top-left corner of the image, y pointing down, so that the center of
#'   the top-left pixel is (0.5, 0.5).
#' @param extent Side length of the window in map units, rounded to whole pixels at the
#'   resolution of \code{raster} (halves round up). The window is placed on the pixel grid of the
#'   raster: its center lies at most half a pixel from \code{coords}, and pixel values are never
#'   interpolated.
#' @param center,scale \code{NULL}, or one value per channel (or a single value for all channels).
#'   Every channel is transformed as \code{(value - center) / scale}, after \code{transform}.
#' @param standardize Logical. \code{TRUE} estimates \code{center} and \code{scale} as the mean and
#'   standard deviation of every channel over all windows of \code{coords} (after \code{transform},
#'   without missing pixels). This reads every window once. Build it on the training coordinates
#'   only and pass \code{like} for validation, test and prediction sites, so that their statistics
#'   do not enter the scaling.
#' @param transform \code{NULL} or a function applied to every batch before \code{center} and
#'   \code{scale}: it receives a float tensor \code{[windows, bands, height, width]} in the units of
#'   the raster, with \code{NaN} for missing pixels, and returns a tensor with the same first
#'   dimension, e.g. \code{function(x) torch::torch_log1p(x)} or a band ratio. Channels and spatial
#'   size may change. It must treat every window on its own: statistics over the batch would make
#'   predictions depend on the batch size.
#' @param fill Value for missing pixels (outside the raster, NoData, \code{NaN}, and non-finite results of
#'   \code{transform}), in the units after \code{center} and \code{scale}: with \code{standardize = TRUE},
#'   the default 0 imputes the mean; without it, 0 is a value in the units of the raster, and
#'   \code{na_channel = TRUE} lets the network tell it from an observed 0.
#' @param na_channel Logical. \code{TRUE} appends one channel that is 1 where any channel of the pixel is
#'   missing and 0 otherwise, so that the network can tell an imputed pixel from an observed one.
#' @param like A \code{citoraster} whose raster, extent and preprocessing are reused for new
#'   \code{coords}, e.g. to predict at new sites with the \code{center} and \code{scale} estimated on
#'   the training windows. Only \code{coords} may be given together with \code{like}.
#'
#' @details
#' Reading speed depends almost entirely on how the raster is stored. Every read decompresses whole
#' storage blocks. A raster stored in strips (the GeoTIFF default) has blocks of full-width rows, so a
#' window of 100 x 100 pixels from a 37800 pixel wide raster decompresses 100 rows of 37800 pixels; a tiled
#' raster has square blocks, and a window decompresses only the few tiles it overlaps. The gain grows with
#' the width of the raster: in our benchmarks a tiled copy was about 2 to 7 times faster for rasters 10000
#' to 12000 pixels wide and roughly 35 to 125 times faster for one 37800 pixels wide, and tiles of about the window size were
#' best (128 x 128 pixel tiles about twice as fast as 256 x 256 for windows of 100 pixels). Whether a file
#' is tiled is shown by \code{X$info$tiled} and \code{X$info$block}, and a large raster in strips gives a
#' warning. Converting it is a single step, done once before training:
#' \code{terra::writeRaster(terra::rast("x.tif"), "x_tiled.tif", gdal = c("TILED=YES", "BLOCKXSIZE=128",
#' "BLOCKYSIZE=128", "COMPRESS=DEFLATE", "PREDICTOR=2"))}, with \code{"PREDICTOR=3"} for floating point
#' rasters, or outside R, with less memory, \code{gdal_translate -co TILED=YES -co BLOCKXSIZE=128
#' -co BLOCKYSIZE=128 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=IF_SAFER x.tif x_tiled.tif}.
#' The values are unchanged; only their arrangement in the file differs. Only the first image of the file
#' is read (no overviews).
#' The number of reader threads is \code{num_threads} of \code{\link{cnn}}; \code{\link{mmn}} uses 4.
#'
#' Missing pixels are those outside the raster, \code{NaN}, and pixels equal to the NoData value of
#' the GDAL_NODATA tag. Masks, alpha bands (read as an ordinary band), and scale and offset metadata
#' are not applied; apply an offset or scale factor with \code{center} and \code{scale} or \code{transform}.
#'
#' Every batch is preprocessed in this order: the raw values are read (missing pixels as \code{NaN}),
#' \code{transform} is applied, then \code{(value - center) / scale}, then missing pixels are set to
#' \code{fill} and the \code{na_channel} is appended. Unlike images read from a folder, values are not
#' rescaled to the unit interval: rasters carry physical quantities, so normalization is left to these
#' arguments. Preprocessing is part of the \code{citoraster}, so a model fitted on it predicts on the same
#' scale; for new sites pass \code{like}. \code{predict()} refuses windows of another shape and warns
#' when the preprocessing differs from that of the training windows.
#'
#' The raster is opened once per session and kept open. A file replaced by another one with the same
#' modification time is not noticed; a file whose dimensions changed is refused.
#'
#' @return An object of class \code{citoraster}, which can be subset by observation with \code{[}.
#' @example /inst/examples/raster_patches-example.R
#' @export
raster_patches = function(raster, coords, extent, center = NULL, scale = NULL, standardize = FALSE,
                          transform = NULL, fill = 0, na_channel = FALSE, like = NULL) {
  coords = as.matrix(coords)
  if(ncol(coords) != 2L || !is.numeric(coords) || !all(is.finite(coords))) stop("coords must have two numeric columns, x and y, with finite values.", call. = FALSE)
  if(nrow(coords) == 0L) stop("coords has no rows.", call. = FALSE)
  if(!is.null(like)) {
    checkmate::assert_class(like, "citoraster")
    given = c(raster = !missing(raster), extent = !missing(extent), center = !missing(center), scale = !missing(scale),
              standardize = !missing(standardize), transform = !missing(transform), fill = !missing(fill), na_channel = !missing(na_channel))
    if(any(given)) stop("like = reuses the raster, extent and preprocessing of a citoraster; drop ", paste(names(given)[given], collapse = ", "), ".", call. = FALSE)
    like$coords = coords
    warn_outside(like)
    return(like)
  }
  checkmate::qassert(raster, "S1")
  checkmate::assert_file_exists(raster)
  checkmate::qassert(extent, "N1(0,)")
  checkmate::qassert(fill, "N1()")
  checkmate::qassert(center, c("0", "N+()"))
  checkmate::qassert(scale, c("0", "N+()"))
  checkmate::qassert(standardize, "B1")
  checkmate::qassert(na_channel, "B1")
  checkmate::assert_function(transform, null.ok = TRUE)
  if(any(scale == 0)) stop("scale must not be 0.", call. = FALSE)
  if(standardize && (!is.null(center) || !is.null(scale))) stop("Give either center and scale, or standardize = TRUE.", call. = FALSE)
  if(!isTRUE(cito_decoder_available()[3])) stop("cito was installed without libtiff, which raster_patches() needs.", call. = FALSE)
  info = cito_raster_info(normalizePath(raster))
  gt = info$geotransform
  if(gt[3] != 0 || gt[5] != 0) stop(raster, " is rotated; only north-up rasters are supported.", call. = FALSE)
  size = round_half_up(extent / abs(gt[c(2, 6)]))
  if(any(size < 1)) stop("extent is smaller than one pixel of ", raster, call. = FALSE)
  if(prod(size) * info$bands > .Machine$integer.max)
    stop("extent = ", extent, " gives windows of ", size[1], " x ", size[2], " pixels. Is extent in the map units of the raster (pixel size ",
         signif(abs(gt[2]), 3), ")?", call. = FALSE)
  if(!info$tiled && as.numeric(info$width) * info$height > 4e6)
    warning(raster, " is stored in strips, so every window decompresses full-width rows. ",
            "A tiled copy reads windows much faster (2 to 125 times in our benchmarks, more for wider rasters) and is made in one step, ",
            "e.g. gdal_translate -co TILED=YES -co BLOCKXSIZE=128 -co BLOCKYSIZE=128 -co COMPRESS=DEFLATE in.tif out.tif; see ?raster_patches.", call. = FALSE)
  out = structure(list(raster = normalizePath(raster), coords = coords, size = as.integer(size), center = NULL, scale = NULL,
                       transform = transform, fill = fill, na_channel = na_channel, info = info),
                  class = "citoraster")
  warn_outside(out)
  probe = raw_windows(out, 1L, transform = TRUE)
  if(!inherits(probe, "torch_tensor") || probe$dim() != 4L || probe$size(1) != 1L)
    stop("transform must return a tensor [windows, channels, height, width] with one window per input window.", call. = FALSE)
  channels = probe$size(2)
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
  out$shape = dim(preprocess_windows(out, raw_windows(out, 1L)))[-1]
  out
}

# R's round() goes to the even neighbour on ties, so sites on pixel centres would alternate between
# two window offsets and neighbouring sites would share a window
round_half_up = function(v) floor(v + 0.5)

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
  t = torch::torch_tensor_from_buffer(buf, shape = c(length(o$x0), x$info$bands, x$size[2], x$size[1]), dtype = "float")$clone()
  if(transform && !is.null(x$transform)) t = x$transform(t)
  t
}

channel_stats = function(x, channels) {
  s = s2 = k = numeric(channels)
  # about 64 MB of raw windows per chunk, whatever the window size
  per = max(1L, floor(2^24 / (prod(x$size) * x$info$bands)))
  for(idx in split(seq_len(length(x)), ceiling(seq_len(length(x)) / per))) {
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
  if(!is.null(x$transform)) t = x$transform(t)$to(dtype = torch::torch_float32())
  C = t$size(2)
  t = (t - torch::torch_tensor(x$center)$view(c(1L, C, 1L, 1L))) / torch::torch_tensor(x$scale)$view(c(1L, C, 1L, 1L))
  miss = torch::torch_isfinite(t)$logical_not()
  t = t$masked_fill(miss, x$fill)
  if(x$na_channel) t = torch::torch_cat(list(t, miss$any(dim = 2L, keepdim = TRUE)$to(dtype = t$dtype)), 2L)
  t
}

# newdata of a fitted model must give windows of the training shape, and should carry the training preprocessing
check_raster_newdata = function(new, old, input = NULL) {
  if(!inherits(new, "citoraster")) return(invisible(TRUE))
  if(!is.null(input) && !is.null(new$shape) && !identical(as.integer(new$shape), as.integer(input)))
    stop("newdata gives windows of ", paste(new$shape, collapse = " x "), " (channels x height x width), the model was fitted on ",
         paste(input, collapse = " x "), ". Build newdata with raster_patches(coords = ..., like = <training windows>).", call. = FALSE)
  if(inherits(old, "citoraster") &&
     (!isTRUE(all.equal(unclass(old)[c("center", "scale", "fill", "na_channel")], unclass(new)[c("center", "scale", "fill", "na_channel")])) ||
      !identical(deparse(old$transform), deparse(new$transform))))
    warning("newdata is preprocessed differently from the training windows (center, scale, fill, transform or na_channel). ",
            "For new sites use raster_patches(coords = ..., like = <training windows>).", call. = FALSE)
  invisible(TRUE)
}

#' @param x A \code{citoraster}.
#' @param ... Not used.
#' @return \code{as.array()}: the windows as the network receives them, an array
#'   \code{[windows, channels, height, width]} with all preprocessing applied. It reads every window
#'   of \code{x}, so subset first, e.g. \code{as.array(X[1:16])}.
#' @rdname raster_patches
#' @export
as.array.citoraster = function(x, ...) torch::as_array(preprocess_windows(x, raw_windows(x, seq_len(length(x)))))

#' @export
length.citoraster = function(x) nrow(x$coords)

#' @export
`[.citoraster` = function(x, i, ..., drop = TRUE) {
  x$coords = x$coords[i, , drop = FALSE]
  if(anyNA(x$coords)) stop("NA in the index of a citoraster.", call. = FALSE)
  x
}

#' @export
print.citoraster = function(x, ...) {
  shape = if(is.null(x$shape)) c(x$info$bands, x$size[2], x$size[1]) else x$shape
  cat(sprintf("%d windows of %d x %d pixels x %d channels from %s (%d bands)\n", length(x), shape[3], shape[2], shape[1],
              basename(x$raster), x$info$bands))
  cat(sprintf("  preprocessing: %s(value - center) / scale, missing pixels = %g%s\n", if(is.null(x$transform)) "" else "transform, then ",
              x$fill, if(x$na_channel) ", plus a missing-pixel channel" else ""))
  cat("  center:", format(signif(x$center, 4)), "\n  scale: ", format(signif(x$scale, 4)), "\n")
  invisible(x)
}

# One reader (a worker thread plus one TIFF handle per reader thread) per raster and setting, shared
# by every loader: a reader per loader held them open until gc(), 125 descriptors after 150 predict() calls.
# The process id is part of the key because a forked child inherits the pointer but not the worker thread.
raster_readers = new.env()

raster_reader = function(x, nthreads) {
  key = paste(x$raster, format(as.numeric(file.mtime(x$raster)), digits = 17), nthreads, Sys.getpid())
  if(is.null(raster_readers[[key]])) {
    now = cito_raster_info(x$raster)
    if(!identical(now[c("width", "height", "bands")], x$info[c("width", "height", "bands")]) || !isTRUE(all.equal(now$geotransform, x$info$geotransform)))
      stop(basename(x$raster), " has changed since raster_patches() was called; call it again.", call. = FALSE)
    raster_close(keep = 15L)
    raster_readers[[key]] = list(reader = cito_raster_open(x$raster, nthreads, rep(0, x$info$bands), rep(1, x$info$bands), NaN),
                                 path = x$raster, pid = Sys.getpid(), opened = as.numeric(Sys.time()))
  }
  raster_readers[[key]]$reader
}

# Drops all but the `keep` most recently opened cached readers of this process (of `path` only, if given).
# A loader that still holds a reader keeps it alive until the loader is collected.
raster_close = function(path = NULL, keep = 0L) {
  entries = mget(ls(raster_readers), envir = raster_readers)
  mine = vapply(entries, function(e) e$pid == Sys.getpid() && (is.null(path) || e$path == normalizePath(path)), logical(1))
  keys = names(entries)[mine][order(vapply(entries[mine], `[[`, numeric(1), "opened"))]
  drop = keys[seq_len(max(0L, length(keys) - keep))]
  rm(entries)
  rm(list = drop, envir = raster_readers)
  # the TIFF handle closes when the reader is collected; Windows cannot replace a file that is still open
  if(length(drop)) gc(verbose = FALSE)
  invisible(length(drop))
}

# top-left pixel of every window; clamped so that coordinates far outside stay outside instead of becoming NA
raster_offsets = function(x, index) {
  gt = x$info$geotransform
  xy = x$coords[index, , drop = FALSE]
  off = function(v) as.integer(pmin(pmax(round_half_up(v), -2^30), 2^30))
  list(x0 = off((xy[, 1] - gt[1]) / gt[2] - x$size[1] / 2),
       y0 = off((xy[, 2] - gt[4]) / gt[6] - x$size[2] / 2))
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
