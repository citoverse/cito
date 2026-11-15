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
#' @param center,scale \code{NULL}, or one value per band (or a single value for all bands).
#'   Every band is transformed as \code{(value - center) / scale} while it is read.
#' @param fill Value for pixels outside the raster, NoData and NaN.
#'
#' @details
#' Reading speed depends almost entirely on how the raster is stored. A tiled GeoTIFF reads
#' a window from the few tiles it overlaps; a raster stored in strips (the GeoTIFF default)
#' decompresses every full-width row the window touches, which is two to three orders of
#' magnitude slower for large rasters. Write rasters for training as tiled files, e.g.
#' \code{terra::writeRaster(r, "x.tif", gdal = c("TILED=YES", "COMPRESS=ZSTD", "PREDICTOR=2"))}.
#' The number of reader threads is \code{num_threads} of \code{\link{cnn}}.
#'
#' Unlike images read from a folder, values are not rescaled to the unit interval: rasters carry physical
#' quantities, so normalization is left to \code{center} and \code{scale}.
#'
#' @return An object of class \code{citoraster}, which can be subset by observation with \code{[}.
#' @example /inst/examples/raster_patches-example.R
#' @export
raster_patches = function(raster, coords, extent, center = NULL, scale = NULL, fill = 0) {
  checkmate::qassert(raster, "S1")
  checkmate::assert_file_exists(raster)
  checkmate::qassert(extent, "N1(0,)")
  checkmate::qassert(fill, "N1")
  checkmate::qassert(center, c("0", "N+()"))
  checkmate::qassert(scale, c("0", "N+()"))
  if(any(scale == 0)) stop("scale must not be 0.", call. = FALSE)
  if(!isTRUE(cito_decoder_available()[3])) stop("cito was installed without libtiff, which raster_patches() needs.", call. = FALSE)
  coords = as.matrix(coords)
  if(ncol(coords) != 2L || !is.numeric(coords) || anyNA(coords)) stop("coords must have two numeric columns, x and y, without NA.", call. = FALSE)
  info = cito_raster_info(normalizePath(raster))
  gt = info$geotransform
  if(gt[3] != 0 || gt[5] != 0) stop(raster, " is rotated; only north-up rasters are supported.", call. = FALSE)
  size = round(extent / abs(gt[c(2, 6)]))
  if(any(size < 1)) stop("extent is smaller than one pixel of ", raster, call. = FALSE)
  per_band = function(v, default) {
    if(is.null(v)) return(rep(default, info$bands))
    if(length(v) == 1L) return(rep(v, info$bands))
    if(length(v) != info$bands) stop("center and scale need one value per band (", info$bands, ") or a single value.", call. = FALSE)
    v
  }
  if(!info$tiled && as.numeric(info$width) * info$height > 4e6)
    warning(raster, " is stored in strips, so every window decompresses full-width rows. ",
            "A tiled copy reads windows up to 1000x faster, see ?raster_patches.", call. = FALSE)
  out = structure(list(raster = normalizePath(raster), coords = coords, size = as.integer(size),
                       center = per_band(center, 0), scale = per_band(scale, 1), fill = fill, info = info),
                  class = "citoraster")
  # a coordinate system mismatch puts every window outside, and the model then trains on constants
  o = raster_offsets(out, seq_len(nrow(coords)))
  outside = o$x0 + size[1] <= 0 | o$x0 >= info$width | o$y0 + size[2] <= 0 | o$y0 >= info$height
  if(any(outside)) warning(sum(outside), " of ", length(outside), " windows lie entirely outside ", basename(raster),
                           " and consist of fill only. Are coords in the coordinate system of the raster?", call. = FALSE)
  out
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
  invisible(x)
}

# One reader (a worker thread plus one TIFF handle per reader thread) per raster and setting, shared
# by every loader: a reader per loader held them open until gc(), 125 descriptors after 150 predict() calls.
raster_readers = new.env()

raster_reader = function(x, nthreads) {
  key = paste(x$raster, file.mtime(x$raster), nthreads, paste(x$center, collapse = ","), paste(x$scale, collapse = ","), x$fill)
  if(is.null(raster_readers[[key]])) raster_readers[[key]] = cito_raster_open(x$raster, nthreads, x$center, x$scale, x$fill)
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
