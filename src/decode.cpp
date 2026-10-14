// One decoder for every format cito reads. Decodes into a single pre-allocated buffer at the
// file's native bit depth, so no R double array is ever built. Errors are collected per file and
// raised on the R side: nothing in here may call the R API, because it runs under OpenMP.
#include <Rcpp.h>
#include <cstdio>
#include <cstring>
#include <csetjmp>
#include <cstdlib>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
// each decoder is optional: configure defines the macro only when the library was found,
// so a machine without the -dev packages still builds and falls back to the R readers
extern "C" {
#ifdef HAVE_JPEG
#include <jpeglib.h>
#endif
#ifdef HAVE_PNG
#include <png.h>
#endif
#ifdef HAVE_TIFF
#include <tiffio.h>
#endif
}

enum Fmt { FMT_UNKNOWN = 0, FMT_JPEG, FMT_PNG, FMT_TIFF };

struct Meta {
  int h, w, c, bits, pages;
  Fmt fmt;
  int status;              // 0 ok, non-zero means msg is set
  char msg[220];
};

static void set_err(Meta& m, const char* what) {
  m.status = 1;
  std::snprintf(m.msg, sizeof(m.msg), "%s", what);
}

// returns FMT_UNKNOWN for a readable file of unknown type; *openable tells the two apart
static Fmt sniff(const char* path, bool* openable) {
  FILE* f = std::fopen(path, "rb");
  *openable = (f != NULL);
  if(!f) return FMT_UNKNOWN;
  unsigned char b[8] = {0};
  size_t n = std::fread(b, 1, 8, f);
  std::fclose(f);
  if(n < 4) return FMT_UNKNOWN;
  if(b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) return FMT_JPEG;
  if(n >= 8 && b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G') return FMT_PNG;
  // classic TIFF (42) and BigTIFF (43), both byte orders; libtiff handles either
  if((b[0] == 'I' && b[1] == 'I' && (b[2] == 42 || b[2] == 43) && b[3] == 0) ||
     (b[0] == 'M' && b[1] == 'M' && b[2] == 0 && (b[3] == 42 || b[3] == 43))) return FMT_TIFF;
  return FMT_UNKNOWN;
}

#ifdef HAVE_JPEG
/* ---- libjpeg: default error_exit calls exit(), so every call gets its own setjmp ---- */
struct jerr_mgr { struct jpeg_error_mgr pub; jmp_buf jb; char msg[JMSG_LENGTH_MAX]; };
static void jerr_exit(j_common_ptr ci) {
  jerr_mgr* e = reinterpret_cast<jerr_mgr*>(ci->err);
  (*ci->err->format_message)(ci, e->msg);
  std::longjmp(e->jb, 1);
}
static void jerr_silent(j_common_ptr, int) {}

// no non-trivial C++ object may be live across the setjmp, hence the raw FILE* and manual cleanup
static int jpeg_probe_decode(const char* path, Meta& m, unsigned char* dst) {
  struct jpeg_decompress_struct ci;
  jerr_mgr jerr;
  FILE* f = std::fopen(path, "rb");
  if(!f) { set_err(m, "cannot open file"); return 1; }
  ci.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jerr_exit;
  jerr.pub.emit_message = jerr_silent;
  if(setjmp(jerr.jb)) {
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    set_err(m, jerr.msg);
    return 1;
  }
  jpeg_create_decompress(&ci);
  jpeg_stdio_src(&ci, f);
  jpeg_read_header(&ci, TRUE);
  ci.out_color_space = (ci.num_components == 1) ? JCS_GRAYSCALE : JCS_RGB;
  if(dst == NULL) {
    m.h = ci.image_height;
    m.w = ci.image_width;
    m.c = (ci.num_components == 1) ? 1 : 3;
    m.bits = 8;
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    return 0;
  }
  jpeg_start_decompress(&ci);
  const int stride = ci.output_width * ci.output_components;
  while(ci.output_scanline < ci.output_height) {
    unsigned char* row = dst + (size_t)ci.output_scanline * stride;
    jpeg_read_scanlines(&ci, &row, 1);
  }
  jpeg_finish_decompress(&ci);
  jpeg_destroy_decompress(&ci);
  std::fclose(f);
  return 0;
}

#endif /* HAVE_JPEG */

#ifdef HAVE_PNG
/* ---- libpng ---- */
#ifndef PNG_SETJMP_SUPPORTED
#error "cito needs a libpng built with setjmp support"
#endif
// libpng's default error function prints to stderr and leaves the caller with no message,
// so the handler records what failed and only then longjmps
struct png_err_ctx { char msg[200]; };
static void cito_png_error(png_structp p, png_const_charp msg) {
  png_err_ctx* c = (png_err_ctx*) png_get_error_ptr(p);
  if(c) std::snprintf(c->msg, sizeof(c->msg), "%s", msg);
  std::longjmp(png_jmpbuf(p), 1);
}
static void cito_png_warn(png_structp, png_const_charp) {}

static int png_probe_decode(const char* path, Meta& m, unsigned char* dst) {
  png_err_ctx ctx;
  ctx.msg[0] = '\0';
  FILE* f = std::fopen(path, "rb");
  if(!f) { set_err(m, "cannot open file"); return 1; }
  png_structp p = png_create_read_struct(PNG_LIBPNG_VER_STRING, &ctx, cito_png_error, cito_png_warn);
  if(!p) { std::fclose(f); set_err(m, "png_create_read_struct failed"); return 1; }
  png_infop info = png_create_info_struct(p);
  if(!info) { png_destroy_read_struct(&p, NULL, NULL); std::fclose(f); set_err(m, "png_create_info_struct failed"); return 1; }
  png_bytep* rows = NULL;
  if(setjmp(png_jmpbuf(p))) {
    if(rows) std::free(rows);
    png_destroy_read_struct(&p, &info, NULL);
    std::fclose(f);
    set_err(m, ctx.msg[0] ? ctx.msg : "libpng error while reading");
    return 1;
  }
  png_init_io(p, f);
  png_read_info(p, info);
  png_uint_32 w = 0, h = 0;
  int depth = 0, ctype = 0;
  png_get_IHDR(p, info, &w, &h, &depth, &ctype, NULL, NULL, NULL);
  if(ctype == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(p);
  if(ctype == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(p);
  if(png_get_valid(p, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(p);
  png_read_update_info(p, info);
  const int channels = png_get_channels(p, info);
  const int out_depth = (depth == 16) ? 16 : 8;
  if(dst == NULL) {
    m.h = (int)h; m.w = (int)w; m.c = channels; m.bits = out_depth;
    png_destroy_read_struct(&p, &info, NULL);
    std::fclose(f);
    return 0;
  }
  if(depth == 16) png_set_swap(p);   // PNG is big-endian on disk, torch wants native order
  const size_t rowbytes = png_get_rowbytes(p, info);
  rows = (png_bytep*) std::malloc(sizeof(png_bytep) * h);
  if(!rows) { png_destroy_read_struct(&p, &info, NULL); std::fclose(f); set_err(m, "out of memory"); return 1; }
  for(png_uint_32 r = 0; r < h; r++) rows[r] = dst + (size_t)r * rowbytes;
  png_read_image(p, rows);
  std::free(rows);
  rows = NULL;
  png_destroy_read_struct(&p, &info, NULL);
  std::fclose(f);
  return 0;
}

#endif /* HAVE_PNG */

#ifdef HAVE_TIFF
/* ---- libtiff: the error handler is global, so the message lands in a thread-local slot ---- */
#if defined(__GNUC__) || defined(__clang__)
#define CITO_TLS __thread
#else
#define CITO_TLS thread_local
#endif
static CITO_TLS char tiff_msg[200];
static void tiff_err_handler(const char*, const char* fmt, va_list ap) {
  std::vsnprintf(tiff_msg, sizeof(tiff_msg), fmt, ap);
}

static int tiff_probe_decode(const char* path, Meta& m, unsigned char* dst) {
  tiff_msg[0] = '\0';
  TIFF* t = TIFFOpen(path, "r");
  if(!t) { set_err(m, tiff_msg[0] ? tiff_msg : "cannot open TIFF"); return 1; }
  uint32_t w = 0, h = 0;
  uint16_t spp = 1, bps = 8, fmtfield = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG;
  TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLESPERPIXEL, &spp);
  TIFFGetFieldDefaulted(t, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLEFORMAT, &fmtfield);
  TIFFGetFieldDefaulted(t, TIFFTAG_PLANARCONFIG, &planar);
  if(bps != 8 && bps != 16) {
    TIFFClose(t);
    std::snprintf(m.msg, sizeof(m.msg), "unsupported bits per sample: %d", (int)bps);
    m.status = 1;
    return 1;
  }
  if(dst == NULL) {
    m.h = (int)h; m.w = (int)w; m.c = (int)spp; m.bits = (int)bps;
    // load_tiff() stacks every directory of a multi-page file, which this path does not do,
    // so those files are handed back to the R reader rather than silently losing pages
    m.pages = (int) TIFFNumberOfDirectories(t);
    TIFFClose(t);
    return 0;
  }
  const size_t bytes = (size_t)bps/8;
  const size_t px = bytes * spp;
  if(planar == PLANARCONFIG_SEPARATE) {
    // one plane per sample: read each separately and interleave into the HWC buffer
    const tmsize_t ss = TIFFScanlineSize(t);
    std::vector<unsigned char> buf(ss);
    for(uint16_t sm = 0; sm < spp; sm++) {
      for(uint32_t r = 0; r < h; r++) {
        if(TIFFReadScanline(t, buf.data(), r, sm) < 0) { TIFFClose(t); set_err(m, tiff_msg[0] ? tiff_msg : "TIFFReadScanline failed"); return 1; }
        unsigned char* dstrow = dst + (size_t)r*w*px + (size_t)sm*bytes;
        for(uint32_t col = 0; col < w; col++)
          std::memcpy(dstrow + (size_t)col*px, buf.data() + (size_t)col*bytes, bytes);
      }
    }
    TIFFClose(t);
    return 0;
  }
  if(TIFFIsTiled(t)) {
    uint32_t tw = 0, th = 0;
    TIFFGetField(t, TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(t, TIFFTAG_TILELENGTH, &th);
    const tmsize_t tsz = TIFFTileSize(t);
    std::vector<unsigned char> tile(tsz);
    for(uint32_t y = 0; y < h; y += th) {
      for(uint32_t x = 0; x < w; x += tw) {
        if(TIFFReadTile(t, tile.data(), x, y, 0, 0) < 0) { TIFFClose(t); set_err(m, tiff_msg[0] ? tiff_msg : "TIFFReadTile failed"); return 1; }
        const uint32_t ny = (y + th > h) ? h - y : th;
        const uint32_t nx = (x + tw > w) ? w - x : tw;
        for(uint32_t r = 0; r < ny; r++)
          std::memcpy(dst + ((size_t)(y + r)*w + x)*px, tile.data() + (size_t)r*tw*px, (size_t)nx*px);
      }
    }
  } else {
    const tmsize_t ss = TIFFScanlineSize(t);
    std::vector<unsigned char> buf(ss);
    for(uint32_t r = 0; r < h; r++) {
      if(TIFFReadScanline(t, buf.data(), r, 0) < 0) { TIFFClose(t); set_err(m, tiff_msg[0] ? tiff_msg : "TIFFReadScanline failed"); return 1; }
      std::memcpy(dst + (size_t)r*w*px, buf.data(), (size_t)w*px);
    }
  }
  TIFFClose(t);
  return 0;
}

#endif /* HAVE_TIFF */

static int dispatch(const char* path, Meta& m, unsigned char* dst) {
  switch(m.fmt) {
#ifdef HAVE_JPEG
    case FMT_JPEG: return jpeg_probe_decode(path, m, dst);
#endif
#ifdef HAVE_PNG
    case FMT_PNG:  return png_probe_decode(path, m, dst);
#endif
#ifdef HAVE_TIFF
    case FMT_TIFF: return tiff_probe_decode(path, m, dst);
#endif
    case FMT_UNKNOWN: set_err(m, "unrecognised file format (not JPEG, PNG or TIFF)"); return 1;
    default: set_err(m, "cito was built without support for this format"); return 1;
  }
}

// [[Rcpp::export]]
Rcpp::LogicalVector cito_decoder_available() {
  Rcpp::LogicalVector v = Rcpp::LogicalVector::create(
#ifdef HAVE_JPEG
    true,
#else
    false,
#endif
#ifdef HAVE_PNG
    true,
#else
    false,
#endif
#ifdef HAVE_TIFF
    true
#else
    false
#endif
  );
  v.names() = Rcpp::CharacterVector::create("jpeg", "png", "tiff");
  return v;
}

// [[Rcpp::export]]
Rcpp::List cito_decode_batch(Rcpp::CharacterVector paths, int nthreads = 1) {
  const int n = paths.size();
  std::vector<std::string> p(n);
  for(int i = 0; i < n; i++) p[i] = Rcpp::as<std::string>(paths[i]);

#ifdef HAVE_TIFF
  // these setters are process-global, so whatever tiff/terra installed has to be put back:
  // leaving our handlers in place would disarm their error reporting for the rest of the session
  TIFFErrorHandler prev_err = TIFFSetErrorHandler(tiff_err_handler);
  TIFFErrorHandler prev_warn = TIFFSetWarningHandler(NULL);  // GeoTIFF tags are not errors
  struct TiffHandlerGuard {
    TIFFErrorHandler e, w;
    ~TiffHandlerGuard() { TIFFSetErrorHandler(e); TIFFSetWarningHandler(w); }
  } tiff_guard{prev_err, prev_warn};
#endif

  std::vector<Meta> meta(n);
  for(int i = 0; i < n; i++) {
    std::memset(&meta[i], 0, sizeof(Meta));
    bool openable = false;
    meta[i].fmt = sniff(p[i].c_str(), &openable);
    if(!openable) set_err(meta[i], "cannot open file");
    else dispatch(p[i].c_str(), meta[i], NULL);
  }

  for(int i = 0; i < n; i++) {
    if(!meta[i].status && meta[i].pages > 1)
      return Rcpp::List::create(Rcpp::_["ok"] = false, Rcpp::_["multipage"] = true);
  }

  Rcpp::CharacterVector errs(n);
  int first_ok = -1;
  for(int i = 0; i < n; i++) {
    errs[i] = meta[i].status ? meta[i].msg : "";
    if(!meta[i].status && first_ok < 0) first_ok = i;
  }
  if(first_ok < 0)
    return Rcpp::List::create(Rcpp::_["ok"] = false, Rcpp::_["multipage"] = false,
                              Rcpp::_["errors"] = errs);

  // the reference geometry is the modal one, so one odd file cannot mark every other as mismatched
  int best = first_ok, best_count = 0;
  for(int i = 0; i < n; i++) {
    if(meta[i].status) continue;
    int c = 0;
    for(int j = 0; j < n; j++)
      if(!meta[j].status && meta[j].h == meta[i].h && meta[j].w == meta[i].w &&
         meta[j].c == meta[i].c && meta[j].bits == meta[i].bits) c++;
    if(c > best_count) { best_count = c; best = i; }
  }
  const Meta& ref = meta[best];
  for(int i = 0; i < n; i++) {
    if(meta[i].status) continue;
    if(meta[i].h != ref.h || meta[i].w != ref.w || meta[i].c != ref.c)
      std::snprintf(meta[i].msg, sizeof(meta[i].msg),
                    "geometry %dx%dx%d does not match the batch (%dx%dx%d)",
                    meta[i].h, meta[i].w, meta[i].c, ref.h, ref.w, ref.c);
    else if(meta[i].bits != ref.bits)
      std::snprintf(meta[i].msg, sizeof(meta[i].msg),
                    "%d-bit does not match the batch (%d-bit)", meta[i].bits, ref.bits);
    else continue;
    meta[i].status = 1;
    errs[i] = meta[i].msg;
  }

  const size_t per = (size_t)ref.h * ref.w * ref.c * (ref.bits/8);
  Rcpp::RawVector out(per * n);
  unsigned char* base = (unsigned char*) RAW(out);

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
  #endif
  for(int i = 0; i < n; i++) {
    if(meta[i].status) continue;
    dispatch(p[i].c_str(), meta[i], base + (size_t)i * per);
  }
  for(int i = 0; i < n; i++) if(meta[i].status && errs[i] == "") errs[i] = meta[i].msg;

  return Rcpp::List::create(
    Rcpp::_["ok"] = true,
    Rcpp::_["multipage"] = false,
    Rcpp::_["data"] = out,
    Rcpp::_["dim"] = Rcpp::IntegerVector::create(n, ref.h, ref.w, ref.c),
    Rcpp::_["bits"] = ref.bits,
    Rcpp::_["errors"] = errs);
}
