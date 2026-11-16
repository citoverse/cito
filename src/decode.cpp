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
#include <map>
#include <array>
#include <memory>
#include <algorithm>
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
  int h, w, c, bits, pages, fallback, sformat;  // sformat 0 uint, 1 int, 2 float
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
  // libjpeg cannot convert CMYK or YCCK to RGB, and asking it to fails only at
  // jpeg_start_decompress, by which point the batch buffer is already sized from the probe
  if(ci.num_components != 1 && ci.num_components != 3) {
    const int nc = ci.num_components;
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    (void)nc;
    m.fallback = 1;   // CMYK/YCCK: jpeg::readJPEG reads it, so hand the batch over
    return 1;
  }
  // base_loader() expands greyscale to three channels, so the compiled path must too:
  // otherwise the first conv layer depends on which libraries were present at install time
  ci.out_color_space = JCS_RGB;
  if(dst == NULL) {
    m.h = ci.image_height;
    m.w = ci.image_width;
    m.c = 3;
    m.bits = 8;
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    return 0;
  }
  if((int)ci.image_height != m.h || (int)ci.image_width != m.w) {
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    set_err(m, "file changed between probe and decode");
    return 1;
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

static bool host_is_little_endian() {
  const uint16_t one = 1;
  unsigned char b;
  std::memcpy(&b, &one, 1);
  return b == 1;
}

static int png_probe_decode(const char* path, Meta& m, unsigned char* dst) {
  png_err_ctx ctx;
  ctx.msg[0] = '\0';
  FILE* f = std::fopen(path, "rb");
  if(!f) { set_err(m, "cannot open file"); return 1; }
  png_structp p = png_create_read_struct(PNG_LIBPNG_VER_STRING, &ctx, cito_png_error, cito_png_warn);
  if(!p) { std::fclose(f); set_err(m, "png_create_read_struct failed"); return 1; }
  png_infop info = png_create_info_struct(p);
  if(!info) { png_destroy_read_struct(&p, NULL, NULL); std::fclose(f); set_err(m, "png_create_info_struct failed"); return 1; }
  png_bytep* volatile rows = NULL;
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
  const bool has_trns = png_get_valid(p, info, PNG_INFO_tRNS) != 0;
  if(has_trns) png_set_tRNS_to_alpha(p);
  // base_loader() expands a 2-D grey array to three channels but leaves grey+alpha at two.
  // tRNS has just turned GRAY into GRAY_ALPHA, so testing the original colour type would
  // expand it as well and give four channels where the R reader gives two.
  if(ctype == PNG_COLOR_TYPE_GRAY && !has_trns) png_set_gray_to_rgb(p);
  // libpng wants the byte-order transform before update_info, not after
  if(depth == 16 && host_is_little_endian()) png_set_swap(p);
  png_read_update_info(p, info);
  const int channels = png_get_channels(p, info);
  const int out_depth = (depth == 16) ? 16 : 8;
  if(dst == NULL) {
    m.h = (int)h; m.w = (int)w; m.c = channels; m.bits = out_depth;
    png_destroy_read_struct(&p, &info, NULL);
    std::fclose(f);
    return 0;
  }
  if((int)h != m.h || (int)w != m.w || channels != m.c || out_depth != m.bits) {
    png_destroy_read_struct(&p, &info, NULL);
    std::fclose(f);
    set_err(m, "file changed between probe and decode");
    return 1;
  }
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

struct TiffCloser { void operator()(TIFF* t) const { if(t) TIFFClose(t); } };

// libtiff's handlers are process-global; whatever tiff or terra installed has to be put back.
// An R allocation can longjmp past a destructor, so this must never span one.
struct TiffHandlerGuard {
  TIFFErrorHandler e, w;
  TiffHandlerGuard() : e(TIFFSetErrorHandler(tiff_err_handler)), w(TIFFSetWarningHandler(NULL)) {}
  ~TiffHandlerGuard() { TIFFSetErrorHandler(e); TIFFSetWarningHandler(w); }
};

static int tiff_probe_decode(const char* path, Meta& m, unsigned char* dst) {
  tiff_msg[0] = '\0';
  // libtiff reports through return codes rather than longjmp, so RAII is safe here and
  // removes six hand-written TIFFClose calls from the error paths
  std::unique_ptr<TIFF, TiffCloser> tp(TIFFOpen(path, "r"));
  TIFF* t = tp.get();
  if(!t) { set_err(m, tiff_msg[0] ? tiff_msg : "cannot open TIFF"); return 1; }
  uint32_t w = 0, h = 0;
  uint16_t spp = 1, bps = 8, fmtfield = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG, photo = PHOTOMETRIC_MINISBLACK;
  TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLESPERPIXEL, &spp);
  TIFFGetFieldDefaulted(t, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLEFORMAT, &fmtfield);
  TIFFGetFieldDefaulted(t, TIFFTAG_PLANARCONFIG, &planar);
  TIFFGetFieldDefaulted(t, TIFFTAG_PHOTOMETRIC, &photo);
  const bool is_float = (bps == 32 && fmtfield == SAMPLEFORMAT_IEEEFP);
  const bool is_int   = (bps == 16 && fmtfield == SAMPLEFORMAT_INT);
  const bool is_uint  = ((bps == 8 || bps == 16) && fmtfield == SAMPLEFORMAT_UINT);
  if(!is_float && !is_int && !is_uint) { m.fallback = 1; return 1; }
  m.sformat = is_float ? 2 : (is_int ? 1 : 0);
  // only layouts whose scanline is exactly w*spp*bytes are safe for the flat copy below.
  // Palette needs a colormap lookup and subsampled YCbCr packs fewer bytes than w*px, which
  // would make the memcpy read past the scanline buffer. Both go back to the R reader.
  if(photo != PHOTOMETRIC_MINISBLACK && photo != PHOTOMETRIC_RGB &&
     photo != PHOTOMETRIC_MINISWHITE && photo != PHOTOMETRIC_PALETTE) { m.fallback = 1; return 1; }
  // a palette image stores one index per pixel and expands to RGB, which is what readTIFF does
  if(photo == PHOTOMETRIC_PALETTE && (spp != 1 || bps > 8)) { m.fallback = 1; return 1; }
  // every layout check must happen here, in the probe: the batch is rejected before the buffer
  // is allocated, and a check in the decode pass would come too late to stop it
  if(planar == PLANARCONFIG_SEPARATE && TIFFIsTiled(t)) { m.fallback = 1; return 1; }
  {
    const size_t bytes0 = (size_t)bps/8, px0 = bytes0 * spp;
    if(TIFFIsTiled(t)) {
      uint32_t tw0 = 0, th0 = 0;
      TIFFGetField(t, TIFFTAG_TILEWIDTH, &tw0);
      TIFFGetField(t, TIFFTAG_TILELENGTH, &th0);
      if((size_t)TIFFTileSize(t) != (size_t)tw0*th0*px0) { m.fallback = 1; return 1; }
    } else if(planar == PLANARCONFIG_CONTIG && (size_t)TIFFScanlineSize(t) != (size_t)w*px0) {
      m.fallback = 1;
      return 1;
    } else if(planar == PLANARCONFIG_SEPARATE && (size_t)TIFFScanlineSize(t) != (size_t)w*bytes0) {
      m.fallback = 1;
      return 1;
    }
  }
  if(dst == NULL) {
    m.h = (int)h; m.w = (int)w; m.bits = (int)bps;
    m.c = (photo == PHOTOMETRIC_PALETTE) ? 3 : (int)spp;
    // A cloud-optimised GeoTIFF stores its overviews as extra directories flagged
    // FILETYPE_REDUCEDIMAGE. Counting those as pages would refuse every COG, so only
    // full-resolution directories count; load_tiff() stacks genuine pages and this path cannot.
    {
      int pages = 0;
      tdir_t cur = TIFFCurrentDirectory(t);
      TIFFSetDirectory(t, 0);
      do {
        uint32_t sub = 0;
        TIFFGetFieldDefaulted(t, TIFFTAG_SUBFILETYPE, &sub);
        // overviews carry FILETYPE_REDUCEDIMAGE and internal masks FILETYPE_MASK; neither is
        // a page, and a mask directory is present in essentially every modern COG
        if(!(sub & (FILETYPE_REDUCEDIMAGE | FILETYPE_MASK))) pages++;
      } while(TIFFReadDirectory(t));
      TIFFSetDirectory(t, cur);
      m.pages = pages;
    }
    return 0;
  }
  const bool palette = (photo == PHOTOMETRIC_PALETTE);
  const int want_c = palette ? 3 : (int)spp;
  if((int)h != m.h || (int)w != m.w || want_c != m.c || (int)bps != m.bits) {
    set_err(m, "file changed between probe and decode");
    return 1;
  }
  // the palette expansion needs somewhere to put the indices before overwriting dst with RGB
  std::vector<unsigned char> idx;
  unsigned char* target = dst;
  if(palette) {
    idx.resize((size_t)h*w);
    target = idx.data();
  }
  const size_t bytes = (size_t)bps/8;
  const size_t px = bytes * spp;
  if(planar == PLANARCONFIG_SEPARATE) {
    // one plane per sample: read each separately and interleave into the HWC buffer
    const tmsize_t ss = TIFFScanlineSize(t);
    std::vector<unsigned char> buf(ss);
    for(uint16_t sm = 0; sm < spp; sm++) {
      for(uint32_t r = 0; r < h; r++) {
        if(TIFFReadScanline(t, buf.data(), r, sm) < 0) { set_err(m, tiff_msg[0] ? tiff_msg : "TIFFReadScanline failed"); return 1; }
        unsigned char* dstrow = target + (size_t)r*w*px + (size_t)sm*bytes;
        for(uint32_t col = 0; col < w; col++)
          std::memcpy(dstrow + (size_t)col*px, buf.data() + (size_t)col*bytes, bytes);
      }
    }
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
        if(TIFFReadTile(t, tile.data(), x, y, 0, 0) < 0) { set_err(m, tiff_msg[0] ? tiff_msg : "TIFFReadTile failed"); return 1; }
        const uint32_t ny = (y + th > h) ? h - y : th;
        const uint32_t nx = (x + tw > w) ? w - x : tw;
        for(uint32_t r = 0; r < ny; r++)
          std::memcpy(target + ((size_t)(y + r)*w + x)*px, tile.data() + (size_t)r*tw*px, (size_t)nx*px);
      }
    }
  } else {
    const tmsize_t ss = TIFFScanlineSize(t);
    std::vector<unsigned char> buf(ss);
    for(uint32_t r = 0; r < h; r++) {
      if(TIFFReadScanline(t, buf.data(), r, 0) < 0) { set_err(m, tiff_msg[0] ? tiff_msg : "TIFFReadScanline failed"); return 1; }
      std::memcpy(target + (size_t)r*w*px, buf.data(), (size_t)w*px);
    }
  }
  if(palette) {
    uint16_t *rmap = NULL, *gmap = NULL, *bmap = NULL;
    if(!TIFFGetField(t, TIFFTAG_COLORMAP, &rmap, &gmap, &bmap)) {
      set_err(m, "palette image without a colormap");
      return 1;
    }
    // libtiff hands back 16-bit colormap entries; readTIFF scales them to 8 bits the same way
    for(size_t i = 0, n = (size_t)h*w; i < n; i++) {
      const unsigned v = idx[i];
      dst[i*3 + 0] = (unsigned char)(rmap[v] >> 8);
      dst[i*3 + 1] = (unsigned char)(gmap[v] >> 8);
      dst[i*3 + 2] = (unsigned char)(bmap[v] >> 8);
    }
  }
  // MINISWHITE is not inverted here: both GDAL and tiff::readTIFF return the stored samples and
  // treat the photometric as a display hint, and for training data the stored values are what
  // is wanted. Inverting would make this path the only one that disagrees.
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
    default: m.fallback = 1; return 1;   // built without that library: the R reader takes it
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
  if(nthreads < 1) nthreads = 1;
  const int n = paths.size();
  std::vector<std::string> p(n);
  for(int i = 0; i < n; i++) p[i] = Rcpp::as<std::string>(paths[i]);

  // libtiff's handlers are process-global, so whatever tiff/terra installed has to be put back.
  // An R allocation can longjmp past a C++ destructor, so the guard must never span one: it is
  // scoped tightly around the two phases that actually call libtiff.
#ifdef HAVE_TIFF
#define CITO_TIFF_GUARD TiffHandlerGuard cito_tiff_guard_
#else
#define CITO_TIFF_GUARD do {} while(0)
#endif

  std::vector<Meta> meta(n);
  {
    CITO_TIFF_GUARD;
    for(int i = 0; i < n; i++) {
      std::memset(&meta[i], 0, sizeof(Meta));
      bool openable = false;
      meta[i].fmt = sniff(p[i].c_str(), &openable);
      if(!openable) set_err(meta[i], "cannot open file");
      else dispatch(p[i].c_str(), meta[i], NULL);
    }
  }

  // anything the compiled path cannot read safely goes back to the R reader for the whole batch
  for(int i = 0; i < n; i++) {
    if(meta[i].fallback || (!meta[i].status && meta[i].pages > 1))
      return Rcpp::List::create(Rcpp::_["ok"] = false, Rcpp::_["fallback"] = true);
  }

  Rcpp::CharacterVector errs(n);
  int first_ok = -1;
  for(int i = 0; i < n; i++) {
    errs[i] = meta[i].status ? meta[i].msg : "";
    if(!meta[i].status && first_ok < 0) first_ok = i;
  }
  if(first_ok < 0)
    return Rcpp::List::create(Rcpp::_["ok"] = false, Rcpp::_["fallback"] = false,
                              Rcpp::_["errors"] = errs);

  // the reference geometry is the modal one, so one odd file cannot mark every other as mismatched
  std::map<std::array<int, 4>, int> counts;
  for(int i = 0; i < n; i++)
    if(!meta[i].status) counts[{meta[i].h, meta[i].w, meta[i].c, meta[i].bits*8 + meta[i].sformat}]++;
  // seeded from the first readable file so that a tie goes to it rather than to whichever
  // geometry happens to sort first
  std::array<int, 4> modal{meta[first_ok].h, meta[first_ok].w, meta[first_ok].c,
                           meta[first_ok].bits*8 + meta[first_ok].sformat};
  int best_count = counts[modal];
  for(const auto& kv : counts)
    if(kv.second > best_count) { best_count = kv.second; modal = kv.first; }
  int best = first_ok;
  for(int i = 0; i < n; i++)
    if(!meta[i].status &&
       std::array<int, 4>{meta[i].h, meta[i].w, meta[i].c,
                          meta[i].bits*8 + meta[i].sformat} == modal) { best = i; break; }
  const Meta& ref = meta[best];
  for(int i = 0; i < n; i++) {
    if(meta[i].status) continue;
    if(meta[i].h != ref.h || meta[i].w != ref.w || meta[i].c != ref.c)
      std::snprintf(meta[i].msg, sizeof(meta[i].msg),
                    "geometry %dx%dx%d does not match the batch (%dx%dx%d)",
                    meta[i].h, meta[i].w, meta[i].c, ref.h, ref.w, ref.c);
    else if(meta[i].bits != ref.bits || meta[i].sformat != ref.sformat) {
      // Both are reachable at 16 bits: signed elevation and unsigned reflectance in one folder
      // is ordinary. Decoding them under one interpretation rescales half the batch silently,
      // and which half depends on the shuffle, so the same file would differ between epochs.
      return Rcpp::List::create(Rcpp::_["ok"] = false, Rcpp::_["fallback"] = true);
    }
    else continue;
    meta[i].status = 1;
    errs[i] = meta[i].msg;
  }

  const size_t per = (size_t)ref.h * ref.w * ref.c * (ref.bits/8);
  Rcpp::RawVector out(per * n);
  unsigned char* base = (unsigned char*) RAW(out);

  {
    CITO_TIFF_GUARD;
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    #endif
    for(int i = 0; i < n; i++) {
      if(meta[i].status) continue;
      // an exception leaving an OpenMP structured block is std::terminate, so a bad_alloc from
      // a hostile tile geometry has to be caught here
      try {
        dispatch(p[i].c_str(), meta[i], base + (size_t)i * per);
      } catch(const std::exception& e) {
        set_err(meta[i], e.what());
      } catch(...) {
        set_err(meta[i], "unknown error while decoding");
      }
    }
  }
  for(int i = 0; i < n; i++) {
    // fallback can only be decided in the probe; reaching it here means the file changed
    if(meta[i].fallback && !meta[i].status) set_err(meta[i], "file changed between probe and decode");
    if(meta[i].status) errs[i] = meta[i].msg[0] ? meta[i].msg : "decoding failed";
  }

  return Rcpp::List::create(
    Rcpp::_["ok"] = true,
    Rcpp::_["fallback"] = false,
    Rcpp::_["data"] = out,
    Rcpp::_["dim"] = Rcpp::IntegerVector::create(n, ref.h, ref.w, ref.c),
    Rcpp::_["bits"] = ref.bits,
    Rcpp::_["sformat"] = ref.sformat,
    Rcpp::_["errors"] = errs);
}

/* ---- windows from one large raster, read while the previous batch trains -------------------- */
#ifdef HAVE_TIFF
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <future>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// libtiff 4.5 gave handles their own error handlers. Only then may a read run on a thread of its
// own: the global handlers belong to whatever tiff or terra installed and cannot be swapped
// underneath R. Older libtiff reads synchronously, inside the usual guard (-DCITO_RASTER_SYNC forces it).
#if defined(TIFFLIB_VERSION) && TIFFLIB_VERSION >= 20221213 && !defined(CITO_RASTER_SYNC)
#define CITO_RASTER_ASYNC 1
#endif

struct Raster {
  int w = 0, h = 0, c = 0, bits = 0, sformat = 0, tiled = 0, separate = 0;
  uint32_t bw = 0, bh = 0;                 // block = tile, or strip of rows_per_strip rows
  double gt[6] = {0, 1, 0, 0, 0, 1};
  int georef = 0, has_nodata = 0, point = 0;
  double nodata = 0;
};

// GeoTIFF tags are unknown to libtiff and arrive as anonymous fields; how their count is passed
// depends on the field definition, so it is looked up rather than assumed
template<class T>
static bool tag_array(TIFF* t, uint32_t tag, TIFFDataType type, std::vector<T>& out) {
  const TIFFField* fi = TIFFFindField(t, tag, TIFF_ANY);
  if(!fi || TIFFFieldDataType(fi) != type) return false;
  void* p = NULL;
  uint32_t n = 0;
  if(TIFFFieldPassCount(fi)) {
    if(TIFFFieldReadCount(fi) == TIFF_VARIABLE2) {
      if(!TIFFGetField(t, tag, &n, &p)) return false;
    } else {
      uint16_t n16 = 0;
      if(!TIFFGetField(t, tag, &n16, &p)) return false;
      n = n16;
    }
  } else {
    if(!TIFFGetField(t, tag, &p)) return false;
    n = type == TIFF_ASCII ? (uint32_t)std::strlen((const char*)p) : (uint32_t)TIFFFieldReadCount(fi);
  }
  if(!p) return false;
  out.assign((const T*)p, (const T*)p + n);
  return true;
}

static std::string raster_geometry(TIFF* t, Raster& r) {
  uint32_t w = 0, h = 0;
  uint16_t spp = 1, bps = 8, fmt = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG;
  TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLESPERPIXEL, &spp);
  TIFFGetFieldDefaulted(t, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLEFORMAT, &fmt);
  TIFFGetFieldDefaulted(t, TIFFTAG_PLANARCONFIG, &planar);
  const bool ok = (fmt == SAMPLEFORMAT_UINT && (bps == 8 || bps == 16 || bps == 32)) ||
                  (fmt == SAMPLEFORMAT_INT && (bps == 8 || bps == 16 || bps == 32)) ||
                  (fmt == SAMPLEFORMAT_IEEEFP && (bps == 32 || bps == 64));
  if(!ok) return "unsupported sample type (" + std::to_string(bps) + " bit, format " + std::to_string(fmt) + ")";
  r.w = (int)w;
  r.h = (int)h;
  r.c = (int)spp;
  r.bits = (int)bps;
  r.sformat = fmt == SAMPLEFORMAT_IEEEFP ? 2 : (fmt == SAMPLEFORMAT_INT ? 1 : 0);
  r.separate = planar == PLANARCONFIG_SEPARATE;
  r.tiled = TIFFIsTiled(t);
  const size_t sample = (size_t)bps / 8 * (r.separate ? 1 : spp);
  if(r.tiled) {
    TIFFGetField(t, TIFFTAG_TILEWIDTH, &r.bw);
    TIFFGetField(t, TIFFTAG_TILELENGTH, &r.bh);
    if((size_t)TIFFTileSize(t) != (size_t)r.bw * r.bh * sample) return "unsupported tile layout";
  } else {
    uint32_t rps = h;
    TIFFGetFieldDefaulted(t, TIFFTAG_ROWSPERSTRIP, &rps);
    r.bw = w;
    r.bh = std::min(rps, h);
    if((size_t)TIFFScanlineSize(t) != (size_t)w * sample) return "unsupported strip layout";
  }

  std::vector<double> scale, tie, mat;
  std::vector<uint16_t> keys;
  std::vector<char> nd;
  if(tag_array(t, 34264, TIFF_DOUBLE, mat) && mat.size() >= 8) {
    r.gt[0] = mat[3]; r.gt[1] = mat[0]; r.gt[2] = mat[1];
    r.gt[3] = mat[7]; r.gt[4] = mat[4]; r.gt[5] = mat[5];
    r.georef = 1;
  } else if(tag_array(t, 33550, TIFF_DOUBLE, scale) && tag_array(t, 33922, TIFF_DOUBLE, tie) &&
            scale.size() >= 2 && tie.size() >= 6) {
    r.gt[1] = scale[0];
    r.gt[5] = -scale[1];
    r.gt[0] = tie[3] - tie[0] * scale[0];
    r.gt[3] = tie[4] + tie[1] * scale[1];
    r.georef = 1;
  }
  // GTRasterTypeGeoKey = PixelIsPoint: the coordinates name pixel centres, not corners
  if(tag_array(t, 34735, TIFF_SHORT, keys) && keys.size() >= 4)
    for(size_t k = 4; k + 3 < keys.size(); k += 4)
      if(keys[k] == 1025 && keys[k + 1] == 0 && keys[k + 3] == 2) r.point = 1;
  if(r.georef && r.point) {
    r.gt[0] -= 0.5 * r.gt[1] + 0.5 * r.gt[2];
    r.gt[3] -= 0.5 * r.gt[4] + 0.5 * r.gt[5];
  }
  if(tag_array(t, 42113, TIFF_ASCII, nd) && !nd.empty()) {
    std::string s(nd.begin(), nd.end());
    char* end = NULL;
    const double v = std::strtod(s.c_str(), &end);
    // a float32 sample holds the tag value rounded to float, so compare against that (GDAL does the same)
    if(end != s.c_str()) { r.nodata = (r.sformat == 2 && r.bits == 32) ? (double)(float)v : v; r.has_nodata = 1; }
  }
  return "";
}

template<class T>
static void convert(const unsigned char* src, size_t stride, int n, float* dst, size_t dst_stride,
                    const Raster& r, double center, double scale, float fill) {
  for(int i = 0; i < n; i++) {
    T v;
    std::memcpy(&v, src + i * stride, sizeof(T));
    const double d = (double)v;
    dst[i * dst_stride] = (std::isnan(d) || (r.has_nodata && d == r.nodata)) ? fill : (float)((d - center) / scale);
  }
}

static void convert_run(const unsigned char* src, size_t stride, int n, float* dst, size_t dst_stride,
                        const Raster& r, double center, double scale, float fill) {
  if(r.sformat == 2) {
    if(r.bits == 32) convert<float>(src, stride, n, dst, dst_stride, r, center, scale, fill);
    else convert<double>(src, stride, n, dst, dst_stride, r, center, scale, fill);
  } else if(r.sformat == 1) {
    if(r.bits == 8) convert<int8_t>(src, stride, n, dst, dst_stride, r, center, scale, fill);
    else if(r.bits == 16) convert<int16_t>(src, stride, n, dst, dst_stride, r, center, scale, fill);
    else convert<int32_t>(src, stride, n, dst, dst_stride, r, center, scale, fill);
  } else {
    if(r.bits == 8) convert<uint8_t>(src, stride, n, dst, dst_stride, r, center, scale, fill);
    else if(r.bits == 16) convert<uint16_t>(src, stride, n, dst, dst_stride, r, center, scale, fill);
    else convert<uint32_t>(src, stride, n, dst, dst_stride, r, center, scale, fill);
  }
}

#ifdef CITO_RASTER_ASYNC
static int raster_err_handler(TIFF*, void* user, const char*, const char* fmt, va_list ap) {
  std::vsnprintf((char*)user, 200, fmt, ap);
  return 1;
}
static int raster_warn_handler(TIFF*, void*, const char*, const char*, va_list) { return 1; }
#endif

// one per OpenMP thread: a handle and the last decoded block, which neighbouring windows reuse
struct Slot {
  TIFF* t = NULL;
  std::vector<unsigned char> block;
  long cached = -1;
  char msg[200] = {0};
  ~Slot() { if(t) TIFFClose(t); }
};

struct RasterReader {
  std::string path;
  Raster r;
  int nthreads = 1;
  std::vector<double> center, scale;
  float fill = 0;
  std::vector<std::unique_ptr<Slot>> slots;
  std::map<int, std::shared_ptr<std::string>> errors;
  std::map<int, Rcpp::RawVector> out;       // allocated and released on the main thread only
  std::map<int, std::future<void>> jobs;
  int next = 0;
  std::deque<std::packaged_task<void()>> queue;
  std::mutex mu;
  std::condition_variable cv;
  bool stop = false;
  std::thread worker;

  TIFF* open(Slot& s) {
#ifdef CITO_RASTER_ASYNC
    TIFFOpenOptions* o = TIFFOpenOptionsAlloc();
    TIFFOpenOptionsSetErrorHandlerExtR(o, raster_err_handler, s.msg);
    TIFFOpenOptionsSetWarningHandlerExtR(o, raster_warn_handler, NULL);
    TIFF* t = TIFFOpenExt(path.c_str(), "r", o);
    TIFFOpenOptionsFree(o);
    return t;
#else
    tiff_msg[0] = '\0';
    TIFF* t = TIFFOpen(path.c_str(), "r");
    if(!t) std::snprintf(s.msg, sizeof(s.msg), "%s", tiff_msg);
    return t;
#endif
  }

  // the message of the failing call: per handle with libtiff >= 4.5, thread-local before
  static const char* message(Slot& s) {
#ifndef CITO_RASTER_ASYNC
    if(!s.msg[0]) std::snprintf(s.msg, sizeof(s.msg), "%s", tiff_msg);
#endif
    return s.msg;
  }

  // block (bx, by) of sample plane `plane` into s.block
  bool load(Slot& s, uint32_t bx, uint32_t by, int plane) {
    const long id = ((long)plane * ((r.h + r.bh - 1) / r.bh) + by / r.bh) * ((r.w + r.bw - 1) / r.bw) + bx / r.bw;
    if(s.cached == id) return true;
    s.cached = -1;
    s.msg[0] = '\0';
    tmsize_t got = r.tiled ? TIFFReadEncodedTile(s.t, TIFFComputeTile(s.t, bx, by, 0, plane), s.block.data(), (tmsize_t)s.block.size())
                           : TIFFReadEncodedStrip(s.t, TIFFComputeStrip(s.t, by, plane), s.block.data(), (tmsize_t)s.block.size());
    if(got < 0) return false;
    s.cached = id;
    return true;
  }

  // one window into o (bands x wh x ww), already filled; the part outside the raster stays fill
  bool window(Slot& s, int x0, int y0, int ww, int wh, float* o) {
    const size_t bytes = (size_t)r.bits / 8;
    const size_t px = r.separate ? bytes : bytes * r.c;
    const int c0 = std::max(0, x0), c1 = std::min(r.w, x0 + ww);
    const int r0 = std::max(0, y0), r1 = std::min(r.h, y0 + wh);
    if(c0 >= c1 || r0 >= r1) return true;
    for(int p = 0; p < (r.separate ? r.c : 1); p++)
      for(uint32_t by = (r0 / r.bh) * r.bh; by < (uint32_t)r1; by += r.bh)
        for(uint32_t bx = (c0 / r.bw) * r.bw; bx < (uint32_t)c1; bx += r.bw) {
          if(!load(s, bx, by, p)) return false;
          const int rr0 = std::max<int>(r0, by), rr1 = std::min<int>(r1, by + r.bh);
          const int cc0 = std::max<int>(c0, bx), cc1 = std::min<int>(c1, bx + r.bw);
          for(int row = rr0; row < rr1; row++)
            for(int b = 0; b < (r.separate ? 1 : r.c); b++) {
              const int band = r.separate ? p : b;
              const unsigned char* src = s.block.data() + ((size_t)(row - by) * r.bw + (cc0 - bx)) * px + (r.separate ? 0 : b * bytes);
              float* d = o + ((size_t)band * wh + (row - y0)) * ww + (cc0 - x0);
              convert_run(src, px, cc1 - cc0, d, 1, r, center[band], scale[band], fill);
            }
        }
    return true;
  }

  void read(const std::vector<int>& x0, const std::vector<int>& y0, int ww, int wh, float* dst, std::string* err) {
    const int n = (int)x0.size();
    const size_t per = (size_t)r.c * ww * wh;
    std::mutex mu_err;
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    #endif
    for(int i = 0; i < n; i++) {
      #ifdef _OPENMP
      Slot& s = *slots[omp_get_thread_num()];
      #else
      Slot& s = *slots[0];
      #endif
      float* o = dst + per * i;
      std::fill(o, o + per, fill);
      std::string what;
      try {
        if(!s.t) {
          s.msg[0] = '\0';
          s.t = open(s);
          if(s.t) s.block.resize(r.tiled ? (size_t)TIFFTileSize(s.t) : (size_t)TIFFStripSize(s.t));
          else what = message(s)[0] ? message(s) : "cannot open raster";
        }
        if(s.t && !window(s, x0[i], y0[i], ww, wh, o)) what = message(s)[0] ? message(s) : "reading a raster block failed";
      } catch(const std::exception& e) {
        what = e.what();
      } catch(...) {
        what = "unknown error while reading the raster";
      }
      if(!what.empty()) {
        std::lock_guard<std::mutex> lk(mu_err);
        if(err->empty()) *err = what;
      }
    }
  }

  void start() {
#ifdef CITO_RASTER_ASYNC
    worker = std::thread([this]() {
      for(;;) {
        std::packaged_task<void()> job;
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [this]() { return stop || !queue.empty(); });
          if(queue.empty()) return;
          job = std::move(queue.front());
          queue.pop_front();
        }
        job();
      }
    });
#endif
  }

  ~RasterReader() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_one();
    if(worker.joinable()) worker.join();
  }
};

static Rcpp::List raster_list(const Raster& r) {
  return Rcpp::List::create(
    Rcpp::_["width"] = r.w, Rcpp::_["height"] = r.h, Rcpp::_["bands"] = r.c, Rcpp::_["bits"] = r.bits,
    Rcpp::_["sformat"] = r.sformat, Rcpp::_["tiled"] = (bool)r.tiled,
    Rcpp::_["block"] = Rcpp::IntegerVector::create(r.bw, r.bh),
    Rcpp::_["geotransform"] = Rcpp::NumericVector(r.gt, r.gt + 6), Rcpp::_["georeferenced"] = (bool)r.georef,
    Rcpp::_["nodata"] = r.has_nodata ? Rcpp::NumericVector::create(r.nodata) : Rcpp::NumericVector(0));
}

static std::string raster_probe(const std::string& path, Raster& r) {
  TiffHandlerGuard guard;
  tiff_msg[0] = '\0';
  std::unique_ptr<TIFF, TiffCloser> t(TIFFOpen(path.c_str(), "r"));
  if(!t) return tiff_msg[0] ? std::string(tiff_msg) : "cannot open TIFF";
  return raster_geometry(t.get(), r);
}

// [[Rcpp::export]]
Rcpp::List cito_raster_info(std::string path) {
  Raster r;
  std::string err = raster_probe(path, r);
  if(!err.empty()) Rcpp::stop(path + ": " + err);
  return raster_list(r);
}

// [[Rcpp::export]]
SEXP cito_raster_open(std::string path, int nthreads, Rcpp::NumericVector center, Rcpp::NumericVector scale, double fill) {
  Raster r;
  std::string err = raster_probe(path, r);
  if(!err.empty()) Rcpp::stop(path + ": " + err);
  if(center.size() != r.c || scale.size() != r.c) Rcpp::stop("center and scale need one value per band");
  RasterReader* rd = new RasterReader();
  rd->path = path;
  rd->r = r;
  rd->nthreads = std::max(1, nthreads);
  rd->center.assign(center.begin(), center.end());
  rd->scale.assign(scale.begin(), scale.end());
  rd->fill = (float)fill;
  for(int k = 0; k < rd->nthreads; k++) rd->slots.emplace_back(new Slot());
  rd->start();
  return Rcpp::XPtr<RasterReader>(rd, true);
}

// queue the windows (top-left pixel x0, y0, all ww x wh) and return a ticket for cito_raster_fetch()
// [[Rcpp::export]]
int cito_raster_submit(SEXP reader, Rcpp::IntegerVector x0, Rcpp::IntegerVector y0, int ww, int wh) {
  Rcpp::XPtr<RasterReader> rd(reader);
  const int id = rd->next++;
  Rcpp::RawVector buf((size_t)x0.size() * rd->r.c * ww * wh * sizeof(float));
  rd->out[id] = buf;
  float* dst = (float*) RAW(buf);
  std::vector<int> x(x0.begin(), x0.end()), y(y0.begin(), y0.end());
  RasterReader* p = rd.get();
  std::shared_ptr<std::string> err(new std::string());
  rd->errors[id] = err;
  std::packaged_task<void()> job([p, x, y, ww, wh, dst, err]() { p->read(x, y, ww, wh, dst, err.get()); });
  rd->jobs[id] = job.get_future();
#ifdef CITO_RASTER_ASYNC
  {
    std::lock_guard<std::mutex> lk(rd->mu);
    rd->queue.push_back(std::move(job));
  }
  rd->cv.notify_one();
#else
  {
    TiffHandlerGuard guard;
    job();
  }
#endif
  return id;
}

// [[Rcpp::export]]
Rcpp::RawVector cito_raster_fetch(SEXP reader, int id) {
  Rcpp::XPtr<RasterReader> rd(reader);
  auto j = rd->jobs.find(id);
  if(j == rd->jobs.end()) Rcpp::stop("unknown raster batch");
  j->second.wait();
  rd->jobs.erase(j);
  Rcpp::RawVector res = rd->out[id];
  rd->out.erase(id);
  const std::string err = *rd->errors[id];
  rd->errors.erase(id);
  if(!err.empty()) Rcpp::stop(rd->path + ": " + err);
  return res;
}
#else
// without libtiff the R side never gets here: raster_patches() checks cito_decoder_available().
// No export markers: compileAttributes() ignores the #ifdef and would register them twice.
static void no_tiff() { Rcpp::stop("cito was installed without libtiff, which raster inputs need"); }
Rcpp::List cito_raster_info(std::string path) { no_tiff(); return Rcpp::List(); }
SEXP cito_raster_open(std::string path, int nthreads, Rcpp::NumericVector center, Rcpp::NumericVector scale, double fill) { no_tiff(); return R_NilValue; }
int cito_raster_submit(SEXP reader, Rcpp::IntegerVector x0, Rcpp::IntegerVector y0, int ww, int wh) { no_tiff(); return 0; }
Rcpp::RawVector cito_raster_fetch(SEXP reader, int id) { no_tiff(); return Rcpp::RawVector(); }
#endif /* HAVE_TIFF */
