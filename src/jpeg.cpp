// The JPEG decoder lives in its own translation unit because it must not see any R header:
// R defines FALSE/TRUE as enumerators of Rboolean, and IJG libjpeg 9 (CRAN macOS) defines its
// own enum boolean with the same enumerators, so the two cannot share one file.
#ifdef HAVE_JPEG
#include <cstdio>
#include <csetjmp>
extern "C" {
#include <jpeglib.h>
}

/* ---- libjpeg: default error_exit calls exit(), so every call gets its own setjmp ---- */
struct jerr_mgr { struct jpeg_error_mgr pub; jmp_buf jb; char msg[JMSG_LENGTH_MAX]; };
static void jerr_exit(j_common_ptr ci) {
  jerr_mgr* e = reinterpret_cast<jerr_mgr*>(ci->err);
  (*ci->err->format_message)(ci, e->msg);
  std::longjmp(e->jb, 1);
}
static void jerr_silent(j_common_ptr, int) {}

static int fail(char* msg, int msglen, const char* what) {
  std::snprintf(msg, msglen, "%s", what);
  return 1;
}

// Returns 0 on success, 1 with msg set on error, 2 for a layout libjpeg cannot convert to RGB.
// dst == NULL probes h and w; otherwise h and w must match the probe and dst is filled as RGB8.
// No non-trivial C++ object may be live across the setjmp, hence the raw FILE* and manual cleanup.
int cito_jpeg_decode(const char* path, int* h, int* w, unsigned char* dst, char* msg, int msglen) {
  struct jpeg_decompress_struct ci;
  jerr_mgr jerr;
  FILE* f = std::fopen(path, "rb");
  if(!f) return fail(msg, msglen, "cannot open file");
  ci.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jerr_exit;
  jerr.pub.emit_message = jerr_silent;
  if(setjmp(jerr.jb)) {
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    return fail(msg, msglen, jerr.msg);
  }
  jpeg_create_decompress(&ci);
  jpeg_stdio_src(&ci, f);
  jpeg_read_header(&ci, TRUE);
  // libjpeg cannot convert CMYK or YCCK to RGB, and asking it to fails only at
  // jpeg_start_decompress, by which point the batch buffer is already sized from the probe
  if(ci.num_components != 1 && ci.num_components != 3) {
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    return 2;
  }
  // base_loader() expands greyscale to three channels, so the compiled path must too:
  // otherwise the first conv layer depends on which libraries were present at install time
  ci.out_color_space = JCS_RGB;
  if(dst == NULL) {
    *h = ci.image_height;
    *w = ci.image_width;
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    return 0;
  }
  if((int)ci.image_height != *h || (int)ci.image_width != *w) {
    jpeg_destroy_decompress(&ci);
    std::fclose(f);
    return fail(msg, msglen, "file changed between probe and decode");
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
