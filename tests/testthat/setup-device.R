# Run the whole suite on one device:  CITO_TEST_DEVICE=cuda Rscript -e 'devtools::test()'
# Most tests never pass device=, so without this they always run on the default. Reordering the
# match.arg choices moves the default without invalidating an explicit device= in a test.
local({
  dev = Sys.getenv("CITO_TEST_DEVICE")
  if(!nzchar(dev)) return(invisible(NULL))
  choices = unique(c(dev, "cpu", "cuda", "mps"))
  envs = list(asNamespace("cito"), as.environment("package:cito"))
  for(f in c("dnn", "cnn", "mmn")) {
    fun = get(f, envir = asNamespace("cito"))
    formals(fun)$device = choices
    for(e in envs) {
      unlockBinding(f, e)
      assign(f, fun, envir = e)
      lockBinding(f, e)
    }
  }
  message("cito tests: default device set to ", dev)
})
