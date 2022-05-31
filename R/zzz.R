
.onAttach <- function(libname, pkgname){
if(!torch::torch_is_installed()){
  packageStartupMessage("Torch is not yet installed")
  packageStartupMessage("Please run the following before using cito")
  packageStartupMessage("\"library(\"torch\")\"")
  packageStartupMessage("\"install_torch()\"")
  packageStartupMessage("see: https://torch.mlverse.org/docs/articles/installation.html")
  }
}
