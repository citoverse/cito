
.onLoad = function(libname, pkgname){
if(!torch::torch_is_installed()){
  message("Torch is not yet installed")
  message("Please run the following before using cito")
  message("\"library(\"torch\")\"")
  message("\"install_torch()\"")
  }
}
