skip_if_no_torch = function() {
  if (!torch::torch_is_installed())
    skip("required torch version not available for testing")
}
