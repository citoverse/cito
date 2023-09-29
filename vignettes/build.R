# rmarkdown::render(input = "vignettes_build/A-Introduction_to_cito.Rmd",
#                   output_format = "html_document", output_file = "../vignettes/A-Introduction_to_cito.html")
# rmarkdown::render(input = "vignettes_build/B-Training_neural_networks.Rmd",
#                   output_format = "html_document", output_file = "../vignettes/B-Training_neural_networks.html")
# rmarkdown::render(input = "vignettes_build/C-Example_Species_distribution_modeling.Rmd",
#                   output_format = "html_document", output_file = "../vignettes/C-Example_Species_distribution_modeling.html")
# rmarkdown::render(input = "vignettes_build/D-Advanced_custom_loss_functions.Rmd",
#                   output_format = "html_document", output_file = "../vignettes/D-Advanced_custom_loss_functions.html")
#
#
# paths =
#   list.files("vignettes_build/", full.names = FALSE)[stringr::str_detect( list.files("vignettes_build/") , ".Rmd")]
# file.remove(paste0("vignettes/", paths))
# file.copy(paste0("vignettes_build/", paths), paste0("vignettes/", paths))
pkgdown::build_site()
# file.remove(paste0("vignettes/", paths))


# setwd("vignettes/")
# knitr::knit("A-Introduction_to_cito.Rmd.orig",
#             output = "A-Introduction_to_cito.Rmd"
#             )
# knitr::knit("B-Training_neural_networks.Rmd.orig",
#             output = "B-Training_neural_networks.Rmd"
# )
# knitr::knit("C-Example_Species_distribution_modeling.Rmd.orig",
#             output = "C-Example_Species_distribution_modeling.Rmd"
# )
# knitr::knit("D-Advanced_custom_loss_functions.Rmd.orig",
#             output = "D-Advanced_custom_loss_functions.Rmd"
# )
