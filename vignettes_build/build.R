rmarkdown::render(input = "vignettes_build/A-Introduction_to_cito.Rmd",
                  output_format = "html_document", output_file = "../vignettes/A-Introduction_to_cito.html")
rmarkdown::render(input = "vignettes_build/B-Training_neural_networks.Rmd",
                  output_format = "html_document", output_file = "../vignettes/B-Training_neural_networks.html")
rmarkdown::render(input = "vignettes_build/C-Example_Species_distribution_modeling.Rmd",
                  output_format = "html_document", output_file = "../vignettes/C-Example_Species_distribution_modeling.html")
rmarkdown::render(input = "vignettes_build/D-Advanced_custom_loss_functions.Rmd",
                  output_format = "html_document", output_file = "../vignettes/D-Advanced_custom_loss_functions.html")

