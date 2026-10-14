# cito 1.2

## New features

* Analytical ce 
* Analytical ALE
* Importance based on ALE
* MMN
* CNN: cnn(), create_architecture(), conv(), linear(), maxPool(), avgPool() and transfer() are now exported
* weights argument for custom likelihoods
* custom argument for dnn() and cnn(): user-provided nn_module architectures, also usable as submodules within mmn()
* new losses: bernoulli, nbinom, multinomial and clogit; loss = "softmax" is deprecated in favor of loss = "cross-entropy"
* tree_data dataset
* data augmentation for cnn() (data_augmentation), and image folders (jpeg/png/tiff) as input
* transfer(rgb = FALSE) averages the pretrained weights of the first convolutional layer for non-RGB data
* ignite_adam optimizer
* hooks argument of dnn(), evaluated after every epoch
* blocked resampling: config_tuning(blocking = ) for the CV folds and dnn(bootstrap_blocking_variable = ) for the bootstrap
* ce and ce_method arguments of dnn(): conditional effects are computed once at the end of the training and stored on the model, so summary() and ALE() do not have to recompute them. ce = <number> sets the number of observations used for them
* predict(..., batchsize = ) and predict(..., return_embeddings = )

## Breaking changes

* loss = "binomial" no longer accepts a 0/1 vector. It now expects a factor/character response, or an
  integer matrix/data.frame with two columns (successes, failures). Use loss = "bernoulli" for 0/1
  presence-absence data, including multi-column 0/1 matrices.

## Changes in default behavior

* summary() now reports the ALE-based importance by default (importance = "ale"), the previous behavior is importance = "permutation", which is now reported as the relative increase in error (previously the ratio)
* ALE() computes the curve analytically from the conditional effects by default (analytical = TRUE)
* continue_training() no longer has a verbose argument, the setting of the original fit is reused
* the conditional effects calculated after training are now based on at most 5000 observations, and on fewer when the model has many features, because the stored effects grow with the number of features squared. The averages they are used for are unaffected (below 2% deviation from the full data at 2000 observations). dnn(ce = <number>) sets the number of observations, dnn(ce = FALSE) turns them off
* the ALE-based importance now weights the curve by the empirical frequency of the feature values instead of by a kernel density estimate. The density weighting squared the empirical density, which over-weighted the mode and recovered only about 60% of the variance of the predictions in an additive simulation
* Depends on R (>= 4.1.0)

## Bug fixes

* models saved with saveRDS are now rebuilt via build_model() when reloaded (invalid pointer state)
* baseloss and burnin are now forwarded to bootstrap replicates, and weighted (custom) losses receive the weights as torch tensor when the baseloss is computed
* ACE() (conditionalEffects(..., method = "finite")) now returns the same object structure as ACEanalytical() (both scales, vars, row_indices), return_vars works with the finite method
* summary(), plot() and ALE() work for models fitted with ce = FALSE
* conditionalEffects(method = "finite", interactions = TRUE) scaled the cross derivatives by the step size of the first feature squared instead of by the product of the two step sizes, so every off-diagonal effect was wrong by the ratio of the two feature standard deviations
* conditionalEffects(data = , subsample = ) drew the subsample size from the training data and failed whenever the new data had fewer rows
* the ALE weights are normalised for every weighting scheme; weighted = FALSE previously returned unnormalised weights and produced meaningless importances
* formulas with derived terms (dnn(y ~ a*b), dnn(y ~ log(a) + b)) no longer fail with "undefined columns selected" while the factor levels are collected. The derived columns are still listed as separate variables by the xAI functions, which report an importance of zero for them
* ALE(analytical = TRUE) uses the conditional effects of the data it is given instead of the ones stored from the training data
* continue_training() updates the stored conditional effects, summary() and ALE() no longer report the effects of the model before the call
* hyperparameter tuning works with a custom loss function that is not defined in the global environment
* mmn() resolves symbols in the formula in the environment of the formula, so it can be called via do.call() or from within another function
* mmn() accepts custom submodules for both dnn() and cnn() without warning
* weights are rejected with an informative error for likelihoods that cannot use them, instead of failing during the first epoch
* cnn(weights = ) works for factor responses
* config_lr_scheduler("multiplicative") passes lr_lambda on, the scheduler failed on the first step before
* summary() of a bootstrapped model works when the replicates were fitted with ce = FALSE
* the nbinom likelihood exposes the dispersion via parameter_link() again, and simulate() works
* permutation importance no longer has an unreachable exhaustive branch that errored for large n_permute
* visualize.training() no longer errors when the loss is NA for every epoch (aborted training)
* dnn(), cnn() and mmn() no longer hang on data sets with 5 or fewer observations. The default batchsize is round(0.1*n), which was 0 there, and a dataloader with batch_size = 0 has length Inf and yields empty batches forever
* predict(model, type = "class") works for bootstrapped models. The replicates are now averaged on the response scale and the class is taken from the aggregate, previously the per-replicate factors were passed to abind() and the call errored
* predict(model, newdata = ) works when newdata contains NAs. model.matrix() drops those rows, but the sample names and the embedding codes were still taken from the full newdata, which errored or misaligned the predictions
* config_optimizer() accepts vector-valued parameters (adam's betas, rprop's etas and step_sizes). The parameter names were dropped when the optimizer was built, so those values were passed positionally and shifted every following argument
* continue_training() passes changed_params on to the replicates of a bootstrapped model instead of ignoring it
* blocked hyperparameter tuning no longer warns about a recycling mismatch when the number of blocks is not a multiple of CV
* conditionalEffects(data = ) works when the new data contains NAs. Those rows are dropped from the model frame, but the row indices were still drawn from the full data, which aborted the analytical method inside torch ("index out of bounds") and made the finite method fail on a length mismatch
* the mvp likelihood returns one value per observation like every other likelihood, instead of a single reduced value
* the development notes at the repository root (and E-CNN_and_MMN.md) are excluded from the build, they were shipped in the tarball before
* images read from a folder are no longer scaled down by a factor of 255. base_loader() returns values in [0, 1] and cito divided by 255 on top of that, so JPEG and PNG inputs reached the network in [0, 1/255] while TIFF inputs, which were not divided, reached it in [0, 1]
* multi-band TIFFs that store their bands as separate planes (GDAL's INTERLEAVE=BAND) are no longer read incorrectly. tiff::readTIFF() returns wrong pixel values for that layout without raising an error, so models were trained on corrupted inputs. cito now reads such files through its compiled decoder, and the fallback reader refuses them with an explanatory message instead of continuing silently
* folder images are decoded by a compiled reader when libjpeg, libpng and libtiff are available at install time, which is 9 to 14 times faster than the previous path and never materialises an R double array. Without those libraries cito falls back to the R readers, one format at a time
* GeoTIFF tags no longer raise four warnings per file
* folder inputs with .JPG, .PNG, .JPEG or .tif extensions are read instead of being rejected. The file type was matched case-sensitively, .tif was missing entirely, and the pattern was unanchored with an unescaped dot, so a file called "x.png.bak" was passed to the image reader while "photo.JPG" was not
* RGBA images keep their alpha channel on every decode path
* folder inputs are decoded through magick when it is installed, which builds the tensor from raw bytes instead of materialising an R double array first. Measured 3.2x on 1024px images; no change at 224px
* cnn() gains num_workers for folder inputs. The dataset no longer stores tensors, which previously made torch's parallel dataloader fail with "external pointer is not valid"
* weighted cross-validation in the hyperparameter tuning subsets matrix weights by row
* print() of a citoarchitecture without an input_shape no longer errors
* models fitted with cito <= 1.1 are rejected with an informative message instead of an internal error
* cito no longer registers head() methods for formulas, calls and names


# cito 1.1

## New features
* hyperparameter tuning (experimental)
* burnin parameter
* multivariate probit model
* X and Y support (alternative interface)
* negative binomial distribution

## Minor changes
* Improved vignette
* Improved README

## Bug fixes
* dropout is turned off after training (into evaluation mode)
* predict type was changed
* small bug in the activation functions
* extended support for mps devices



# cito 1.0.2

## New features
* conditional Effects (approximate linear effects)
* uncertainties via bootstrapping (can be forwarded to all functions)
* summary() can return standard errors and p-values for xAI metrics
* improved documentation / several new vignettes
* baseline loss 
* loss = inf/na is not captured, training is aborted and user will be warned
* mps (M1/M2 gpu) device is now supported
 
## Bug fixes
* early stopping (ignored validation loss)
* weights are only saved for best and last epoch
* gaussian likelihood works now properly
* regularization loss is not visualized
* reduce lr on plateau works now with validation loss


# cito 1.0.1

## New features 
* predict function can now return directly the class
* custom loss and parameter can now also be optimized
* summary function (importances) does now support loss = binomial


## Minor changes 
* print of summary is now more clear 

## Bug fixes
* in ALE function providing new data did not work properly
* Performance improvements with new dataloader
* ALE/PDP work now correctly for softmax
* PDP ICE return now correct curves
* Early stopping works now
* lr reducer on plateau didn't reduce lr
* Predictions are now made on cuda of the model is stored on cuda
