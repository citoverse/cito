# cito 1.0.2

## New features
* conditional Effects (approximate linear effects)
* uncertainties via bootstrapping (can be forwarded to all functions)
* summary() can return standard errors and p-values for xAI metrics
* improved documentation / several new vignettes
* baseline loss 
* loss = inf/na is not captured, training is aborted and user is warned
* mps (M1/M2 gpu) device is not supported
 
## Bug fixes
* early stopping (ignored validation loss)
* weights are only saved for best and last epoch
* gaussian likelihood works now properly
* reguarlization loss is not visualized
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
