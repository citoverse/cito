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
