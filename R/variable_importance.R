get_importance<- function(model, n_permute= NULL, data = NULL, device = "cpu"){

  if(is.null(n_permute)) n_permute <- ceiling(sqrt(nrow(model$data$data))*3)
  model<- check_model(model)
  softmax = FALSE
  if(inherits(model$loss$call, "character")) {
    if(!any(model$loss$call  == c("softmax","mse", "mae"))){ return(NULL)}
    if(model$loss$call  == "softmax") {
      softmax = TRUE
    }
  } else {
    if(!any(model$loss$call$family == c("binomial")  )){ return(NULL)}
  }
  loss<- model$loss$loss

  true = model$data$Y

  if(inherits(model$loss$call, "character")) {
    true = torch::torch_tensor(model$data$Y)
  } else {
    if(model$loss$call$family  == c("binomial") ){
      mode(true) = "numeric"
      true = torch::torch_tensor(true)
    }
  }

  n_outputs = model$model_properties$output

  if(softmax) {
    n_outputs = 1
  }

  out = NULL

  for(n_prediction in 1:n_outputs) {

    if(n_outputs > 1) true_tmp = true[,n_prediction,drop=FALSE]
    else true_tmp = true

    if(!softmax) org_err <- as.numeric(loss( pred = torch::torch_tensor(stats::predict(model,model$data$data, type = "link", device = device)[,n_prediction,drop=FALSE]) ,true = true_tmp)$mean())
    else org_err <- as.numeric(loss( pred = torch::torch_tensor(stats::predict(model,model$data$data, type = "link", device = device)) ,true = true_tmp)$mean())

    importance <- data.frame(variable = get_var_names(model$training_properties$formula, model$data$data[1,]),
                             importance = c(0))

    for(i in seq_len(nrow(importance))){

      new_err <-c()
      if(n_permute < ((nrow(model$data$data)**2)-1)){
        for(k in seq_len(n_permute)){

          perm_preds <- c()

          perm_data <- model$data$data
          perm_data[, importance$variable[i]] <- perm_data[sample.int(n = nrow(perm_data),replace = FALSE),importance$variable[i]]

          if(!softmax) perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device)[,n_prediction,drop=FALSE])
          else perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device))

          new_err <- append(new_err, as.numeric(loss(pred = torch::torch_tensor(perm_preds),
                                                     true = true_tmp)$mean() ))


        }
      }else{
        for(j in seq_len(nrow(model$data$data))){
          perm_data <- model$data$data[j,]
          for(k in seq_len(nrow(model$data$data))[-j]){
            perm_data[i] <- model$data$data[k,i]
            if(!softmax) perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device)[,n_prediction,drop=FALSE])
            else perm_preds <- rbind(perm_preds, stats::predict(model, perm_data, type = "link", device = device))
            true <- append(true_tmp, model$data$Y[j])
          }
        }
      }

      importance$importance[i] <- mean(new_err)/org_err

    }
    colnames(importance)[2] = paste0("importance_", n_prediction)
    if(n_prediction > 1) importance = importance[,2,drop=FALSE]
    out[[n_prediction]] = importance

  }

  return(do.call(cbind, out))
}
