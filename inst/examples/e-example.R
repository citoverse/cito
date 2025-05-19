\donttest{
if(torch::torch_is_installed()){

# The following example shows that groups with similar responses will cluster in embedding space
set.seed(123)

n = 10000 # observations
m = 100 # groups / individuals
k = 10 # cluster of groups with the same behavior

dat = data.frame(f1 = runif(n),
                 f2 = runif(n),
                 f3 = runif(n),
                 ind = rep(1:m, each = n/m),
                 cluster = rep(1:k, each = n/k),
                 response = NA)

slopes = matrix(runif(3*k, min = -10,max = 10), nrow = k, ncol = 3)

for(i in 1:k) dat$response[dat$cluster == i] =
  as.matrix(dat[dat$cluster == i, 1:3]) %*% slopes[i,] + rnorm(n/k, sd = 0.2)

mod <- dnn(response~f1+f2+f3 + e(ind,dim = 2),
           data = dat, epochs = 200L, optimizer = config_optimizer("adam"))

embeddings = coef(mod)[[1]][[1]] # extract embeddings
plot(embeddings, col = c(rep(1:k, each = m/k))) # plot clusters in embedding space
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)

ace = conditionalEffects(mod) # extract conditional effects
# now average conditional effects per cluster
ind_ace =
  sapply(1:m, function(ind) {
    tmp = ace[[1]]$result[dat$ind==ind,,]
    return(diag(apply(tmp, 2:3, mean)))
  })

# to create biplot, multiply beta of each cluster with coordinates
coord = ind_ace %*% embeddings/m
arrows(x0 = rep(0, 3), x1 = coord[,1], y0 = rep(0,3), y1 =coord[,2])

}
}
