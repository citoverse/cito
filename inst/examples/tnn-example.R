\donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(42)
n = 300
env = data.frame(temp = rnorm(n), prec = rnorm(n))
Y = sapply(1:8, function(j) rbinom(n, 1, plogis((j - 4) / 2 * env$temp)))
colnames(Y) = paste0("sp", 1:8)

# The response as tokens: a masked, conditionally queryable model
fit = tnn(Y ~ dnn(~ temp + prec, data = env) + tokens(Y), dataList = list(Y = Y, env = env),
          loss = "bernoulli", epochs = 20L, validation = 0.2, plot = FALSE,
          groups = list(first = c("sp1", "sp2", "sp3", "sp4")))

# unconditional: every species from the environment
head(predict(fit, type = "response"))
# the first four species given the observed others
head(predict(fit, Y = Y, query = first ~ ., type = "response"))

# A transformer instead of the fusion DNN of mmn(), read out from a cls token
y = matrix(env$temp + rnorm(n, sd = 0.2))
fit2 = tnn(y ~ dnn(~ temp, data = env) + dnn(~ prec, data = env), dataList = list(y = y, env = env),
           epochs = 10L, plot = FALSE)
head(predict(fit2))
}
}
