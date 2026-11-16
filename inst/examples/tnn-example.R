\donttest{
if(torch::torch_is_installed()){
library(cito)

# A joint species distribution model: 55 butterfly species, the community enters as tokens
data(butterflies)
Y = butterflies$PA
env = as.data.frame(scale(butterflies$env))
fit = tnn(Y ~ dnn(~ ., data = env) + tokens(Y), dataList = list(Y = Y, env = env),
          loss = "bernoulli", layers = 1L, epochs = 10L, validation = 0.2, plot = FALSE,
          groups = list(fritillaries = grep("fritillary", colnames(Y), value = TRUE)))

# unconditional: every species from the environment
head(predict(fit, type = "response")[, 1:5])
# the fritillaries given the observed other species
head(predict(fit, Y = Y, query = fritillaries ~ . - fritillaries, type = "response")[, 1:5])
# each species from all others
p_loo = predict(fit, Y = Y[1:100, ], newdata = list(env = env[1:100, ]), loo = TRUE, type = "response")

# A transformer instead of the fusion DNN of mmn(), read out from a cls token
y = matrix(rowSums(Y))
fit2 = tnn(y ~ dnn(~ climate, data = env) + dnn(~ blwood + conwood + chalk_limestone, data = env),
           dataList = list(y = y, env = env), loss = "poisson", epochs = 10L, plot = FALSE)
head(predict(fit2, type = "response"))
}
}
