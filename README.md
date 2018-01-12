# Adversarial Machine Translation

Experimental Neural Machine Translation model,
based on Transformer Networks with Adversarial Target.

Instead of learning translation model by Likelihood maximization,
this model relies on minimization Wasserstein-1 distance between real
translations and parametrized with this model distribution.