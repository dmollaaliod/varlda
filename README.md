# varlda
An implementation of Variational LDA. See the related [blogger blogpost](http://diegosmusings.blogspot.com.au/2015/11/deriving-latent-dirichlet-allocation.html) for the details of how the formulas were derived. But in a nutshell.

Sorry for the LaTeX jargon! I wish github could render the
formulas. To produce a PDF version of this file, try this command:

```
pandoc -t latex -o README.pdf README.md
```


## Generative model

1.  For k = 1 .. K:
    1. $\varphi_k \sim \hbox{Dirichlet}_V(\beta)$
2. For m = 1..M:
    1. $\theta_m \sim \hbox{Dirichlet}_K(\alpha)$
    2. For n = 1 .. N_m:
        1. $z_{mn} \sim \hbox{Multinomial}_K(\theta_m)$
        2. $w_{mn} \sim \hbox{Multinomial}_V(\sum_{i=1}^KZ_{mni}\varphi_i)$

## Algorithm


1. For m=1..M, n=1..N, k=1..K
    1. $p_{z_{mnk}}=1/k$
2. Repeat
    1. For k=1..K, v=1..V
        1. $\beta_{\varphi_{kv}}=\beta+\sum_{m=1}^M\sum_{n=1}^Nw_{mnv}p_{z_{mnk}}$
    2. For m=1..M, k=1..K
        1. $\alpha_{\theta_{mk}}=\alpha+\sum_{n=1}^Np_{z_{mnk}}$
    3. For m=1..M, n=1..N, k=1..K
        1. $p_{z_{mnk}}=\exp\left(\psi(\alpha_{\theta_{mk}}) - \psi\left(\sum_{k'=1}^K\alpha_{\theta_{mk'}}\right) + \sum_{v=1}^Vw_{mnv}\left(\psi(\beta_{\varphi_{kv}})-\psi\left(\sum_{k'=1}^K\beta_{\varphi_{k'v}}\right)\right)\right)$



