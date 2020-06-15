# Poster
Poster session at CIMAT (Centro de Investigacion en Matematicas) June 10,2020. The repository includes the code to reproduce the results presented. Any doubts please email me at javier.aguilar at cimat dot com 

We present estimation of the parameters of a SIR model applied to Eyam, UK 1666, where there was an outbreak of the black plague. You can read about it here

https://www.bbc.com/news/uk-england-35064071 

https://www.bbc.com/news/uk-england-derbyshire-51904810

The parameters of interest are \alpha, \beta, I(0). We assume

y_i | \alpha, \beta, I(0) \sim Binom(N, R(\alpha, \beta, I(0))/N )

\alpha, \beta \sim Ga(0,1)

I(0) \sim Binom( N, 5/N)

N has a fixed value of 261, the population village. 80% of the population died.

The posterior is multimodal in all parameters. The code uses the emcee and t walk to sample from the posterior. Both samplers find the correct set of parameters. Parallel Tempering experiments were done but are not included (yet!).

---------------------------------------------------------------------------

Files included:

1) Eyam_time_SIR.csv
Contains observed deaths from Black Plague in Eyam, UK 1666
2) Poster.pdf
3) sir-blackplague.py

