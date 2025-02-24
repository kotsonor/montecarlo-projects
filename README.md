# Monte Carlo Simulations and PRNG Analysis

This repository contains two projects developed as part of the *Introduction to Simulations and Monte Carlo Methods* course.

## Projects

### PRNG Analysis (`prng/`)
This project investigates various pseudorandom number generators (PRNGs), including Linear Congruential Generators (LCG), Generalized LCG (GLCG), RC4, and the Mersenne Twister. 
It evaluates the statistical properties of these generators using tests like Chi-Square, Kolmogorov-Smirnov, and the Frequency Test within a Block. 
In addition, second-level testing is performed by subdividing the generated sequences into multiple subsets and analyzing the distribution of p-values to further verify uniformity and randomness.

### Variance Reduction in Monte Carlo Simulations (`variance-reduction/`)
This project applies Monte Carlo methods to financial models by simulating Brownian motion to price European and Asian options. 
It focuses on improving simulation efficiency through variance reduction techniques such as antithetic variates and control variates.

PDF reports for each project are included in the repository for detailed explanations of the methodologies and results.
