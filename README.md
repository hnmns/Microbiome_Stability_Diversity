# Microbiome Stability and Diversity

Code appendix for "Relating stability and diversity in ecological
dynamic models" (in prep) by Riley Kouns and Benjamin J. Ridenhour (University of Idaho)

## Methodology overview

In microbial ecology, increased diversity within community is often assumed to favor stability against perturbations and favor persistence of bacterial lineages. We set out to investigate whether stability is related to diversity in two theoretical community types often used to model microbial community dynamics. Specifically, we utilized generalized Lotka-Volterra (GLV) and resource competition (RC) models; these systems rely on notably different mechanisms for species growth and interaction. 

Our basic research approach for RC models is as follows.
* For each species count $`N \in \{2,3,5,10,15\}`$,
    * For each $N \times N$ consumer parameter matrix configuration $`C\in \{C_{L_1}, C_{R_1}, C_{L_2}, C_{R_2}\}`$,
        * Solve for remaining RC model parameters and model equilibrium using given formulas.
        * Compute the resulting community matrix.
        * Compute the eigenvalues of the community matrix evaluated at equilibrium.
        * Compute each of four stability measures.
        * Compute the diversity measure (effective number of species) from the equilibrium vectors found in (3).
* Plot each stability measure against diversity in each combination of species count consumer matrix configuration.


## Code overview

The notebook `microbiome_stab_div.ipynb` contains nearly all relevant code, outlining the specifics of how each model parameter configuration was generated, saved, and plotted.

Functions for the many measures and formulas used throughout are found in modules `Arnoldi_Tools.py`, `Lotka_Volterra_Tools.py`, `Diversity_Measures.py`, and `ODwyer_Tools.py`.

The folder `OD_Systems_Various_C` houses `pickle`'d RC system values, as the sampling method used to generate the systems at larger species counts experienced some slowdown when run locally.