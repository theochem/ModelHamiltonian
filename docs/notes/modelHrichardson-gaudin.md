# Richardson-Gaudin Model Tutorials

The Richardson-Gaudin (RG) models are exactly solvable quantum many-body systems.

## Mathematical Formulation

### Standard Richardson-Gaudin Model
The Hamiltonian is given by:
$$ \hat{H}_{RG} = \sum_p (\mu_p^Z - J_{pp}^{eq}) S_p^Z + \sum_{pq} J_{pq}^{eq} S_p^+ S_q^- $$

where:
- $\mu_p^Z$ represents the Zeeman term
- $J_{pq}^{eq}$ is the equatorial interaction term
- $S_p^{\pm}$ are spin raising/lowering operators
- $S_p^Z$ is the spin projection operator


### Picket-Fence Model
A special case with equally spaced levels:
$$ \mu_p^Z = p\Delta $$
where $\Delta$ is the level spacing.

### Conserved Quantities
The model has $N$ conserved quantities:
$$ R_p = S_p^Z + g\sum_{q\neq p}\frac{S_p^+ S_q^- - S_p^- S_q^+}{\mu_p^Z - \mu_q^Z} $$

### Bethe Ansatz Solution
The eigenvalues are given by:
$$ E = \sum_{p=1}^M E_p $$
where $E_p$ satisfy the Richardson equations:
$$ \frac{2}{g} + \sum_{q=1}^N \frac{1}{\mu_q^Z - E_p} - \sum_{q\neq p} \frac{2}{E_q - E_p} = 0 $$

The pairing correlation is:
$$ \Delta = \frac{g}{N}\sum_{p,q} \langle S_p^+ S_q^- \rangle $$

## Basic Richardson-Gaudin Model
For API reference, see [here](https://modelh.qcdevs.org/api/spin.html#moha.HamRG).
### Setup and Implementation
```python
from moha import HamRG
import numpy as np

n_sites = 6
J_eq = 1.0 
mu = np.zeros(n_sites)


connectivity = np.zeros((n_sites, n_sites))
for p in range(n_sites):
    for q in range(n_sites):
        if p != q:
            connectivity[p, q] = 1
            
ham = HamRG(mu=mu, J_eq=J_eq, connectivity=connectivity)

h = ham.generate_one_body_integral(basis='spatial', dense=True)
v = ham.generate_two_body_integral(basis='spatial', dense=True, sym=4)
```

### Key Properties
- Uniform coupling between all pairs of sites
- Conserved total spin
- Exactly solvable through the Bethe Ansatz

## Picket-Fence Richardson-Gaudin Model

### Setup and Implementation
```python
n_sites = 8
level_spacing = 1.0
g = 0.5

mu = np.arange(n_sites) * level_spacing
ham = HamRG(mu=mu, J_eq=g)

h = ham.generate_one_body_integral(dense=True)
v = ham.generate_two_body_integral(dense=True)
```

### Physical Properties

#### Phase Transitions
The picket-fence model shows quantum phase transition as a function of the $g$:
- **Weak coupling** ($g \approx 0$): Nearly independent particles
- **Strong coupling** ($g \gg \Delta$): Strongly correlated pairs
- **Critical point**: $g_c \approx \Delta$

#### Wavefunction Evolution observations
As the coupling strength increases, the wavefunction shows interesting behavior:
1. $|\langle \Psi(0)|\Psi(g) \rangle| \to 0$ for large systems