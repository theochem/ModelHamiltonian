# H&uuml;ckel Model Hamiltonian

## The H&uuml;ckel Hamiltonian 
The Hamiltonian is a one-electron site-interaction model, with one orbital (traditionally a $\pi$ orbital) per site. 
$$ \hat{H} = \sum_{p,q=1}^n h_{pq} a^{\dagger}_p a_q $$(1)
Traditionally there are two parameters: 
- an $\alpha_{pp}$ parameter that is the diagonal terms, representing the energy associated with an electron on the site $p$ 
- a $\beta_{pq}=\beta_{qp}$ parameter that is the resonance/hopping/bond term between sites $p$ and $q$. 
 
> Note: Traditionally H&uuml;ckel theory works by assuming that orbitals on different sites do not overlap. H&uuml;ckel Hamiltonians are always restricted.

Traditionally, the values of $\alpha$ and $\beta$ are defined for the 2p orbital in an $sp^2$ hybridized carbon atom, with reasonable values being:
$$ \alpha = -11.26 \text{ eV} = 0.414 \text{ Hartree}$$(2)
$$ \beta = -1.45 \text{ eV} = 0.0533 \text{ Hartree}$$(3)
The $\alpha$ value is chosen from the binding energy (minus one times the ionization potential) of the Carbon atom; the $\beta$ value is defined by Rauk based on the carbon-carbon bond strength in ethylene, and is on the high end of the "accepted" range of values.
For heteroatoms, these values are shifted by:
$$ \alpha_X = \alpha + h_X \cdot |\beta| $$(4)
$$ \beta_{XY} = k_{XY} |\beta|$$(5)
A table of these values is given by Rauk. 

## Input Formats
### Atom-type input of connectivity with (predefined/precomputed) parameters:
The user can pass in a list of atom types and their connectivity in the Gaussian format
> Atom-index Atom-index-bonded-to (-1, 0, or bond length) Atom-index-bonded-to (-1, 0, or bond length) ...
The list of atom labels can include anything from Rauk's table; if that is done, then a -1 in the above input indicates that the atoms are connected; a zero indicates that they are not connected. If a non-supported atom-type is used, or if the user wishes to explicitly supply bond lengths and not just connectivity data, the ionization potential (electron binding energy) will be used to define $\alpha$ and the Wolfsberg-Helmolz approximation to define $\beta$. 
$$ \alpha_X = - (\text{ionization potential of atom X})$$(6)
$$ \beta_{XY} = 1.75 S_{XY} \frac{\alpha_X+\alpha_Y}{2}$$(7)
The second approximation is the Wolfsberg-Helmholz approximation. We don't expect the user to pass an overlap matrix (and, indeed, at least the high-level-API we need not support this), but a reasonable approximation is obtained by the following procedure.

First, define the variables:
$$
p = -\frac{\alpha_X + \alpha_Y}{2} R_{XY}
$$(8a)
$$
t = \left| \frac{\alpha_X - \alpha_Y }{\alpha_X + \alpha_Y} \right|
$$(8b)
$$
a_n(x) = n! \sum_{k=1}^{n+1} \dfrac{1}{x^k(n-k+1)!}
$$(8c)
$$
b_n(x) = n! \sum_{k=1}^{n+1} \dfrac{(-1)^{n-k}}{x^k(n-k+1)!}
$$(8c)
where $R_{XY}$ is the distance between atoms $X$ and $Y$. The overlap between two $2p$ orbitals binding in the $\pi$ orientation can be deduced from the reference of Mulliken, Rieke, Orloff, and Orloff (MROO) as:
$$
S_{XY} = 
\begin{cases}
e^{-p}(1+p+\tfrac{2}{5}p^2+\tfrac{1}{15}p^3) \qquad &t=0 \text{ (Eq. 61 in MROO)} \\
\dfrac{(1-t^2)^{5/2}p^5}{120}\left[A_4(B_0 - B_2) - A_2 (B_0 - B_4) + A_0(B_2 - B_4) \right] \qquad &t \ne 0 \text{ (Eq. 46 in MROO)}
\end{cases}
$$(9)
where
$$
A_n = e^{-p} a_n(p)
$$(10)
$$
B_k = 
\begin{cases}
\frac{2}{k+1} \qquad &t=0 \\
-e^{-pt} a_n(pt) - e^{pt} b_n(pt) \qquad &t > 0
\end{cases}
$$(11)

If a non-supported atom type is invoked but a bond length is not selected, a default bond length of 90% of the sum of the atoms' covalent radii is used. (The 90% is to compensate for the fact there is a multiple bond.) These approximations are all very rough, but it's just intended for ease; the "direct input" option provides all the flexibility that can possibly be required. The above integral is the overlap integral for two $p$ orbitals in $\pi$-binding geometry.  These formulas, which I transcribed from MROO, should be double-checked. One unit test is to affirm that the $t=0$ limit is recovered when $t \rightarrow 0^+$. Another is to check that $R\rightarrow 0^+$, $S_{XY} = (1-t^2)^{5/2}$ (Eq. 68 in MROO).

An alternative is available when we know a value of $\beta$ for a reference (equilibrium geometry, $R_0$), using *Mulliken's magic formula*. The solution is to use:
$$
\beta(R) = \beta(R_0) \frac{S(R)}{S(R_0)}
$$
For 2pz orbitals in a $\pi$-binding configuration, one has:
$$
S(R) = \tfrac{1}{15} e^{-\rho} (\rho^3 + 6 \rho^2 + 15 \rho + 15)
$$
with
$$
\rho = \xi R
$$
(with $R$ measured in atomic units of Bohr). Paldus and Chin indicate that one-half the effective nuclear charge should be used for $\xi$, and give a value of $\xi=1.625$ for the Carbon atom, based on Slater's rules. That gives a quite reasonable prescription, and is noted to correspond beautifully to the overlap formula above. The Mulliken-magic-formula should be preferred over the Wolfsberg-Helmholz approximation where there is enough data. 


### Direct input of parameters: 
The user can pass in the $h_{pq}$ matrix directly as an $n \times n$ matrix, where the diagonal entries are interpreted as $\alpha_{pp}$ and the off-diagonal entries are interpreted as $\beta_{pq}$. Optionally, the user can pass in explicit values for $\alpha$ and/or $\beta$. If so, then $h_{pq}$ is used to define atomic connectivity. Specifically, non-zero off-diagonal elements get replaced by $\beta$ and nonzero diagonal elements get replaced by $\alpha$. Zero elements of $h_{pq}$ remain zero. 
$$
h_{pq} = 
\begin{cases}
\alpha \qquad &p=q \text{ and } h_{pp} \ne 0 \\
\beta \qquad &p \ne q \text{ and } h_{pq} \ne 0 \\
0 \qquad &h_{pq} = 0
\end{cases}
$$(12)
Only atomic-connectivity data *or* direct input is supported. Direct specification of $\alpha$ and $\beta$ is ignored if connectivity information is provided. Connectivity information is ignored if $h_{pq}$ is provided.

### Restricted or Unrestricted or Generalized?
H&uuml;ckel theory always uses restricted Hamiltonians. So $h_{p \alpha q \alpha} = h_{p \beta q \beta}$ and $h_{p \alpha q \beta} = h_{p \beta q \alpha} = 0$.

### References
- Parameters from [Arvi Rauk, The Orbital Interaction Theory of Organic Chemistry , Second Edition, Wiley-Interscience, New York, 2001.](https://onlinelibrary.wiley.com/doi/book/10.1002/0471220418).
- Overlaps of $\pi$-bonding $2p$ orbitals from [R. S. Mulliken, C. A. Rieke, D. Orloff, and H. Orloff, "Formulas and Numerical Tables for Overlap Integrals", *J. Chem. Phys.* **17**, 1248-1267 (1949)](https://aip.scitation.org/doi/10.1063/1.1747150). 
- Mulliken Magic Formula from R. S. Mulliken, J. Chim. Phys. 46, 675 (1949).Direct reference was Eq. (6) of [J. Paldus and E. Chin, "Bond Length Alternation in Cyclic Polyenes. I. Restricted Hartree-Fock Method, *Int. J. Quantum Chem.* **24** 373-394 (1983).](https://onlinelibrary.wiley.com/doi/10.1002/qua.560240405)