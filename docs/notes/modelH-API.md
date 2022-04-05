# Model Hamiltonians
The basic purpose of Model Hamiltonians is that in many cases, the low-energy spectrum and qualitative features of a complicated many-body system can be approximated by a simplified model Hamiltonian. This most frequently occurs when the state of every atom/site/group/moiety in a molecule/crystal can be well described by its occupation and/or spin. 
> An excellent introduction to this strategy can be found in [B. J. Powell, "An introduction to effective low-energy hamiltonians in condensed matter physics and chemistry." In: *Computational methods for large systems: electronic structure approaches for biotechnology and nanotechnology*, J. R. Reimers, editor; (Hoboken, Wiley, 2011). Chapter 1, 309–366.](https://arxiv.org/pdf/0906.1640.pdf)

## Objectives
Many model Hamiltonians can be cast into a form that is conveniently solved by standard quantum-chemistry packages. The goal of this package is to automate this transformation. The objectives are to:
- Make it simple for users to specify model Hamiltonians. 
- Output model Hamiltonians in a way that is compatible with `gbasis`. I.e., the output is simply 1- and 2-electron integrals.
- Provide support for `FCIDump` files through the link between `gbasis` and [`IOData`](iodata.qcdevs.org). 
- (Longer term) Provide benchmark data for certain model Hamiltonians, which can be used for assessing approximate methods for solving the corresponding Schr&ouml;dinger equations.
 
## Quantum Chemistry Hamiltonian
The quantum chemistry Hamiltonian consists of 1- and 2-electron integrals. The normal form, in second quantization, is 
$$
\hat{H} = \sum_{pq} h_{pq} a_p^{\dagger} a_q + \tfrac{1}{2} \sum_{pqrs} g_{pqrs} a_p^{\dagger} a_r^{\dagger} a_s a_q
$$
This equation chooses the orbital-indexing convention from [Molecular Electronic Structure Theory (Helgaker, Jorgensen, Olsen)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119019572). The sums are over spin-orbitals,
$$
\phi_p(\mathbf{r})|\sigma_p \rangle = a_p^{\dagger} |\text{vacuum}\rangle
$$
in some cases it is important to explicitly denote spin. Even though it makes our notation a little clunky, in these cases we choose to explicitly declare the spin. E.g. we'll write $a_{p\alpha}^{\dagger}$ or $a_{q\beta}$. When spin is not specified, the default is to assume that all spin-orbital interactions are the same except for where they must vanish by symmetry. I.e.,
$$
h_{p\alpha, q\alpha} = h_{p\beta, q\beta} = h_{pq} \\ 
h_{p\alpha, q\beta} = h_{p\beta, q\alpha} = 0 
$$
$$
g_{p\alpha, q\alpha, r \alpha, s\alpha} = g_{p\beta, q\beta, r \beta, s\beta} = g_{p\alpha, q\alpha, r \beta, s\beta} = g_{p\beta, q\beta, r \alpha, s\alpha} = g_{pqrs}\\ 
$$
In addition, one has that:
$$
g_{p \sigma_p, q \sigma_q, r \sigma_r, s \sigma_s,} = 0 \qquad \text{ if } \sigma_p \ne \sigma_q \text{ and/or } \sigma_r \ne \sigma_s 
$$
There are also matrix elements that must be identical by symmetry because we assume that orbitals are real-valued:
$$
h_{pq} = h_{qp} \\
g_{pqrs} = g_{rspq} = g_{qprs} = g_{rsqp} = g_{pqsr} = g_{srpq} = g_{qpsr} = g_{srqp}
$$

## Model Hamiltonians Based on Occupation Numbers
In many cases, this form is rather impractical. For example, there may be only a few "key" electrons, and the remaining electrons could then be treated with an effective Hamiltonian (on top of a "frozen core") of other orbitals. For example, in some cases, it is more convenient to think only about the occupation number of a atomic (or functional moiety) site in a molecule in a crystal, or only the spin of the site. Occupation numbers are easily written in terms of the second-quantized operators,
$$
\hat{n}_p = a_p^\dagger a_p
$$
The most general occupation-number-ish Hamiltonian we consider is the generalized Pariser-Parr-Pople + pairing (PPP+P) Hamiltonian, 
$$
\hat{H}_{\text{PPP+P}} = \sum_{pq} h_{pq} a_p^\dagger a_q + \sum_p U_p \hat{n}_{p\alpha}\hat{n}_{p\beta} + \frac{1}{2}\sum_{p\ne q} \gamma_{pq} (\hat{n}_{p \alpha} + \hat{n}_{p \beta} - Q_p)(\hat{n}_{q \alpha} + \hat{n}_{q \beta} - Q_q) + \sum_{p \ne q} g_{pq} a_{p \alpha}^\dagger a_{p \beta}^\dagger a_{q \beta} a_{q \alpha}
$$
This Hamiltonian has seniority zero if $h_{p \ne q} = 0$. It includes all possible seniority-zero Hamiltonians. 

The first terms, $h_{pq}$, could be anything, but are usually approximated at the level of [H&uuml;ckel theory](notes/huckel-model-hamiltonian.md). (In the solid-state literature, these are usually denoted $h_{pq} = t_{pq}$.) The $U_p$ term denotes the repulsion of electrons on the same atom/group site, whilst the $\gamma_{pq}$ term denotes the interaction between electrons and (possibly charged; usually $Q_p = 1$ so that the net charge of a system with one electron per site is zero) and other sites (including electrons on those sites). The $g_{pq}$ term captures interactions between electron pairs; the $g_{pq}$ term is redundant with the on-site repulsion, $g_{pp} = U_p$. This is the same as the PPP-Heisenberg model of Karwowski and Flocke.

Special cases of this Hamiltonian include:
- **Pariser-Parr-Pople (PPP).** The PPP model is obtained when one chooses $g_{pq} = 0$. It can be invoked by choosing `g_pair = 0`.
- **extended Hubbard.** The extended Hubbard model corresponds to choosing $Q_p = 0$. It can be invoked by choosing `charges = 0` and `g_pair = 0`. 
- **Hubbard.** The Hubbard model corresponds to choosing $\gamma_{pq} = 0$. It can be invoked by choosing `gamma = 0`. 
- **H&uuml;ckel.** [The H&uuml;ckle model](notes/huckel-model-hamiltonian.md) corresponds to choosing $U_p = \gamma_{pq} = 0$. It can be invoked by choosing `U_onsite = 0` and `gamma = 0`. 
- **electronegativity equalization.** The electronegativity equalization model has a specific form, which can be cast in this form when $g_{pq} = h_{p \ne q} = 0$. 
- **addition of a magnetic field.** A uniform magnetic field oriented in the $z$ direction, $B_z$, splits the one-electron energy levels by:
$$
h_{p \alpha, p \alpha} \rightarrow h_{p \alpha, p \alpha} + \tfrac{g_e}{2} B_z \\
h_{p \beta, p \beta} \rightarrow h_{p \beta, p \beta} - \tfrac{g_e}{2}B_z
$$
Here $g_e = 2.00231$ is the g-factor for the electron (`scipy.constants.value("electron g factor")`).

## Model Hamiltonians Based on Spins
Sometimes it is sensible to assume that sites in a molecule or material are associated with spins; this typically occurs when the number of electrons on a site is basically fixed, but the spin of the electron on the site is variable. 

Spin-operators can be cast in terms of fermion creation/annihilation operators in several different ways. While the most familiar approach is probably the Jordan-Wigner approximation, the most convenient for our purposes is:
$$
S_p^+ = a_{p \alpha}^\dagger a_{p \beta}^\dagger \\
S_p^- = a_{p \beta} a_{p \alpha} \\
S_p^Z = \tfrac{1}{2} \left(a_{p \alpha}^\dagger a_{p \alpha}+a_{p \beta}^\dagger a_{p \beta} - 1\right)
$$
It is easily verified that the key commutation relations for spins are satisfied,
$$
\left[S_p^Z, S_q^{\pm} \right] = \pm \delta_{pq} S_p^\pm \\
\left[S_p^+, S_q^- \right] = 2 \delta_{pq} S_p^Z \\
\left[S_p^\pm, S_q^{\pm} \right] = 0 \\
S_p^+ S_p^- + S_p^- S_p^+ = 1
$$
so one can also define the other Cartesian-spin operators as:
$$
S_p^X =\tfrac{1}{2} \left(S_p^+ + S_p^- \right) \\
S_p^Y =\tfrac{1}{2i} \left(S_p^+ - S_p^- \right)
$$
The most general spin-ish Hamiltonian we consider is the generalized Heisenberg model, but only the XXZ Heisenberg model is amenable to quantum chemistry software, because the others lead to non-number-conserving terms when they are mapped onto the fermion creator/annihilator operators. I.e., the Hamiltonian has terms beyond the quantum-chemical Hamiltonian, terms like $a_{p \beta} a_{p \alpha} a_{q \beta} a_{q \alpha}$ and $a_{p \alpha}^{\dagger} a_{p \beta}^{\dagger} a_{q\alpha}^\dagger a_{q \beta}^\dagger$. The model we consider is, specifically,
$$\hat{H}_{XXZ} = \sum_p \mu_p^Z S_p^Z + \sum_{pq} \left[ J_{pq}^{\text{ax}} S_p^Z S_q^Z + J_{pq}^{\text{eq}} \left(S_p^X S_q^X + S_p^Y S_q^Y \right) 
\right]$$
This can be rewritten in terms of spin-raising and spin-lowering operators, 
$$\hat{H}_{XXZ} = \sum_p \mu_p^Z S_p^Z + \sum_{pq} J_{pq}^{\text{ax}} S_p^Z S_q^Z + \frac{1}{2}\sum_{pq} J_{pq}^{\text{eq}} \left(S_p^+ S_q^- + S_p^- S_q^+ \right) 
$$
and simplified using the commutation relations 
$$\hat{H}_{XXZ} = \sum_p \left(\mu_p^Z - J_{pp}^{\text{eq}}\right)S_p^Z + \sum_{pq} J_{pq}^{\text{ax}} S_p^Z S_q^Z + \sum_{pq} J_{pq}^{\text{eq}} S_p^+ S_q^-  
$$

After this Hamiltonian is rewritten in terms of electronic creation/annihilation operators, it includes all seniority-zero Hamiltonians. 

- This is a general XXZ Heisenberg model with distinct axial and equatorial couplings. If $J^{\text{ax}} = J^{\text{eq}}$ then this is the XXX Heisenberg model.  
- The Ising model is associated with $J^{\text{eq}} = 0$. 
- (Generalized) Richardson-Gaudin Models occur when $J^{\text{ax}} = 0$. An interesting special case is the picket-fence model from [R.W. Richardson, *Phys. Rev.* **141**, 949 (1966)](https://journals.aps.org/pr/abstract/10.1103/PhysRev.141.949), 
$$
\hat{H}_{\text{picket fence}} = \sum_{p=0}^{N-1} \left(p - \frac{N-1}{2}\right) + J^{\text{eq}} \sum_{pq} S_p^+ S_q^-
$$

## Model Hamiltonians Using Spins and Occupation Numbers
For modelling superconductors, it is sometimes useful to embellish every site with both occupation numbers and spins. In the most general case we can support, the new Hamiltonian includes all the terms from the most general occupation-number Hamiltonian, all the terms from the most general spin-Hamiltonian, and a coupling term
$$
\hat{H}_{occ-spin} = \sum_p \varrho_{p} (\hat{n}_{p_\alpha} + \hat{n}_{p_\beta})\hat{S}_p^Z + \sum_p \varsigma_{p} (\hat{n}_{p_\alpha} - \hat{n}_{p_\beta})\hat{S}_p^Z +
\sum_{p \ne q} \varrho_{pq} (\hat{n}_{p_\alpha} + \hat{n}_{p_\beta})\hat{S}_q^Z + \sum_{p \ne q} \varsigma_{pq} (\hat{n}_{p_\alpha} - \hat{n}_{p_\beta})\hat{S}_q^Z
$$
These coupling terms are usually ignored. The following models are directly supported:  
- **[t-J-U-V model.](https://arxiv.org/pdf/cond-mat/0210169.pdf)** In this model, ordinarily all the interaction strengths are the same, but a general model would look like the sum of the general spin and occupation number models, with `charge = 0`, `g_pair = 0`, and the aforementioned spin-occupation coupling terms set to zero:
$$
\hat{H}_{\text{tJUV}} = \sum_{pq} h_{pq} a_p^\dagger a_q + \sum_p U_p \hat{n}_{p\alpha}\hat{n}_{p\beta} + \frac{1}{2}\sum_{p\ne q} V_{pq} (\hat{n}_{p \alpha} + \hat{n}_{p \beta})(\hat{n}_{q \alpha} + \hat{n}_{q \beta}) + \sum_{pq} \left[ J_{pq}^{\text{ax}} S_p^Z S_q^Z + J_{pq}^{\text{eq}} \left(S_p^X S_q^X + S_p^Y S_q^Y \right) - \frac{1}{4}(\hat{n}_{p \alpha} + \hat{n}_{p \beta})(\hat{n}_{q \alpha} + \hat{n}_{q \beta}) \right] $$
- **[t-J model](https://en.wikipedia.org/wiki/T-J_model)** This is the limit of the t-J-U-V model where $U = V = 0$.
- See whether we should add results from [Gus's work](https://arxiv.org/pdf/2107.07922.pdf).


## Forms of Input
- At the most basic level, the user may specify the relevant input interaction matrices directly. This is the low-level code, and is always what happens "under the hood" no matter what type of porcelain might be invoked.
- The user can specify atom types and their connectivity or bond lengths; this can be done either through an adjacency/distance matrix or a format inspired by the Gaussian connectivity input format. 
- The user can specify a lattice; the connectivity/distances are defined intrinsically.  
- Parameters can be passed as arrays (explicit definitions for every atom or atom pair) or dictionaries (explicit definitions for every atom-type or pair-of-atom-types). Default dictionaries are provided.
- Specific special classes of Hamiltonians have reasonable defaults. 
  
## Forms of Output
- 1-electron integrals; 2-electron integrals; overlap matrices
- FCIDump file (via `iodata`)

## References
[Flocke, Norbert and Karwowski, Jacek *Theoretical and Computational Chemistry: Valence Bond Theory* Volume 10 "Symmetric group approach to the theory of Heisenberg lattices." 603–634. doi:10.1016/s1380-7323(02)80020-6](https://sci-hub.mksa.top/10.1016/s1380-7323(02)80020-60)
[Jacek Karwowski and Norbert Flocke, "Relations Between Pariser-Parr-Pople and Heisenberg Models" *Int J. Quantum Chem.* **90** 1091-1098 (2002).](https://onlinelibrary.wiley.com/doi/10.1002/qua.10260)
