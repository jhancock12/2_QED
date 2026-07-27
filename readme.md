Tools for studying QED in two spatial dimensions using Pauli spin variables

This repository provides a complete Python toolkit for studying (2+1)-dimensional quantum electrodynamics (QED) in the staggered fermion formulation, mapped to Pauli spin operators. The code is designed for both classical (exact diagonalisation) and quantum (variational quantum eigensolver) simulations, with a focus on Hamiltonian lattice gauge theory.

It was developed as part of a PhD thesis investigating the effects of a background electric field on screening, confinement, and chiral symmetry breaking in compact (2+1)-QED.

---

Key features

- Hamiltonian builder – constructs the full (2+1)-D QED Hamiltonian from a set of physical parameters (mass, coupling, lattice spacing, static charges, background electric field) using the Kogut–Susskind formulation.
- Flexible lattice class – supports arbitrary rectangular lattices, Gauss‑law reduction, and choice of dynamical links.
- Gray‑code truncation – maps the gauge field to qubits with a user‑specified number of levels per link.
- Multiple solvers – includes exact (sparse/dense diagonalisation) and approximate (VQE) solvers, with the ability to compute a wide range of observables:
  - Energy, particle number, chiral condensate
  - Electric and magnetic field expectation values
  - Gauss‑law violation checks
- Quantum error mitigation – implements measurement error mitigation, symmetry verification, and zero‑noise extrapolation for noisy simulations.
- Optimisation routines – includes Natural Gradient SPSA (NG‑SPSA) and other classical optimisers for VQE.
- Plotting and visualisation – built‑in functions to visualise lattice field configurations and generate publication‑ready plots.

---
Project structure
```
├── hamiltonian_class.py        # Pauli‑string Hamiltonian builder (sparse and dense)
├── lattice_class.py            # Lattice geometry, Gauss‑law reduction, link management
├── operators.py                # Individual Hamiltonian terms (mass, kinetic, magnetic, electric)
├── observables.py              # Wrappers to build the full QED Hamiltonian
├── solvers/
│   └── sparse.py               # Sparse diagonalisation and observable extraction
│   └── dense.py                # Dense diagonalisation and observable extraction
│   └── noiseless.py            # Quantum circuit statevector-based optimization and observable extraction
│   └── noisy.py                # Quantum circuit shots-based optimization and observable extraction
│   └── linear_operator.py      # Linear operator-based optimization and observable extraction
│   └── classical_optimizers.py # Classical optimization routines for quantum circuit simulations
├── plotter.py                  # Publication‑ready plotting (with square/non‑square options)
├── visualization.py            # Lattice field visualisation (charge, electric field)
├── global_helpers.py           # Utility functions (rounding, dictionary printing)
└── generate_data.py            # Example data‑generation script (full sweep)
```
---
Installation & dependencies

The code requires Python 3.8+ and the following libraries:

- numpy
- scipy
- matplotlib
- sympy (for symbolic Gauss‑law reduction)
- qiskit (for VQE simulations)

---

Author

James Leonard Hancock
james.leo.business@gmail.com
University of Plymouth
