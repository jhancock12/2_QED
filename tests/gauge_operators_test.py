# Standard libraries
import itertools
import copy

# Local modules
from hamiltonian_class import Hamiltonian

# Third-party libraries
import numpy as np

def gauge_terms(n_g):
    matrix_dict = {
        'I': np.array([[1, 0], [0, 1]], dtype = complex),
        'X': np.array([[0, 1], [1, 0]], dtype = complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype = complex),
        'Z': np.array([[1, 0],[0, -1]], dtype = complex)
        }

    # --- E_terms ---
    I_string_list = list('I' * n_g)
    coeff = -0.5
    E_terms = {}
    for k in range(n_g - 1):
        temp_string = copy.copy(I_string_list)
        temp_string[k] = 'Z'
        E_terms["".join(temp_string)] = coeff * 2**(k)

    first_string = copy.copy(I_string_list)
    first_string[n_g - 1] = 'Z'
    E_terms["".join(first_string)] = coeff * (2**(n_g - 1) - 1)

    # --- raising-operator mapping, derived from the E eigenvalues ---
    dim = 2 ** n_g
    diag = np.zeros(dim)
    for term, term_coeff in E_terms.items():
        k = term.index('Z')          # position k in the Pauli string -> weight 2**k in the matrix index
        bit = 1 << k
        for i in range(dim):
            diag[i] += term_coeff if not (i & bit) else -term_coeff

    invalid = 1 << (n_g - 1)
    valid_indices = [i for i in range(dim) if i != invalid]
    valid_indices.sort(key=lambda i: diag[i])

    mappings = {}
    n_valid = len(valid_indices)
    for k in range(n_valid):
        from_ = format(valid_indices[k], f'0{n_g}b')
        to_ = format(valid_indices[(k + 1) % n_valid], f'0{n_g}b')
        mappings[from_] = to_
    inv_str = format(invalid, f'0{n_g}b')
    mappings[inv_str] = inv_str

    # --- U_terms ---
    U_terms = {}
    U = np.zeros((dim, dim), dtype=complex)
    for from_, to_ in mappings.items():
        i = int(from_, 2)
        j = int(to_, 2)
        U[j, i] = 1.0
    for labels in itertools.product("IXYZ", repeat = n_g):
        term = np.array([1])
        for l in labels:
            term = np.kron(matrix_dict[l], term)   # <-- matches Hamiltonian.to_matrix()'s convention

        coeff = np.trace(term.conj().T @ U) / dim
        if abs(coeff) > 1e-12:
            U_terms["".join(labels)] = coeff

    return U_terms, E_terms


n_gs = [2, 3, 4, 5, 6]
for n_g in n_gs:
    print("===========")
    U_terms, E_terms = gauge_terms(n_g)
    E_hamiltonian = Hamiltonian(n_g)
    U_hamiltonian = Hamiltonian(n_g)

    E_hamiltonian.add_hamiltonian(E_terms)
    U_hamiltonian.add_hamiltonian(U_terms)

    zero_state = np.array([1.0] + [0.0]*(2**n_g - 1))

    E_matrix = E_hamiltonian.to_matrix()
    U_matrix = U_hamiltonian.to_matrix()

    print("Start:", zero_state.conj() @ E_matrix @ zero_state)
    for k in range(2 ** n_g - 1):
        zero_state = U_matrix @ zero_state

        print(f"{k}th step:", zero_state.conj() @ E_matrix @ zero_state)

    comm = E_hamiltonian.commutator(U_hamiltonian)

    U_hamiltonian.multiply_by_constant(-1)

    U_hamiltonian.add_hamiltonian(comm)

    total = 0
    for key in U_hamiltonian.terms:
        total += abs(U_hamiltonian.terms[key])**2
    print("Difference:", total)


