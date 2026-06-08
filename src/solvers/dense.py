# Standard libraries

# Local modules
from global_helpers import smart_round
from hamiltonian_class import Hamiltonian
from lattice_class import Lattice
from operators import electric_n_direction, magnetic_term_n, particle_n_hamiltonian, charge_n_hamiltonian

# Third-party libraries
import numpy as np

def dense_statevector_solver(hamiltonian : Hamiltonian):
    if not isinstance(hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian must be a hamiltonian_class.Hamiltonian, you have entered a {type(hamiltonian)}")

    H = hamiltonian.to_matrix()
    H = H.astype(np.complex128)

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    groundstate_energy = eigenvalues[0].real
    groundstate = eigenvectors[:, 0]

    return groundstate_energy, groundstate


def _normalize_statevector(psi_vec, lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    psi = np.asarray(psi_vec, dtype = np.complex128).reshape(-1)
    if len(psi) != 2 ** lattice.n_qubits: raise ValueError(f"Input statevector must have length 2^n_qubits = {2 ** lattice.n_qubits}, you have entered a statevector of length {len(psi)}")

    norm = np.linalg.norm(psi)
    if norm == 0: raise ValueError("Input statevector has zero norm.")

    return psi / norm

def electric_field_values_from_statevector(psi_vec, lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    psi = _normalize_statevector(psi_vec, lattice)
    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[site_n]:
            H = electric_n_direction(lattice, site_n, direction)
            H_dense = H.to_matrix()
            values[(lattice.labels[site_n], direction)] = np.real(np.vdot(psi, H_dense @ psi))

    return values

def electric_field_values_squared_from_statevector(psi_vec, lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    psi = _normalize_statevector(psi_vec, lattice)
    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[site_n]:
            H = electric_n_direction(lattice, site_n, direction)
            H.multiply_by_hamiltonian(H)
            H_dense = H.to_matrix()
            values[(lattice.labels[site_n], direction)] = np.real(np.vdot(psi, H_dense @ psi))

    return values

def magnetic_field_values_from_statevector(psi_vec, lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    psi = _normalize_statevector(psi_vec, lattice)
    values = {}

    for site_n in lattice.plaquettes:
        H = magnetic_term_n(lattice, site_n)
        H_dense = H.to_matrix()
        values[lattice.labels[site_n]] = np.real(np.vdot(psi, H_dense @ psi))

    return values
    
def charge_values_from_statevector(psi_vec, lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    psi = _normalize_statevector(psi_vec, lattice)
    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        H = charge_n_hamiltonian(lattice, site_n)
        H_dense = H.to_matrix()
        values[lattice.labels[site_n]] = np.real(np.vdot(psi, H_dense @ psi))

    return values

def particle_number_values_from_statevector(psi_vec, lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    psi = _normalize_statevector(psi_vec, lattice)
    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        H = particle_n_hamiltonian(lattice, site_n)
        H_dense = H.to_matrix()
        values[lattice.labels[site_n]] = np.real(np.vdot(psi, H_dense @ psi))

    return values
    
def observes_reduced_from_statevector(psi_vec, lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    psi = _normalize_statevector(psi_vec, lattice)

    gauss_equations = lattice.gauss_equations
    ef = electric_field_values_from_statevector(psi, lattice)
    ef_sq = electric_field_values_squared_from_statevector(psi, lattice)
    mf = magnetic_field_values_from_statevector(psi, lattice)
    pn = particle_number_values_from_statevector(psi, lattice)
    c = charge_values_from_statevector(psi, lattice)

    total_pn = sum(pn.values())
    total_charge = sum(c.values())

    for dependant_variable in gauss_equations['dependant_variables']:
        equation = gauss_equations['solution'][dependant_variable]
        sub_ins = {}

        for var in gauss_equations['independant_variables']:
            if var in list(gauss_equations['reverse_link_variable_dict']):
                sub_ins[var] = ef[gauss_equations['reverse_link_variable_dict'][var]]
            else:
                site_n_charge = int(str(var)[1:])
                sub_ins[var] = c[lattice.labels[site_n_charge]]

        ef[gauss_equations['reverse_link_variable_dict'][dependant_variable]] = complex(equation.subs(sub_ins))

    site_n = 0
    gl = {}
    for equation in gauss_equations['solution']:
        sub_ins = {}

        for var in gauss_equations['independant_variables'] + gauss_equations['dependant_variables']:
            if var in list(gauss_equations['reverse_link_variable_dict']):
                sub_ins[var] = ef[gauss_equations['reverse_link_variable_dict'][var]]
            else:
                site_n_charge = int(str(var)[1:])
                sub_ins[var] = c[lattice.labels[site_n_charge]]

        gl[lattice.labels[site_n]] = complex(equation.subs(sub_ins))
        site_n += 1

    ef = smart_round(ef, 6)
    ef_sq = smart_round(ef_sq, 6)
    mf = smart_round(mf, 6)
    pn = smart_round(pn, 6)
    c = smart_round(c, 6)
    gl = smart_round(gl, 6)

    total_pn = smart_round(total_pn, 6)
    total_charge = smart_round(total_charge, 6)

    return {
        'electric_field_dict': ef,
        'electric_field_squared_dict': ef_sq,
        'charge_field_dict': c,
        'magnetic_field_dict': mf,
        'particle_number_dict': pn,
        'gauss_law_dict': gl,
        'particle_number_total': total_pn,
        'charge_total' : total_charge
    }

def solve_and_observe_dense(hamiltonian : Hamiltonian, lattice : Lattice):
    if not isinstance(hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian must be a hamiltonian_class.Hamiltonian, you have entered a {type(hamiltonian)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    groundstate_energy, groundstate = dense_statevector_solver(hamiltonian)
    return observes_reduced_from_statevector(groundstate, lattice)