# Standard libraries

# Local modules
from global_helpers import smart_round
from lattice_class import Lattice
from hamiltonian_class import Hamiltonian
from circuits.circuit_class import CircuitForLattice
from circuits.measurer_class import CircuitMeasurer
from solvers.classical_optimizers import natural_gradient_SPSA
from operators import electric_n_direction, magnetic_term_n, particle_n_hamiltonian, charge_n_hamiltonian

# Third-party libraries
import numpy as np

def SPSA_vqe_solver_noisy(hamiltonian : Hamiltonian, lattice : Lattice, vqe_parameters : dict, SPSA_parameters : dict):
    if not isinstance(hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian for measurement must be a hamiltonian_class.Hamiltonian, you have entered a {type(hamiltonian)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(SPSA_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(SPSA_parameters)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    
    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    required_parameters = ['max_iters', 'average_length', 'grad_tol', 'average_tol', 'a', 'c', 'prints', 'diagnostics']
    for key in required_parameters:
        if key not in SPSA_parameters: raise KeyError(f"SPSA_parameters must contain the key = {key}")

    circuit_class = CircuitForLattice(lattice, n_fermion_layers = vqe_parameters['n_fermion_layers'])
    measurer_class = CircuitMeasurer(circuit_class, hamiltonian, vqe_parameters['simulator'], vqe_parameters['shots'], vqe_parameters['MEM'], vqe_parameters['SV'])

    guess = [np.random.uniform(0, 0.1) for _ in range(circuit_class.n_total_thetas)]

    def cost_function(thetas_values : list | np.ndarray):
        if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")

        measurer_class.bind_values(thetas_values)
        cost = measurer_class.expected_value_hamiltonian()
        # print(cost)
        return cost

    return natural_gradient_SPSA(cost_function, guess, SPSA_parameters)

def electric_field_values_noisy(thetas_values : list | np.ndarray, lattice : Lattice, vqe_parameters : dict):
    if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    hamiltonian = electric_n_direction(lattice, 0, 1)
    circuit_class = CircuitForLattice(lattice, n_fermion_layers = vqe_parameters['n_fermion_layers'])
    measurer_class = CircuitMeasurer(circuit_class, hamiltonian, vqe_parameters['simulator'], vqe_parameters['shots'], vqe_parameters['MEM'], vqe_parameters['SV'])

    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[site_n]:
            hamiltonian = electric_n_direction(lattice, site_n, direction)
            measurer_class.change_hamiltonian(hamiltonian)
            values[(lattice.labels[site_n], direction)] = measurer_class.expected_value_hamiltonian()

    return values

def electric_field_values_squared_noisy(thetas_values : list | np.ndarray, lattice : Lattice, vqe_parameters : dict):
    if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    hamiltonian = electric_n_direction(lattice, 0, 1)
    circuit_class = CircuitForLattice(lattice, n_fermion_layers = vqe_parameters['n_fermion_layers'])
    measurer_class = CircuitMeasurer(circuit_class, hamiltonian, vqe_parameters['simulator'], vqe_parameters['shots'], vqe_parameters['MEM'], vqe_parameters['SV'])
    
    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[site_n]:
            hamiltonian = electric_n_direction(lattice, site_n, direction)
            hamiltonian.multiply_by_hamiltonian(hamiltonian)
            measurer_class.change_hamiltonian(hamiltonian)
            values[(lattice.labels[site_n], direction)] = measurer_class.expected_value_hamiltonian()

    return values

def magnetic_field_values_noisy(thetas_values : list | np.ndarray, lattice : Lattice, vqe_parameters : dict):
    if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    hamiltonian = magnetic_term_n(lattice, 0)
    circuit_class = CircuitForLattice(lattice, n_fermion_layers = vqe_parameters['n_fermion_layers'])
    measurer_class = CircuitMeasurer(circuit_class, hamiltonian, vqe_parameters['simulator'], vqe_parameters['shots'], vqe_parameters['MEM'], vqe_parameters['SV'])
    
    values = {}

    for site_n in lattice.plaquettes:
        hamiltonian = magnetic_term_n(lattice, site_n)
        measurer_class.change_hamiltonian(hamiltonian)
        values[lattice.labels[site_n]] = measurer_class.expected_value_hamiltonian()

    return values
    
def charge_values_noisy(thetas_values : list | np.ndarray, lattice : Lattice, vqe_parameters : dict):
    if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    hamiltonian = charge_n_hamiltonian(lattice, 0)
    circuit_class = CircuitForLattice(lattice, n_fermion_layers = vqe_parameters['n_fermion_layers'])
    measurer_class = CircuitMeasurer(circuit_class, hamiltonian, vqe_parameters['simulator'], vqe_parameters['shots'], vqe_parameters['MEM'], vqe_parameters['SV'])

    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        hamiltonian = charge_n_hamiltonian(lattice, site_n)
        measurer_class.change_hamiltonian(hamiltonian)
        values[lattice.labels[site_n]] = measurer_class.expected_value_hamiltonian()
    
    return values

def particle_number_values_noisy(thetas_values : list | np.ndarray, lattice : Lattice, vqe_parameters : dict):
    if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    hamiltonian = charge_n_hamiltonian(lattice, 0)
    circuit_class = CircuitForLattice(lattice, n_fermion_layers = vqe_parameters['n_fermion_layers'])
    measurer_class = CircuitMeasurer(circuit_class, hamiltonian, vqe_parameters['simulator'], vqe_parameters['shots'], vqe_parameters['MEM'], vqe_parameters['SV'])

    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        hamiltonian = particle_n_hamiltonian(lattice, site_n)
        measurer_class.change_hamiltonian(hamiltonian)
        values[lattice.labels[site_n]] = measurer_class.expected_value_hamiltonian()
    
    return values

def observes_reduced_noisy(thetas_values : list | np.ndarray, lattice : Lattice, vqe_parameters : dict):
    if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    ef = electric_field_values_noisy(thetas_values, lattice, vqe_parameters)
    ef_sq = electric_field_values_squared_noisy(thetas_values, lattice, vqe_parameters)
    mf = magnetic_field_values_noisy(thetas_values, lattice, vqe_parameters)
    pn = particle_number_values_noisy(thetas_values, lattice, vqe_parameters)
    c = charge_values_noisy(thetas_values, lattice, vqe_parameters)

    total_pn = sum(pn.values())
    total_charge = sum(c.values())

    for dependant_variable in lattice.gauss_equations['dependant_variables']:
        equation = lattice.gauss_equations['solution'][dependant_variable]
        sub_ins = {}

        for var in lattice.gauss_equations['independant_variables']:
            if var in list(lattice.gauss_equations['reverse_link_variable_dict']):
                sub_ins[var] = ef[lattice.gauss_equations['reverse_link_variable_dict'][var]]
            else:
                site_n_charge = int(str(var)[1:])
                sub_ins[var] = c[lattice.labels[site_n_charge]]

        ef[lattice.gauss_equations['reverse_link_variable_dict'][dependant_variable]] = complex(equation.subs(sub_ins))

    site_n = 0
    gl = {}
    for equation in lattice.gauss_equations['equations']:
        sub_ins = {}

        for var in lattice.gauss_equations['independant_variables'] + lattice.gauss_equations['dependant_variables']:
            if var in list(lattice.gauss_equations['reverse_link_variable_dict']):
                sub_ins[var] = ef[lattice.gauss_equations['reverse_link_variable_dict'][var]]
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

def solve_and_observe_noisy(hamiltonian : Hamiltonian, lattice : Lattice, vqe_parameters : dict, SPSA_parameters : dict):
    if not isinstance(hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian for measurement must be a hamiltonian_class.Hamiltonian, you have entered a {type(hamiltonian)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(SPSA_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(SPSA_parameters)}")
    
    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    required_parameters = ['max_iters', 'average_length', 'grad_tol', 'average_tol', 'a', 'c', 'prints', 'diagnostics']
    for key in required_parameters:
        if key not in SPSA_parameters: raise KeyError(f"SPSA_parameters must contain the key = {key}")

    results = SPSA_vqe_solver_noisy(hamiltonian, lattice, vqe_parameters, SPSA_parameters)
    final_parameters = results['final_paras']

    return observes_reduced_noisy(final_parameters, hamiltonian, lattice, vqe_parameters)

