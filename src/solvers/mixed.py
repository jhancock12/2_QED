# Standard libraries
from copy import copy

# Local modules
from circuits.measurer_class import CircuitMeasurer
from circuits.circuit_class import CircuitForLattice
from hamiltonian_class import Hamiltonian
from lattice_class import Lattice
from solvers.noiseless import SPSA_vqe_solver_noiseless
from operators import electric_n_direction, magnetic_term_n, particle_n_hamiltonian, charge_n_hamiltonian
from global_helpers import smart_round

# Third-party libraries
import numpy as np

def electric_field_values_noisy_local(lattice : Lattice, measurer_class : CircuitMeasurer, ZNE : bool = False):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(measurer_class, CircuitMeasurer): raise TypeError(f"The measurer must be a measurer_class.CircuitMeasurer, you have entered a {type(measurer_class)}")
    if not isinstance(ZNE, bool): raise TypeError(f"ZNE is a boolean check, you have entered a {type(ZNE)}")

    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[site_n]:
            hamiltonian = electric_n_direction(lattice, site_n, direction)
            measurer_class.change_hamiltonian(hamiltonian)
            if ZNE:
                values[(lattice.labels[site_n], direction)] = measurer_class.ZNE_expected_value_hamiltonian(3)
            else:
                values[(lattice.labels[site_n], direction)] = measurer_class.expected_value_hamiltonian()

    return values

def electric_field_values_squared_noisy_local(lattice : Lattice, measurer_class : CircuitMeasurer, ZNE : bool = False):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(measurer_class, CircuitMeasurer): raise TypeError(f"The measurer must be a measurer_class.CircuitMeasurer, you have entered a {type(measurer_class)}")
    if not isinstance(ZNE, bool): raise TypeError(f"ZNE is a boolean check, you have entered a {type(ZNE)}")
    
    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        for direction in lattice.directions[site_n]:
            hamiltonian = electric_n_direction(lattice, site_n, direction)
            hamiltonian.multiply_by_hamiltonian(hamiltonian)
            measurer_class.change_hamiltonian(hamiltonian)
            if ZNE:
                values[(lattice.labels[site_n], direction)] = measurer_class.ZNE_expected_value_hamiltonian(3)
            else:
                values[(lattice.labels[site_n], direction)] = measurer_class.expected_value_hamiltonian()

    return values

def magnetic_field_values_noisy_local(lattice : Lattice, measurer_class : CircuitMeasurer, ZNE : bool = False):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(measurer_class, CircuitMeasurer): raise TypeError(f"The measurer must be a measurer_class.CircuitMeasurer, you have entered a {type(measurer_class)}")
    if not isinstance(ZNE, bool): raise TypeError(f"ZNE is a boolean check, you have entered a {type(ZNE)}")
    
    values = {}

    for site_n in lattice.plaquettes:
        hamiltonian = magnetic_term_n(lattice, site_n)
        measurer_class.change_hamiltonian(hamiltonian)
        if ZNE:
            values[lattice.labels[site_n]] = measurer_class.ZNE_expected_value_hamiltonian(3)
        else:
            values[lattice.labels[site_n]] = measurer_class.expected_value_hamiltonian()

    return values
    
def charge_values_noisy_local(lattice : Lattice, measurer_class : CircuitMeasurer, ZNE : bool = False):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(measurer_class, CircuitMeasurer): raise TypeError(f"The measurer must be a measurer_class.CircuitMeasurer, you have entered a {type(measurer_class)}")
    if not isinstance(ZNE, bool): raise TypeError(f"ZNE is a boolean check, you have entered a {type(ZNE)}")

    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        hamiltonian = charge_n_hamiltonian(lattice, site_n)
        measurer_class.change_hamiltonian(hamiltonian)
        if ZNE:
            values[lattice.labels[site_n]] = measurer_class.ZNE_expected_value_hamiltonian(3)
        else:
            values[lattice.labels[site_n]] = measurer_class.expected_value_hamiltonian()
    
    return values

def particle_number_values_noisy_local(lattice : Lattice, measurer_class : CircuitMeasurer, ZNE : bool = False):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(measurer_class, CircuitMeasurer): raise TypeError(f"The measurer must be a measurer_class.CircuitMeasurer, you have entered a {type(measurer_class)}")
    if not isinstance(ZNE, bool): raise TypeError(f"ZNE is a boolean check, you have entered a {type(ZNE)}")

    values = {}

    for site_n in range(lattice.n_fermion_qubits):
        hamiltonian = particle_n_hamiltonian(lattice, site_n)
        measurer_class.change_hamiltonian(hamiltonian)
        if ZNE:
            values[lattice.labels[site_n]] = measurer_class.ZNE_expected_value_hamiltonian(3)
        else:
            values[lattice.labels[site_n]] = measurer_class.expected_value_hamiltonian()
    
    return values

def observes_reduced_noisy_local(thetas_values : list | np.ndarray, lattice : Lattice, measurer_class : CircuitMeasurer, ZNE : bool = False):
    if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(measurer_class, CircuitMeasurer): raise TypeError(f"The measurer must be a measurer_class.CircuitMeasurer, you have entered a {type(measurer_class)}")
    if not isinstance(ZNE, bool): raise TypeError(f"ZNE is a boolean check, you have entered a {type(ZNE)}")

    measurer_class.bind_values(thetas_values)

    ef = electric_field_values_noisy_local(lattice, measurer_class, ZNE)
    ef_sq = electric_field_values_squared_noisy_local(lattice, measurer_class, ZNE)
    mf = magnetic_field_values_noisy_local(lattice, measurer_class, ZNE)
    pn = particle_number_values_noisy_local(lattice, measurer_class, ZNE)
    c = charge_values_noisy_local(lattice, measurer_class, ZNE)

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

def solve_noiseless_sample_many(hamiltonian : Hamiltonian, lattice : Lattice, vqe_parameters : dict, SPSA_parameters : dict, measurer_classes : list[CircuitMeasurer], names : list[str]):
    if not isinstance(hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian for measurement must be a hamiltonian_class.Hamiltonian, you have entered a {type(hamiltonian)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(vqe_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(vqe_parameters)}")
    if not isinstance(SPSA_parameters, dict): raise TypeError(f"SPSA parameters must be a dictionary, you have entered a {type(SPSA_parameters)}")
    if not isinstance(measurer_classes, list): raise TypeError(f"A list of circuit_class.CircuitMeasurer's must be inputted, you have entered a {type(measurer_classes)}")
    if not isinstance(names, list): raise TypeError(f"A list of measurer names must be inputted, you have entered a {type(names)}")
    
    required_parameters = ['n_fermion_layers', 'shots', 'simulator', 'MEM', 'SV']
    for key in required_parameters:
        if key not in vqe_parameters: raise KeyError(f"vqe_parameters must contain the key = {key}")

    required_parameters = ['max_iters', 'average_length', 'grad_tol', 'average_tol', 'a', 'c', 'prints', 'diagnostics']
    for key in required_parameters:
        if key not in SPSA_parameters: raise KeyError(f"SPSA_parameters must contain the key = {key}")

    for measurer_class_k in range(len(measurer_classes)):
        if not isinstance(measurer_classes[measurer_class_k], CircuitMeasurer): raise TypeError(f"All entries must be a circuit_class.CircuitMeasurer in the, you have entered a {type(measurer_classes[measurer_class_k])} as the {measurer_class_k}-th entry")

    for name_k in range(len(names)):
        if not isinstance(names[name_k], str): raise TypeError(f"All entries must be a string in names, you have entered a {type(names[name_k])} as the {name_k}-th entry")

    if not len(measurer_classes) == len(names): raise ValueError(f"The length of the names ({len(names)}) must be the same as the length of the measurers ({len(measurer_classes)})")

    results = SPSA_vqe_solver_noiseless(hamiltonian, lattice, vqe_parameters, SPSA_parameters)

    full_results = {}

    local_names = copy(names)

    local_names.append(copy(local_names[-1] + " + ZNE"))

    counter = 0
    for measurer_class in measurer_classes:
        print(f"Running {local_names[counter]}")
        observe_results = observes_reduced_noisy_local(results['final_paras'], lattice, measurer_class)
        full_results[local_names[counter]] = observe_results
        measurer_class.change_hamiltonian(hamiltonian)
        full_results[local_names[counter]]['energy'] = measurer_class.expected_value_hamiltonian()
        counter += 1
    print("Running ZNE")
    full_results[local_names[-1]] = observes_reduced_noisy_local(results['final_paras'], lattice, measurer_class, ZNE = True)
    measurer_class.change_hamiltonian(hamiltonian)
    full_results[local_names[-1]]['energy'] = measurer_class.ZNE_expected_value_hamiltonian(3)
    return full_results