# Standard libraries

# Local modules
from hamiltonian_class import Hamiltonian
from lattice_class import Lattice
from operators import mass_term_n, electric_solve_gauss, magnetic_term_n, kinetic_term_n, charge_total_hamiltonian

# Third-party libraries
   
def mass_hamiltonian(parameters : dict, lattice : Lattice, mass_multi = 1):
    if not isinstance(parameters, dict): raise TypeError(f"parameters must be a dictionary, you have entered a {type(parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if 'm' not in parameters: raise KeyError(f"parameters must contain the key = m")
    if not isinstance(parameters['m'], int | float | complex): raise TypeError(f"m must be a number, you have entered a {type(parameters['m'])}")
    if not isinstance(mass_multi, int | float | complex): raise TypeError(f"mass_multi must be a number, you have entered a {type(mass_multi)}")

    mass_coeff = parameters['m'] * mass_multi

    mass_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    for site_n in range(lattice.n_fermion_qubits):
        mass_hamiltonian = mass_term_n(lattice, site_n)
        mass_hamiltonian_total.add_hamiltonian(mass_hamiltonian)

    mass_hamiltonian_total.multiply_by_constant(mass_coeff)
    mass_hamiltonian_total.cleanup()

    return mass_hamiltonian_total

def electric_hamiltonian(parameters : dict, lattice : Lattice, electric_multi = 1):
    if not isinstance(parameters, dict): raise TypeError(f"parameters must be a dictionary, you have entered a {type(parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if 'g' not in parameters: raise KeyError(f"parameters must contain the key = g")
    if not isinstance(parameters['g'], int | float | complex): raise TypeError(f"g must be a number, you have entered a {type(parameters['g'])}")
    if not isinstance(electric_multi, int | float | complex): raise TypeError(f"electric_multi must be a number, you have entered a {type(electric_multi)}")

    electric_coeff = parameters['g']*parameters['g'] / 2
    electric_coeff *= electric_multi

    electric_hamiltonian_total = electric_solve_gauss(lattice)
    electric_hamiltonian_total.multiply_by_constant(electric_coeff)
    electric_hamiltonian_total.cleanup()

    return electric_hamiltonian_total

def magnetic_hamiltonian(parameters : dict, lattice : Lattice, magnetic_multi = 1):
    if not isinstance(parameters, dict): raise TypeError(f"parameters must be a dictionary, you have entered a {type(parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if 'g' not in parameters: raise KeyError(f"parameters must contain the key = g")
    if 'a' not in parameters: raise KeyError(f"parameters must contain the key = a")
    if not isinstance(parameters['g'], int | float | complex): raise TypeError(f"g must be a number, you have entered a {type(parameters['g'])}")
    if not isinstance(parameters['a'], int | float | complex): raise TypeError(f"a must be a number, you have entered a {type(parameters['a'])}")
    if not isinstance(magnetic_multi, int | float | complex): raise TypeError(f"magnetic_multi must be a number, you have entered a {type(magnetic_multi)}")

    if parameters['g'] == 0: raise ValueError(f"g must be non-zero, you entered g = {parameters['g']}")
    if parameters['a'] == 0: raise ValueError(f"a must be non-zero, you entered a = {parameters['a']}")

    magnetic_coeff = -1/(2*(parameters['a']*parameters['a'])*(parameters['g']*parameters['g']))
    magnetic_coeff *= magnetic_multi

    magnetic_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    for site_n in lattice.plaquettes:
        magnetic_hamiltonian = magnetic_term_n(lattice, site_n)
        magnetic_hamiltonian_total.add_hamiltonian(magnetic_hamiltonian)

    magnetic_hamiltonian_total.multiply_by_constant(magnetic_coeff)
    magnetic_hamiltonian_total.cleanup()

    return magnetic_hamiltonian_total

def kinetic_hamiltonian(parameters : dict, lattice : Lattice, kinetic_multi = 1):
    if not isinstance(parameters, dict): raise TypeError(f"parameters must be a dictionary, you have entered a {type(parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if 'a' not in parameters: raise KeyError(f"parameters must contain the key = a")
    if not isinstance(parameters['a'], int | float | complex): raise TypeError(f"a must be a number, you have entered a {type(parameters['a'])}")
    if not isinstance(kinetic_multi, int | float | complex): raise TypeError(f"kinetic_multi must be a number, you have entered a {type(kinetic_multi)}")

    if parameters['a'] == 0: raise ValueError(f"a must be non-zero, you entered a = {parameters['a']}")

    kinetic_coeff = 1/(2*parameters['a'])
    kinetic_coeff *= kinetic_multi

    kinetic_hamiltonian_total = Hamiltonian(lattice.n_qubits)
    for site_n in range(lattice.n_fermion_qubits):
        kinetic_hamiltonian = kinetic_term_n(lattice, site_n)
        kinetic_hamiltonian_total.add_hamiltonian(kinetic_hamiltonian)

    kinetic_hamiltonian_total.multiply_by_constant(kinetic_coeff)
    kinetic_hamiltonian_total.cleanup()

    return kinetic_hamiltonian_total

def charge_hamiltonian(parameters : dict, lattice : Lattice):
    if not isinstance(parameters, dict): raise TypeError(f"parameters must be a dictionary, you have entered a {type(parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if 'charge_weight' not in parameters: raise KeyError(f"parameters must contain the key = charge_weight")
    if not isinstance(parameters['charge_weight'], int | float | complex): raise TypeError(f"charge_weight must be a number, you have entered a {type(parameters['charge_weight'])}")

    charge_hamiltonian_total = charge_total_hamiltonian(lattice)
    charge_hamiltonian_total.multiply_by_hamiltonian(charge_hamiltonian_total)
    charge_hamiltonian_total.multiply_by_constant(parameters['charge_weight'])
    charge_hamiltonian_total.cleanup()

    return charge_hamiltonian_total

def qed_hamiltonian(parameters : dict, lattice : Lattice, mass_multi = 1, electric_multi = 1, magnetic_multi = 1, kinetic_multi = 1):
    if not isinstance(parameters, dict): raise TypeError(f"parameters must be a dictionary, you have entered a {type(parameters)}")
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")

    required_parameters = ['m', 'g', 'a', 'charge_weight']
    for key in required_parameters:
        if key not in parameters: raise KeyError(f"parameters must contain the key = {key}")

    full_hamiltonian = Hamiltonian(lattice.n_qubits)

    mass_hamiltonian_total = mass_hamiltonian(parameters, lattice, mass_multi)
    electric_hamiltonian_total = electric_hamiltonian(parameters, lattice, electric_multi)
    magnetic_hamiltonian_total = magnetic_hamiltonian(parameters, lattice, magnetic_multi)
    kinetic_hamiltonian_total = kinetic_hamiltonian(parameters, lattice, kinetic_multi)
    charge_hamiltonian_total = charge_hamiltonian(parameters, lattice)

    full_hamiltonian.add_hamiltonian(mass_hamiltonian_total)
    full_hamiltonian.add_hamiltonian(electric_hamiltonian_total)
    full_hamiltonian.add_hamiltonian(magnetic_hamiltonian_total)
    full_hamiltonian.add_hamiltonian(kinetic_hamiltonian_total)
    full_hamiltonian.add_hamiltonian(charge_hamiltonian_total)

    full_hamiltonian.cleanup()
    
    return full_hamiltonian