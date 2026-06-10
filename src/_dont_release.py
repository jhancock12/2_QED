# Standard libraries
from copy import copy

# Local modules
from observables import qed_hamiltonian
from circuits.circuit_class import CircuitForLattice
from circuits.measurer_class import CircuitMeasurer
from lattice_class import Lattice
from solvers.mixed import solve_noiseless_sample_many
from solvers.sparse import solve_and_observe_sparse
from global_helpers import smart_round, dict_print

# Third-party libraries
import qiskit_aer
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

def full_runner(lattice_parameters, qed_parameters, vqe_parameters, SPSA_parameters):
    # -------------------------------
    lattice = Lattice(lattice_parameters['L_x'], lattice_parameters['L_y'], 
                    lattice_parameters['n_g'], lattice_parameters['dynamical_links_list'], 
                    lattice_parameters['charge_site'], lattice_parameters['anticharge_site'], 
                    lattice_parameters['background_field'])
    hamiltonian = qed_hamiltonian(qed_parameters, lattice, mass_multi = 1, electric_multi = 1, magnetic_multi = 1, kinetic_multi = 1)

    hamiltonian_electric = qed_hamiltonian(qed_parameters, lattice, mass_multi = 0, electric_multi = 1, magnetic_multi = 0, kinetic_multi = 0)
    hamiltonian_magnetic = qed_hamiltonian(qed_parameters, lattice, mass_multi = 1, electric_multi = 0, magnetic_multi = 0, kinetic_multi = 0)
    hamiltonian_mass = qed_hamiltonian(qed_parameters, lattice, mass_multi = 0, electric_multi = 0, magnetic_multi = 1, kinetic_multi = 0)
    hamiltonian_kinetic = qed_hamiltonian(qed_parameters, lattice, mass_multi = 0, electric_multi = 0, magnetic_multi = 0, kinetic_multi = 1)

    hamiltonians = [hamiltonian, hamiltonian_mass, hamiltonian_electric, hamiltonian_magnetic, hamiltonian_kinetic]
    hamiltonian_names = ["full", "mass", "electric", "magnetic", "kinetic"]

    circuit_class = CircuitForLattice(lattice, vqe_parameters['n_fermion_layers'])
    # -------------------------------
    fake_backend = FakeCairoV2()
    noisy_simulator = qiskit_aer.AerSimulator.from_backend(fake_backend)
    # -------------------------------
    accepted_strings = lattice.SV_strings()

    measurer_noiseless = CircuitMeasurer(circuit_class, hamiltonian, qiskit_aer.AerSimulator(), vqe_parameters['shots'])

    measurer_noisy = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])

    measurer_noisy_SV_only = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])
    measurer_noisy_SV_only.add_SV(accepted_strings)

    measurer_noisy_mem_only = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])
    measurer_noisy_mem_only.build_MEM()
    measurer_noisy_all = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])
    measurer_noisy_all.add_SV(accepted_strings)
    measurer_noisy_all.build_MEM()
    # -------------------------------
    measurer_classes = [measurer_noiseless, measurer_noisy, measurer_noisy_mem_only, measurer_noisy_SV_only, measurer_noisy_all]
    names = ['No quantum noise', 'No QEM', 'MEM only', 'SV only', 'Both']

    # measurer_classes = [measurer_noiseless, measurer_noisy, measurer_noisy_mem_only]
    # names = ['No quantum noise', 'No QEM', 'MEM only']
    # -------------------------------  
    qed_parameters['charge_weight'] = 0.0
    hamiltonian = qed_hamiltonian(qed_parameters, lattice)
    results, final_thetas = solve_noiseless_sample_many(hamiltonian, lattice, vqe_parameters, SPSA_parameters, measurer_classes, names)
    qed_parameters['charge_weight'] = 100.0
    hamiltonian = qed_hamiltonian(qed_parameters, lattice)
    results_sparse = solve_and_observe_sparse(hamiltonian, lattice)
    results['sparse'] = results_sparse
    qed_parameters['charge_weight'] = 0.0
    for key in results:
        print(f"----- {key} -----")
        dict_print(results[key])