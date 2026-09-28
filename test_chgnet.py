import time
from pymatgen.core import Structure, Lattice
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import StructOptimizer

# Define some basic structures (approximate starting guesses)
structures = {
    'NaCl': Structure(Lattice.cubic(5.64), ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl'], 
                      [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0],
                       [0.5,0.5,0.5], [0.5,0,0], [0,0.5,0], [0,0,0.5]]),
    'Si': Structure(Lattice.cubic(5.43), ['Si']*8, 
                    [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0],
                     [0.25,0.25,0.25], [0.25,0.75,0.75], [0.75,0.25,0.75], [0.75,0.75,0.25]]),
    'GaAs': Structure(Lattice.cubic(5.65), ['Ga']*4 + ['As']*4, 
                      [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0],
                       [0.25,0.25,0.25], [0.25,0.75,0.75], [0.75,0.25,0.75], [0.75,0.75,0.25]]),
    'Fe': Structure(Lattice.cubic(2.866), ['Fe', 'Fe'], [[0,0,0], [0.5,0.5,0.5]]),
    'TiO2': Structure(Lattice.tetragonal(4.59, 2.96), ['Ti', 'Ti', 'O', 'O', 'O', 'O'], 
                      [[0,0,0], [0.5,0.5,0.5], 
                       [0.3,0.3,0], [0.7,0.7,0], [0.2,0.8,0.5], [0.8,0.2,0.5]])
}

print("Loading CHGNet...")
chgnet = CHGNet.load()
optimizer = StructOptimizer()

print(f"{'Material':<10} | {'Time (s)':<10} | {'E_relaxed/atom (eV)':<20}")
print("-" * 45)

for name, struct in structures.items():
    t0 = time.time()
    result = optimizer.relax(struct, verbose=False)
    t1 = time.time()
    
    e_total = result['trajectory'].energies[-1]
    e_per_atom = e_total / len(struct)
    print(f"{name:<10} | {t1-t0:<10.3f} | {e_per_atom:<20.6f}")

