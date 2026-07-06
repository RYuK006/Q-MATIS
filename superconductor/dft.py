import os
import logging
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints, Potcar

logger = logging.getLogger(__name__)

def generate_slurm_script(out_dir: str, job_name: str = "vasp_supercon"):
    """
    Generates a standard SLURM submission script for VASP.
    """
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=standard

module load vasp/6.4.1

srun vasp_std > vasp.out
"""
    with open(os.path.join(out_dir, "run_vasp.sh"), "w") as f:
        f.write(slurm_script)

def export_vasp(structure: Structure, out_dir: str, tc: float):
    """
    Generate a full VASP export package for a candidate.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # POSCAR
    poscar = Poscar(structure)
    poscar.write_file(os.path.join(out_dir, "POSCAR"))
    
    # INCAR (Template for accurate structural relaxation and electron-phonon precursor)
    incar_params = {
        "SYSTEM": f"High-Tc Candidate (Predicted Tc: {tc:.2f}K)",
        "PREC": "Accurate",
        "ALGO": "Fast",
        "IBRION": 2,        # Conjugate gradient
        "ISIF": 3,          # Relax ions, cell shape, and volume
        "NSW": 100,         # Number of ionic steps
        "EDIFF": 1e-6,      # Electronic convergence
        "EDIFFG": -0.01,    # Ionic convergence (forces < 0.01 eV/A)
        "ISMEAR": 1,        # Methfessel-Paxton for metals
        "SIGMA": 0.1,
        "LREAL": "Auto",
        "ENMAX": 500,       # Sensible default, should check POTCAR ENMAX
        "LORBIT": 11,       # DOSCAR and PROCAR
    }
    incar = Incar(incar_params)
    incar.write_file(os.path.join(out_dir, "INCAR"))
    
    # KPOINTS
    # High density k-points required for metals/superconductors
    kpoints = Kpoints.automatic_density(structure, kppa=5000)
    kpoints.write_file(os.path.join(out_dir, "KPOINTS"))
    
    # POTCAR
    elements = [str(el) for el in structure.composition.elements]
    try:
        potcar = Potcar(elements)
        potcar.write_file(os.path.join(out_dir, "POTCAR"))
        logger.info(f"Successfully generated POTCAR for {elements}")
    except Exception as e:
        logger.warning(f"Could not generate POTCAR automatically: {e}")
        with open(os.path.join(out_dir, "README_POTCAR.txt"), "w") as f:
            f.write("Please generate a POTCAR concatenating the following PAW potentials:\n")
            f.write(" ".join(elements) + "\n")
            f.write("\nSet PMG_VASP_PSP_DIR in your environment to automate this via pymatgen.\n")
            
    # SLURM script
    generate_slurm_script(out_dir)
        
    logger.info(f"Exported VASP package to {out_dir}")

def export_qe(structure: Structure, out_dir: str, tc: float):
    """
    Generate a Quantum ESPRESSO pw.x input template.
    """
    os.makedirs(out_dir, exist_ok=True)
    from pymatgen.io.pwscf import PWInput
    
    # Simplified pseudo mapping
    pseudo_map = {str(el): f"{el}.UPF" for el in structure.composition.elements}
    
    control = {"calculation": "vc-relax", "pseudo_dir": "./"}
    system = {"ecutwfc": 60, "ecutrho": 240}
    electrons = {"conv_thr": 1e-8}
    
    pw_in = PWInput(structure, pseudo=pseudo_map, control=control, system=system, electrons=electrons)
    pw_in.write_file(os.path.join(out_dir, "pw.in"))
    
    logger.info(f"Exported Quantum ESPRESSO package to {out_dir}")
