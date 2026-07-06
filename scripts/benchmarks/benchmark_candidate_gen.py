import yaml
import time
import logging
import subprocess
from pymatgen.core import Structure, Lattice
from superconductor.candidate_gen import PhysicsAwareCandidateEngine
from superconductor.materials_lake import MaterialsLake
from superconductor.material_registry import MaterialRegistry
from superconductor.experiment_registry import ExperimentRegistry
from superconductor.explorer_dashboard import DashboardGenerator

logging.basicConfig(level=logging.INFO)

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    lake = MaterialsLake(config.get("candidate_generation", {}).get("database_path", "data/qmatis_lake.db"))
    registry = MaterialRegistry(lake)
    exp_registry = ExperimentRegistry(lake)
    
    # 1. Register Experiment
    exp = exp_registry.register_experiment(config, pipeline_version="v2_event_sourcing")
    exp.git_commit = get_git_commit()
    
    # 2. Register Mock Model & Checkpoint
    mod = exp_registry.register_model("MockAlignn", "ALIGNN", "v2", "qmatis_base")
    chk = exp_registry.register_checkpoint(mod.id, epoch=100, val_loss=0.015, weights_path="models/mock.pt")
    
    engine = PhysicsAwareCandidateEngine(config, registry=registry, experiment_id=exp.id, checkpoint_id=chk.id)
    
    # Create a baseline perovskite structure (SrTiO3)
    lattice = Lattice.cubic(3.905)
    structure = Structure(
        lattice,
        ["Sr", "Ti", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5]
        ]
    )
    
    substitutions = [
        {"Sr": "Ba"}, # BaTiO3 (Valid)
        {"Sr": "Ca"}, # CaTiO3 (Valid)
        {"Ti": "Zr"}, # SrZrO3 (Valid)
        {"Sr": "K", "Ti": "Nb"}, # KNbO3 (Valid charge)
        {"Sr": "Li"}, # LiTiO3 (Charge imbalance, size mismatch)
        {"Ti": "F"}, # SrFO3 (Charge imbalance, electronegativity mismatch)
        {"O": "S"} # SrTiS3 (Maybe size mismatch)
    ]
    
    print(f"Running Candidate Generation Benchmark under EXP ID: {exp.id}")
    start_time = time.time()
    
    results = engine.generate(structure, "substitution", substitutions=substitutions)
    
    end_time = time.time()
    exp_registry.end_experiment(exp.id, "COMPLETED")
    
    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
    print(f"Generated {len(results)} valid candidates out of {len(substitutions)} attempts.")
    
    DashboardGenerator.generate_html_summary(lake, "data/qmatis_explorer.html")
    print("Dashboard generated at data/qmatis_explorer.html")

if __name__ == "__main__":
    main()
