import sqlite3
import random
from pymatgen.core import Composition

def main():
    conn = sqlite3.connect("results/scale_test.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT formula FROM work_queue")
    all_formulas = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Total raw candidates loaded: {len(all_formulas)}")
    
    sample_size = min(10000, len(all_formulas))
    sample = random.sample(all_formulas, sample_size)
    
    valid_count = 0
    noise_count = 0
    
    for formula in sample:
        try:
            comp = Composition(formula)
            # Check if any valid oxidation state combination exists that sums to 0
            if comp.oxi_state_guesses():
                valid_count += 1
            else:
                noise_count += 1
        except Exception:
            noise_count += 1
            
    print(f"\n--- Sanity Check Results on Sample of {sample_size} ---")
    print(f"Passed charge balance / oxidation states: {valid_count} ({(valid_count/sample_size)*100:.2f}%)")
    print(f"Failed (combinatorial noise): {noise_count} ({(noise_count/sample_size)*100:.2f}%)")
    
    projected_total = int(len(all_formulas) * (valid_count / sample_size))
    print(f"\nProjected usable candidates out of {len(all_formulas)}: ~{projected_total}")

if __name__ == "__main__":
    main()
