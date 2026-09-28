import random
import itertools
import time

def read_miedema_db(filepath):
    db = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().replace(' ', '').split(',')
            if len(parts) >= 6:
                name = parts[0]
                db[name] = {
                    'phi': float(parts[1]),
                    'nws13': float(parts[2]),
                    'Vm23': float(parts[3]),
                    'RP': float(parts[4]),
                    'TRAN': parts[5]
                }
    return db

def calc_binary_hmix(db, A, B, xA):
    if A not in db or B not in db:
        return 0.0 # Ignore if missing
        
    eA = db[A]
    eB = db[B]
    
    # Calculate RP
    if eA['TRAN'] == 'T' and eB['TRAN'] == 'T':
        RP = 0.0
        P = 0.147
    elif eA['TRAN'] == 'N' and eB['TRAN'] == 'N':
        RP = 0.0
        P = 0.111
    else:
        RP = eA['RP'] * eB['RP'] * 0.73
        P = 0.128
        
    aA = 0.14 if A in ['Li','Na','K','Rb','Sc','Fr'] else 0.0
    aB = 0.14 if B in ['Li','Na','K','Rb','Sc','Fr'] else 0.0
    
    dePhi = eA['phi'] - eB['phi']
    deNws13 = eA['nws13'] - eB['nws13']
    
    xB = 1.0 - xA
    
    A_Vm23Alloy = eA['Vm23'] * (1 + aA * xB * dePhi)
    B_Vm23Alloy = eB['Vm23'] * (1 + aB * xA * (-dePhi))
    
    xAs = (xA * A_Vm23Alloy) / (xA * A_Vm23Alloy + xB * B_Vm23Alloy)
    fxs = xAs * (1.0 - xAs)
    
    g = 2.0 * (xA * A_Vm23Alloy + xB * B_Vm23Alloy) / (1.0/eA['nws13'] + 1.0/eB['nws13'])
    
    Avogardro = 6.02e23
    QP = 9.4
    e = 1.0
    
    deHmix = Avogardro * fxs * g * P * (-e*(dePhi)**2 + QP*(deNws13)**2 - RP) * 1.60217657E-22
    return deHmix

def calc_quaternary_hmix(db, elements, fractions):
    # Using the standard Toop or Muggianu extension: H = sum_i sum_j (x_i * x_j * 4 * H_binary(0.5))
    hmix_total = 0.0
    for i in range(len(elements)):
        for j in range(i+1, len(elements)):
            # get equiatomic binary mix heat (per mole of atoms)
            h_bin = calc_binary_hmix(db, elements[i], elements[j], 0.5)
            # contribution to total
            hmix_total += 4.0 * fractions[i] * fractions[j] * h_bin
    return hmix_total

def main():
    db = read_miedema_db('miedema_calc/database.dat')
    
    # Filter out radioactive/actinide and highly impractical/toxic elements
    excluded = {'U', 'Pu', 'Th', 'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Tl', 'Hg', 'Cd', 'Os', 'Ir'}
    metals = [m for m in db.keys() if m not in excluded]
    
    print(f'Generating 1,000 random quaternary metallic alloys using {len(metals)} filtered metals...')
    random.seed(42)
    sample = []
    metal_combos = list(itertools.combinations(metals, 4))
    
    while len(sample) < 1000:
        combo = random.choice(metal_combos)
        splits = sorted(random.sample(range(1, 20), 3))
        x = splits[0] * 0.05
        y = (splits[1] - splits[0]) * 0.05
        z = (splits[2] - splits[1]) * 0.05
        w = (20 - splits[2]) * 0.05
        sample.append((combo, (x, y, z, w)))
        
    print('Calculating Miedema Heat of Mixing...')
    passed = []
    failed = []
    
    for combo, fracs in sample:
        hmix = calc_quaternary_hmix(db, combo, fracs)
        formula = f"{combo[0]}{fracs[0]:.2f}{combo[1]}{fracs[1]:.2f}{combo[2]}{fracs[2]:.2f}{combo[3]}{fracs[3]:.2f}"
        if hmix < 0: # Negative heat of mixing = thermodynamically favorable to mix
            passed.append((formula, hmix))
        else:
            failed.append((formula, hmix))
            
    print(f'\nTotal Evaluated: {len(sample)}')
    print(f'Passed (H_mix < 0): {len(passed)} ({(len(passed)/1000)*100:.2f}%)')
    
    print('\n3 Examples of Passed (Stable) Alloys:')
    for f, h in passed[:3]:
        print(f"  {f}  ->  dH_mix = {h:.2f} kJ/mol")
        
    print('\n3 Examples of Failed (Phase Separating) Alloys:')
    for f, h in failed[:3]:
        print(f"  {f}  ->  dH_mix = {h:.2f} kJ/mol")

if __name__ == '__main__':
    main()
