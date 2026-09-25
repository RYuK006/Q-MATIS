import sys
import json
import torch
import os
import importlib.util

spec = importlib.util.spec_from_file_location("mod", "experiments/exp3_thermal/02_train.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["mod"] = mod
spec.loader.exec_module(mod)
MatDataset = mod.MatDataset

def check_cache_identical():
    print('--- Check 1: Cache Identical ---')
    with open('data/train_dataset_42.json', 'r') as f:
        data = json.load(f)[:20] # Just use 20 for speed
        
    print('Generating uncached...')
    dataset_uncached = MatDataset(data, run_type='A_prime', cache_name=None)
    
    print('Generating cached (will save)...')
    if os.path.exists('data/cache_test_verify.pt'):
        os.remove('data/cache_test_verify.pt')
    dataset_cached_write = MatDataset(data, run_type='A_prime', cache_name='test_verify')
    
    print('Loading cached (will load)...')
    dataset_cached_read = MatDataset(data, run_type='A_prime', cache_name='test_verify')
    
    match_all = True
    for i in range(len(dataset_uncached)):
        g1 = dataset_uncached[i]
        g2 = dataset_cached_read[i]
        
        match_z = torch.equal(g1.z, g2.z)
        match_pos = torch.equal(g1.pos, g2.pos)
        match_edge = torch.equal(g1.edge_index, g2.edge_index)
        match_y = torch.equal(g1.y, g2.y)
        if not (match_z and match_pos and match_edge and match_y):
            match_all = False
            break
            
    print(f'Match All: {match_all}')
    if match_all:
        print('Identical graphs confirmed.\n')
    else:
        print('MISMATCH DETECTED!\n')

if __name__ == "__main__":
    check_cache_identical()
