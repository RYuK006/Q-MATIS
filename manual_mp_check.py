import os
from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    # Check 1: Ba0.4K0.6Fe2As2 (or Ba-K-Fe-As system)
    print("Checking Ba-K-Fe-As system...")
    docs1 = mpr.summary.search(elements=["Ba", "K", "Fe", "As"], is_stable=True, fields=["formula_pretty"])
    print("Stable Ba-K-Fe-As formulas in MP:", [d.formula_pretty for d in docs1])
    
    # Check 2: Tm4Os6Sn19
    print("\nChecking Tm-Os-Sn system...")
    docs2 = mpr.summary.search(elements=["Tm", "Os", "Sn"], is_stable=True, fields=["formula_pretty"])
    print("Stable Tm-Os-Sn formulas in MP:", [d.formula_pretty for d in docs2])
