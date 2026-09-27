import os
from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    docs1 = mpr.materials.summary.search(elements=["Ba", "K", "Fe", "As"], fields=["formula_pretty", "is_stable"])
    print("All Ba-K-Fe-As formulas in MP:", [(d.formula_pretty, d.is_stable) for d in docs1])
    
    docs2 = mpr.materials.summary.search(elements=["Tm", "Os", "Sn"], fields=["formula_pretty", "is_stable"])
    print("All Tm-Os-Sn formulas in MP:", [(d.formula_pretty, d.is_stable) for d in docs2])
    
    docs3 = mpr.materials.summary.search(formula="Tm4Os6Sn19", fields=["formula_pretty", "is_stable"])
    print("Exact Tm4Os6Sn19:", [(d.formula_pretty, d.is_stable) for d in docs3])
