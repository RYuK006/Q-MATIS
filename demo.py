from download_model import download_if_missing
import argparse
import numpy as np
import joblib
from pymatgen.core import Composition
from superconductor.features import get_node_features
import warnings
warnings.filterwarnings("ignore")

def get_composition_features(formula):
    try:
        comp = Composition(formula).fractional_composition
        features = []
        weights = []
        for el, amt in comp.items():
            class MockSite:
                def __init__(self, e):
                    self.specie = e
            features.append(get_node_features(MockSite(el)))
            weights.append(amt)
        features = np.array(features)
        weights = np.array(weights).reshape(-1, 1)
        mean_feat = np.sum(features * weights, axis=0)
        return mean_feat
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Predict Tc with Uncertainty for any Composition.")
    parser.add_argument("formula", type=str, help="Chemical formula (e.g., Ba0.4K0.6Fe2As2)")
    args = parser.parse_args()

    feat = get_composition_features(args.formula)
    if feat is None:
        print(f"Error: Could not parse formula '{args.formula}'")
        return

    try:
        rf = joblib.load(download_if_missing())
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Get predictions from individual trees in the random forest
    tree_preds = []
    for tree in rf.estimators_:
        pred = tree.predict(feat.reshape(1, -1))[0]
        tree_preds.append(pred)
        
    tree_preds = np.array(tree_preds)
    mean_tc = np.mean(tree_preds)
    std_tc = np.std(tree_preds)
    
    # 95% Confidence Interval
    z = 1.96
    lower_bound = max(0.0, mean_tc - z * std_tc)  # Tc can't be negative
    upper_bound = mean_tc + z * std_tc
    
    print("\n" + "="*50)
    print(f"Q-MATIS Tc Prediction Demo")
    print("="*50)
    print(f"Formula:          {args.formula}")
    print(f"Predicted Tc:     {mean_tc:.2f} K")
    print(f"Uncertainty (1 std dev): +/- {std_tc:.2f} K")
    print(f"95% Conf. Bounds: [{lower_bound:.2f} K, {upper_bound:.2f} K]")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
