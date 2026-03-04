#!/usr/bin/env python3
"""
Mg I abundance corrections (1D LTE − 3D NLTE)

Usage:
    python3 main_mg_aberr.py input.csv output.csv
    python3 main_mg_aberr.py input.xlsx output.csv

Input columns:
    Teff, logg, A(Mg), vmic, line
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import re

from pathlib import Path

#%%
# Physical Constants (For Unified Model) 

HC_eVnm = 1239.841984
LINE_PHYSICS = {
    416.73: dict(elo_eV=4.3458, lggf=-0.746, log_gamma_rad=8.69, sigma=5296.0, alpha=0.508),
    470.30: dict(elo_eV=4.3458, lggf=-0.456, log_gamma_rad=8.70, sigma=2827.0, alpha=0.264),
    473.00: dict(elo_eV=4.3458, lggf=-2.379, log_gamma_rad=8.68, sigma=5928.0, alpha=0.435),
    516.73: dict(elo_eV=2.7091, lggf=-0.854, log_gamma_rad=8.02, sigma=731.0,  alpha=0.240),
    571.11: dict(elo_eV=4.3458, lggf=-1.742, log_gamma_rad=8.69, sigma=1841.0, alpha=0.120),
}
UNIFIED_LINES = list(LINE_PHYSICS.keys())

#%%

def n_air_ciddor(lam_air_nm):
    """Calculates the refractive index of standard dry air."""
    lam_um = np.asarray(lam_air_nm, dtype=float) / 1000.0
    s = 1.0 / lam_um
    n_minus_1 = 1e-8 * (5792105.0/(238.0185 - s**2) + 167917.0/(57.362 - s**2))
    return 1.0 + n_minus_1

def get_features_unified(row, lam):
    """Derives necessary physical features for the unified model."""
    p = LINE_PHYSICS[lam]
    n_air = n_air_ciddor(lam)
    lam_vac = lam * n_air
    deltaE = HC_eVnm / lam_vac
    return {
        "Teff": row["Teff"], "logg": row["logg"], "A(Mg)": row["A(Mg)"], "vmic": row["vmic"],
        "lambda_air": lam, "lambda_vac": lam_vac, "deltaE_eV": deltaE,
        "elo_eV": p["elo_eV"], "eup_eV": p["elo_eV"] + deltaE,
        "lggf": p["lggf"], "log_gamma_rad": p["log_gamma_rad"],
        "sigma": p["sigma"], "alpha": p["alpha"]
    }

def main():
    parser = argparse.ArgumentParser(description="Mg I Abundance Correction Predictor")
    parser.add_argument("input", help="Input file (Excel or CSV). Must contain Teff, logg, A(Mg), vmic, line.")
    parser.add_argument("output", help="Output file path (.xlsx or .csv)")
    args = parser.parse_args()

    # Load models
    try:
        model_unified = joblib.load("unified_mlp_pipeline.joblib")
        model_457 = joblib.load("mlp_pipeline_457nm_aberr.joblib")
    except FileNotFoundError as e:
        print(f"Error: Model files (.joblib) not found in the script's directory.\n{e}")
        return

    # Input Reading
    input_path = Path(args.input)
    if input_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(args.input)
    elif input_path.suffix.lower() == '.csv':
        df = pd.read_csv(args.input)
    else:
        print(f"Unsupported input format: {input_path.suffix}. Please use .xlsx or .csv")
        return

    df.columns = df.columns.str.strip()
    required = ["Teff", "logg", "A(Mg)", "vmic", "line"]
    missing = [col for col in required if col not in df.columns]
    if missing: 
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)
    
    results = []
    print(f"Processing {len(df)} rows from {input_path.name}...")

    for idx, row in df.iterrows():
        try:
            line_val = float(row["line"])
            correction = np.nan
            
            if abs(line_val - 457.1) <= 0.2:
                feat = pd.DataFrame([row[["Teff", "logg", "A(Mg)", "vmic"]].values], 
                                    columns=["Teff", "logg", "A(Mg)", "vmic"])
                correction = model_457.predict(feat)[0]
            
            elif any(abs(line_val - l) <= 0.2 for l in UNIFIED_LINES):
                true_lam = min(UNIFIED_LINES, key=lambda x: abs(x - line_val))
                feat_dict = get_features_unified(row, true_lam)
                feat_df = pd.DataFrame([feat_dict])
                correction = model_unified.predict(feat_df)[0]
            
            results.append(correction)
        except Exception as e:
            results.append(np.nan)
            print(f"Error at row {idx+1}: {e}")

    df["abundance_error (1L-3N)"] = results
    
    # Output Saving
    if args.output.lower().endswith(".csv"):
        df.to_csv(args.output, index=False)
    else:
        df.to_excel(args.output, index=False)
    
    print(f"\nSuccessful, results now saved to '{args.output}'")

if __name__ == "__main__":
    main()
