#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# =========================
# Model paths
# =========================
UNIFIED_MODEL = Path("models/unified_mlp_pipeline.joblib")
MODEL_457 = Path("models/mlp_pipeline_457nm_aberr.joblib")

# =========================
# Constants
# =========================
TARGET_457 = 457.1
TOL_457 = 0.2
HC_eVnm = 1239.841984

LINE_PHYSICS = {
    416.73: dict(elo_eV=4.3458, lggf=-0.746, log_gamma_rad=8.69, sigma=5296.0, alpha=0.508),
    470.30: dict(elo_eV=4.3458, lggf=-0.456, log_gamma_rad=8.70, sigma=2827.0, alpha=0.264),
    473.00: dict(elo_eV=4.3458, lggf=-2.379, log_gamma_rad=8.68, sigma=5928.0, alpha=0.435),
    516.73: dict(elo_eV=2.7091, lggf=-0.854, log_gamma_rad=8.02, sigma=731.0,  alpha=0.240),
    571.11: dict(elo_eV=4.3458, lggf=-1.742, log_gamma_rad=8.69, sigma=1841.0, alpha=0.120),
}

# =========================
# Helpers
# =========================
def nearest_line(lam):
    arr = np.array(list(LINE_PHYSICS.keys()))
    return float(arr[np.argmin(np.abs(arr - lam))])

def n_air_ciddor(lam_air_nm):
    lam_um = lam_air_nm / 1000.0
    s = 1.0 / lam_um
    n_minus_1 = 1e-8 * (5792105.0/(238.0185 - s**2) + 167917.0/(57.362 - s**2))
    return 1.0 + n_minus_1

# =========================
# Main routine
# =========================
def main(infile, outfile):
    df = pd.read_csv(infile)

    required = {"Teff", "logg", "A(Mg)", "vmic", "lambda"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    out = df.copy()
    out["aberr"] = np.nan
    out["model_used"] = ""

    # --- 457 nm model ---
    mask_457 = abs(out["lambda"] - TARGET_457) <= TOL_457
    if mask_457.any():
        model_457 = joblib.load(MODEL_457)
        X_457 = out.loc[mask_457, ["Teff", "logg", "A(Mg)", "vmic"]]
        out.loc[mask_457, "aberr"] = model_457.predict(X_457)
        out.loc[mask_457, "model_used"] = "457nm"

    # --- Unified model ---
    mask_uni = ~mask_457
    if mask_uni.any():
        model_uni = joblib.load(UNIFIED_MODEL)

        lam_air = out.loc[mask_uni, "lambda"].apply(nearest_line).values
        phys = pd.DataFrame([LINE_PHYSICS[l] for l in lam_air])

        lam_vac = lam_air * n_air_ciddor(lam_air)
        deltaE = HC_eVnm / lam_vac
        eup = phys["elo_eV"].values + deltaE

        X_uni = pd.DataFrame({
            "Teff": out.loc[mask_uni, "Teff"],
            "logg": out.loc[mask_uni, "logg"],
            "A(Mg)": out.loc[mask_uni, "A(Mg)"],
            "vmic": out.loc[mask_uni, "vmic"],
            "lambda_air": lam_air,
            "lambda_vac": lam_vac,
            "deltaE_eV": deltaE,
            "elo_eV": phys["elo_eV"],
            "eup_eV": eup,
            "lggf": phys["lggf"],
            "log_gamma_rad": phys["log_gamma_rad"],
            "sigma": phys["sigma"],
            "alpha": phys["alpha"],
        })

        out.loc[mask_uni, "aberr"] = model_uni.predict(X_uni)
        out.loc[mask_uni, "model_used"] = "unified"

    out.to_csv(outfile, index=False)
    print(f"Saved abundance corrections → {outfile}")

# =========================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main_mg_aberr.py input.csv output.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
