# Mg I abundance corrections

This repository provides machine-learning–based abundance corrections  
(1D LTE − 3D NLTE) for selected Mg I spectral lines.

## Supported lines (nm)
- 416.7
- 457.1 (dedicated model)
- 470.3
- 473.0
- 516.7
- 571.1

## Requirements 
numpy\
pandas\
scikit-learn==1.5.1\
joblib

## Usage
```bash
python3 main_mg_aberr.py input_file output_file
```

## Input Format
The input file can be in CSV or XLSX format. The following columns are required:\

Teff: Effective temperature (K).\

logg: Surface gravity ($\log g$).\

A(Mg): The 1D-LTE magnesium abundance (used as the baseline).\

vmic: Microturbulent velocity (km/s).\

line: The central wavelength of the Mg I line (nm).\

### Example 
Teff,logg,A(Mg),vmic,line

5050,4.0,7.2,1.0,457.1

5750,4.5,7.0,1.2,516.7

## Output
The output file will include all original columns plus a new column:\

abundance_error (1L-3N): This represents the difference between the 1D-LTE and 3D-NLTE magnesium abundances.

The code automatically selects a dedicated 457.1 nm model or a unified multiline model.

## Contact
This work is part of the project “Machine Learning Interpolation of Stellar Abundance Corrections”, carried out under the supervision of Anish Amarsi at Uppsala University.

For questions or issues, please contact: biliroglubengu@gmail.com
