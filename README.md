# Mg I abundance corrections

This repository provides machine-learning–based abundance corrections  
(1D LTE − 3D NLTE) for selected Mg I spectral lines.

## Supported lines (air wavelengths, nm)
- 416.7
- 457.1 (dedicated model)
- 470.3
- 473.0
- 516.7
- 571.1

## Requirements 
numpy\
pandas\
scikit-learn\
joblib

## Usage
```bash
python3 main_mg_aberr.py input_file output_file
```

## Input Format
The input file may be provided in CSV or XLSX format and must contain the columns:

Teff,logg,A(Mg),vmic,line

5050,4.0,7.2,1.0,457.1

5750,4.5,7.0,1.2,516.7

## Output
The output file contains an additional column:
aberr : abundance correction (dex)

The code automatically selects a dedicated 457.1 nm model or a unified multiline model.

## Contact
This work is part of the project “Machine Learning Interpolation of Stellar Abundance Corrections”, carried out under the supervision of Anish Amarsi at Uppsala University.

For questions or issues, please contact: biliroglubengu@gmail.com
