# Flow Matching-enabled Test-Time Refinement for Unsupervised Cardiac MR Registration

Flow matching for deformable 3D medical image registration, evaluated on cardiac MRI. The repository also includes reimplementations of baseline methods ([FSDiffReg](https://github.com/xmed-lab/FSDiffReg), [CorrMLP](https://github.com/MungoMeng/Registration-CorrMLP)) for comparison.

## Data Preprocessing

```bash
# ACDC
python data/acdc_preprocess.py --data_dir /path/to/acdc/database --out_dir /path/to/data

# M&Ms-2
python data/mnms2_preprocess.py --data_dir /path/to/mnms2/database --out_dir /path/to/data
```

## Training

```bash
# FlowReg
python -m flowreg.train --data_dir /path/to/data

# FSDiffReg
python -m fsdiffreg.train --data_dir /path/to/data

# CorrMLP
python -m corrmlp.train --data_dir /path/to/data
```

## Testing

```bash
# FlowReg
python -m flowreg.test --data_dir /path/to/data --checkpoint /path/to/checkpoint.pt

# FSDiffReg
python -m fsdiffreg.test --data_dir /path/to/data --checkpoint /path/to/checkpoint.pt

# CorrMLP
python -m corrmlp.test --data_dir /path/to/data --checkpoint /path/to/checkpoint.pt
```
