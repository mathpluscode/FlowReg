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

## Examples

Pretrained checkpoints are available on [HuggingFace Hub](https://huggingface.co/mathpluscode/FlowReg).

| Model   | Dataset | Direction | Subfolder              |
|---------|---------|-----------|------------------------|
| FlowReg | ACDC    | ED→ES     | `flowreg/acdc_ed2es`   |
| FlowReg | ACDC    | ES→ED     | `flowreg/acdc_es2ed`   |
| FlowReg | M&Ms-2  | ED→ES     | `flowreg/mnms2_ed2es`  |
| FlowReg | M&Ms-2  | ES→ED     | `flowreg/mnms2_es2ed`  |
| CorrMLP | ACDC    | ED→ES     | `corrmlp/acdc_ed2es`   |
| CorrMLP | ACDC    | ES→ED     | `corrmlp/acdc_es2ed`   |
| CorrMLP | M&Ms-2  | ED→ES     | `corrmlp/mnms2_ed2es`  |
| CorrMLP | M&Ms-2  | ES→ED     | `corrmlp/mnms2_es2ed`  |

Example scripts under `examples/` download pretrained models and datasets from HuggingFace and run inference on a single test sample.

```bash
# CorrMLP
python examples/inference_corrmlp.py --model_id acdc_ed2es
python examples/inference_corrmlp.py --model_id acdc_es2ed
python examples/inference_corrmlp.py --model_id mnms2_ed2es
python examples/inference_corrmlp.py --model_id mnms2_es2ed

# FlowReg
python examples/inference_flowreg.py --model_id acdc_ed2es
python examples/inference_flowreg.py --model_id acdc_es2ed
python examples/inference_flowreg.py --model_id mnms2_ed2es
python examples/inference_flowreg.py --model_id mnms2_es2ed
```
