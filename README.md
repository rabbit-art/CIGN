# CIGN

Assume the project is located at `/path/to/CIGN` and your Python environment at `/path/to/conda/envs/cign`.

1. Install dependencies in an environment with PyTorch, PyG, and dgNN configured:

```bash
PYTHONNOUSERSITE=1 /path/to/conda/envs/cign/bin/python -m pip install -r /path/to/CIGN/requirements.txt
```

2. Run the Amazon-ratings example:

```bash
cd /path/to/CIGN
PYTHONNOUSERSITE=1 /path/to/conda/envs/cign/bin/python trainb9_use.py
```
