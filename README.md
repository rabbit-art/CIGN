# CIGN

Assume the project is located at `/path/to/CIGN` and your Python environment at `/path/to/conda/envs/cign`.

1. Install dependencies in an environment with PyTorch, PyG, and dgNN configured:

```bash
PYTHONNOUSERSITE=1 /path/to/conda/envs/cign/bin/python -m pip install -r /path/to/CIGN/requirements.txt
```

2. Edit `USER_CONFIG` in `trainb9_use.py`: set `train_script` to `"local_train.py"`. In the existing `base_config`, set `dataset_name` to `"Amazon-ratings"`, set `data_root` to your data directory, and adjust the other hyperparameters as needed.

3. Run:

```bash
PYTHONNOUSERSITE=1 /path/to/conda/envs/cign/bin/python /path/to/CIGN/trainb9_use.py
```
