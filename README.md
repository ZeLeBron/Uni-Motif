# Uni-Motif

### Python environment setup with Conda

```bash
conda create -n molgraph python=3.10
conda activate molgraph

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

conda install fsspec rdkit -c conda-forge

pip install yacs tensorboardX
pip install ogb
pip install wandb

conda clean --all
```


### Running
```bash
conda activate molgraph

# Running with RWSE and tuned hyperparameters for ZINC.
Coming soon

### W&B logging
Set wandb.use False.
