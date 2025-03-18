# Placeholder for our cool team name


## Setup
```sh
conda create --name MAGICuda python=3.10
pip install jupyter notebook matplotlib numpy pyarrow ipython ipykernel scikit-learn pandas seaborn
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

Formatter/Linter: [Ruff](https://github.com/astral-sh/ruff)

## Setup development environment Nix Flakes
```sh
nix develop --no-pure-eval
```

Alternatively use [nix-direnv](https://github.com/nix-community/nix-direnv) to automatically activate the devshell.