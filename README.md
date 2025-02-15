# Pytorch tutorial

```shell
git clone https://github.com/ChernLab/pytorch-tutorial.git
cd pytorch-tutorial
# ignore notebook changes
git update-index --assume-unchanged notebooks/*.ipynb
# undo it by
# git update-index --no-assume-unchanged notebooks/*.ipynb
```

```shell
# the environment should already exist on Prof. Chern’s computer
# conda env create -f environment.yml
# conda activate pytorch-tutorial
# python -m ipykernel install --user --name "pytorch-tutorial"

conda activate pytorch-tutorial
cuda_switch 12.4 # this is a custom shell command
nvcc --version # check CUDA version is 12.4
```
