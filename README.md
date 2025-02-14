# Pytorch tutorial

```shell
# conda create -n pytorch-tutorial python=3.11
# conda activate pytorch tutorial
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install numpy scipy scikit-learn matplotlib tqdm ipykernel
# python -m ipykernel install --user --name "pytorch-tutorial"

conda activate pytorch-tutorial
cuda_switch 12.4 # this is custom shell command
nvcc --version # check CUDA version is 12.4
```
