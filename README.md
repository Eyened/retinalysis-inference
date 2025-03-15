## Retinalysis models inference

This repository implements inference ensembles for retinalysis model releases. It includes the same pre-processing code that was used to train the models. For example, fundus preprocessing will detect bounds, crop the smalles square that contains these bounds, and resize it to a fixed resolution, currently 1024x1024px.

- VascX models are available in [this huggingface repository](https://huggingface.co/Eyened/vascx) but don't need to be downloaded manually. See [this notebook](./notebooks/inference.ipynb).

Models have been tested to run on a single nvidia GPU with at least 10GB VRAM. Using them for distributed inference in multiple GPUs will require some adaptation.

### Installation

1. Install torch and torchvision that match your cuda environment. For example:
```
pip3 install torch torchvision torchaudio  # pip and CUDA 12
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # conda and CUDA 12
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # pip and CUDA 11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # conda and CUDA 11
```

Pytorch installation instructions are [here](https://pytorch.org/get-started/locally/).

We did not include torch as a dependency of rtnls-inference. These must be installed manually beforehand. 

2. Install rtnls_fundusprep and rtnls-inference:

```
pip install retinalysis-fundusprep
pip install retinalysis-inference
```

3. Done! You can now download and use rtnls-inference ensembles. See [this notebook](./notebooks/inference.ipynb) for an example. The models are automatically downloaded from huggingface.

4. (optional) To be able to load manually-downloaded models by name from a folder define:

```
export RTNLS_MODEL_RELEASES = /path/to/models
```

