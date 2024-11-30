# D4: Distance Diffusion for a Truly Equivariant Molecular Design

This is the code which was used to run the experiments for the paper "D4: Distance Diffusion for a Truly Equivariant Molecular Design".

## Requirements

The code was tested with Python 3.11.7 and with CUDA 11.4.0. The requirements can be installed by executing 
- Download anaconda/miniconda if needed
- Create a new environment through the given environment files with the following command:
    ```bash
    conda env create -f <env_file>.yml
    ```
    where \<env_file\> is the name of the environment file to use. It is possible to install dependencies for CPU with `environment_cpu.yml` or for GPU with `environment_cuda.yml`.
- Install this package with the following command:
    ```bash
    pip install -e .
    ```
    which will compile required cython and c++ code.

### Running experiments

The experiments can be run with different modality, the basic command is:
```bash
    python main.py +preset=<preset> seed=<seed> mode=<mode>
```
where the value preset could be chosen between qm9_distance and gdb13_distance.
After the \<preset\> it's possible to choose the \<seed\>, the modality of running \<mode\> between \<eval\>, \<train+eval\> and \<train\>. The defaulf values are mode=eval and seed=0.
All the parameters relative to the training and evaluation phase could be seen inside ./config/ folder.

### Datasets

QM9 will be downloaded automatically to a new directory ./datasets when running an experiment. 
Regarding GDB13 only a random subset of the entire dataset is used, for reproducibility, this could be found directly inside ./dataset/GDB13/raw/ folder.

### Checkpoints

Checkpoints are present in a shared folder on google drive (https://drive.google.com/drive/folders/1SO7mIvRe7mZjPZ8WF14Ygo6pyK5OY3-_?usp=sharing). To run them they should be downloaded and 
placed in the ./checkpoints folder.
