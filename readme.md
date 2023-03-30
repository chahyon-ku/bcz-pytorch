
# Installation

## Create conda environment from `environment.yml`:
```bash
mamba env create -f environment.yml
```

## Install [CoppeliaSim](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)
Donwload and extract to EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR

Add following to ~/.bashrc:
```bash
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

## Install RLBench
```bash
pip install git+https://github.com/stepjam/RLBench.git --no-deps
pip install git+https://github.com/stepjam/PyRep.git --no-deps
```