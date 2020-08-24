# NeuroPsy Model Comparison

A bayesian model comparison with the data that was gathered by the 
[NeuroPsy Research app](https://github.com/OlafHaag/NeuroPsyResearchApp) and preprocessed with the accompanying 
[Analysis Dash app](https://ucmwebapp.herokuapp.com).   


## Description

There are several hypotheses about the distributions of the data in question.
This project examines the probabilities of the hypotheses given the data. 

## Installation
If you're on Windows 10 I recommend using WSL2 for the project. It's easier to setup and run than in a pure Windows environment.  
For PyMC3 to run fast you need the g++ compiler.
```
apt-get install -y build-essential
```

You need to have [miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) or the Anaconda distribution installed.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

You could use [mamba](https://github.com/TheSnakePit/mamba/) for faster dependency solving.
```
conda install mamba -c conda-forge
```
Use conda/mamba to install the environment, e.g.
```
mamba env update --prefix ./condaenv -f environment.yml
```
This will install install the environment into a subfolder of the project called *'condaenv'*.

This project comes with a python package (*src/neuropsymodelcomparison*) that gets installed as part of the environment in editable mode.

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
