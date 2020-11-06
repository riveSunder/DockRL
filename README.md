# DockRL

<div align="center">
<img src="./assets/dockrl_overview.png" width="90%">
</div>

## Set up

You'll need PyTorch, NumPy, and a protein-ligand docking simulation program called Smina, which can be downloaded from [SourceForge](https://sourceforge.net/projects/smina), to use this repository. Smina is a fork of open-source [Autodock Vina](http://vina.scripps.edu/) and has a static executable that needs to be placed in the repositories root folder. 

I am using Ubuntu 18 and `virtualenv` for virtual environment management, but other Linux distributions and/or anaconda should not be too different. To initialize and activate a new virtual environment:

```
virtualenv /env_path --python=python3
ource /env_path/bin/activate
```

DockRL uses MPI for training parallelization. In order to use `mpip4py`, you'll also need to install some dev tools.

```
sudo apt update && sudo apt install libopenmpi-dev
```

Next all you need to do is run `setup.py` to make sure your virtual environment contains dependencies and that python knows where to look for `dockrl` modules. 

```
python setup.py develop
```


## Usage

Starting a typical training run looks like this

```
python dockrl/cmaes.py -p 64 -g 100 -c 8
```

The flags `-p`, `-g`, and `-c` designate the population size, total number of generations, and cpu threads to utilize during CMA-ES training. Note that Smina has its own multithreading capabilities and as of commit `a288043c` the default number of threads for docking is 4. Smina multithreading also depends on exhaustiveness, so changing Smina's `--cpu` flag without adjusting `--exhaustiveness` to be at least as high docking won't utilize all the threads you want. That being said you'll want to balance the number of threads used for CMA-ES and docking to match your machine, as each individual worker will call a seperate instance of Smina. Also remember to add 1 to the total number of CMA-ES workers to account for the mantle process that orchestrates everything.  

<div align="center">
<img src="./assets/signs_of_life.png" width=80%>
</div>


