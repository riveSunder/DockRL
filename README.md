# DockRL

<div align="center">
<img src="./assets/dockrl_overview.png" width="90%">
</div>

## Requirements

You'll need PyTorch, NumPy, and a protein-ligand docking simulation program called Smina, which can be downloaded from [SourceForge](https://sourceforge.net/projects/smina), to use this repository. Smina is a fork of open-source [Autodock Vina](http://vina.scripps.edu/) and has a static executable that needs to be placed in the repositories root folder. 

Note that currently you need to copy the receptor and ligand `pdbqt` files from their folders in `data` to the root folder as well. Smina appears to only look in the root folder when a relative path is provided. 

## Note About Multithreading

Note that in order for Smina to take advantage of multiple cores, `--exhaustiveness` needs to be greater than or equal to the number of threads you want to use. Setting the `--cpu` flag to 32 and `--exhaustiveness` to 1 will only use one thread, for exmaple, but`--exhaustiveness 32 --cpu 16` will utilize 16 as intended. 
