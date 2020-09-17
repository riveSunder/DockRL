# DockRL

<div align="center">
<img src="./assets/dockrl_overview.png" width="90%">
</div>


## Requirements

You'll need PyTorch, NumPy, and a protein-ligand docking simulation program called Smina, which can be downloaded from [SourceForge](https://sourceforge.net/projects/smina), to use this repository. Smina is a fork of open-source [Autodock Vina](http://vina.scripps.edu/) and has a static executable that needs to be placed in the repositories root folder. 

Note that currently you need to copy the receptor and ligand `pdbqt` files from their folders in `data` to the root folder as well. Smina appears to only look in the root folder when a relative path is provided. 

Also note Smina doesn't currently make use of the `--cpu` flag. 
