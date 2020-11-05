from setuptools import setup


setup(name="DockRL",\
        py_modules=["dockrl"],\
        version="0.0",\
        install_requires=["numpy", "torch", "mpi4py"],\
        description="Protein-ligand docking RL/evo",\
        author="Rive Sunder",\
        )

