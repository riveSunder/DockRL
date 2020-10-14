import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser("receptor and ligand")
    parser.add_argument("-r", "--receptor", type=str, \
            default="1oyt", help="name of receptor (and .pdb file)")
    parser.add_argument("-l", "--ligand", type=str, \
            default="fsn", help="name of ligand")
    parser.add_argument("-t", "--test_set", type=bool,
            default=True, help="assign protein-ligand to test dataset?")

    args = parser.parse_args()
    receptor = args.receptor
    ligand = args.ligand

    data_dir = "test" if args.test_set else "train"

    os.system('pymol ./data/pdb/sources/{}.pdb '.format(receptor) \
            + '-d "remove resn HOH; h_add elem O or elem N; select {}-{}, resn {}; '.format(receptor, ligand, ligand) \
            + 'select {}-receptor, {} and not {}-{}; '.format(receptor, receptor, receptor, ligand) \
            + 'save ./data/{}/receptors/{}-receptor.pdb, {}-receptor; '.format(data_dir, receptor, receptor) \
            + 'save ./data/{}/ligands/{}-{}.pdb, {}-{}; quit;"'.format(data_dir, receptor, ligand, receptor, ligand))

    #convert pdb files to pdbqt for smina
    os.system('obabel ./data/{}/ligands/{}-{}.pdb '.format(data_dir, receptor, ligand)\
            + '-xr -O ./data/{}/ligands/{}-{}.pdbqt'.format(data_dir, receptor, ligand))

    os.system('obabel ./data/{}/receptors/{}-receptor.pdb '.format(data_dir, receptor)\
            + '-xr -O ./data/{}/receptors/{}-receptor.pdbqt'.format(data_dir, receptor))
            
