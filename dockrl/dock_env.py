import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import time
import os 
from subprocess import check_output

class DockEnv():

    def __init__(self):

        self.receptors_dir = os.listdir("./data/train/receptors/")
        self.ligands_dir = os.listdir("./data/train/ligands/")

        self.receptors_test_dir = os.listdir("./data/test/receptors/")
        self.ligands_test_dir = os.listdir("./data/test/ligands/")
        
        self.ligand = None
        self.receptor = None
        self.exhaustiveness = 4
        self.smina_cpu = 4
        self.max_steps = 1


    def run_docking(self, action=None, idx=None):

        assert self.ligand is not None, "should be unreachable, call env.reset() first"


#        try: 
        if idx is None:
            obj_filename = "./scoring/dock.score"
            output_filename =  "./output/{}-redocking.pdbqt".format(self.ligand[0:4]) 
        else:
            obj_filename = "./scoring/dock{}.score".format(idx)
            output_filename =  "./output/{}-redocking{}.pdbqt"\
                    .format(self.ligand[0:4], idx) 

        if action is None:
            my_command = "./smina.static"\
                    +" -r ./data/{}/receptors/{}".format(self.mode, self.receptor) \
                    + " -l ./data/{}/ligands/{}".format(self.mode, self.ligand) \
                    + " --autobox_ligand ./data/{}/ligands/{}".format(self.mode, self.ligand) \
                    + " --autobox_add 8 --exhaustiveness {}".format(self.exhaustiveness) \
                    + " -o {}".format(output_filename)\
                    + " -q --cpu {}".format(self.smina_cpu) 
        else:
            #action = np.tanh(action)
            f = open(obj_filename,'w')

            my_scoring_weights = "{:.8f} gauss(o=0,_w=0.5,_c=8)\n".format(action[0])\
                    + "{:.8f} gauss(o=3,_w=2,_c=8)\n".format(action[1])\
                    +"{:.8f} hydrophobic(g=0.5,_b=1.5,_c=8)\n".format(action[2])\
                    +"{:.8f} non_dir_h_bond(g=-0.7,_b=0,_c=8)\n".format(action[3])\
                    + "{:.8f} repulsion(o=0,_c=8)\n".format(action[4])\
                    + "{:.8f} num_tors_div".format(action[5])

            
            f.write(my_scoring_weights)
            f.close()

            my_command = "./smina.static --custom_scoring {}".format(obj_filename)\
                    + " -r ./data/{}/receptors/{}".format(self.mode, self.receptor) \
                    + " -l ./data/{}/ligands/{}".format(self.mode, self.ligand) \
                    + " --autobox_ligand ./data/{}/ligands/{}".format(self.mode, self.ligand) \
                    + " --autobox_add 8 --exhaustiveness {}".format(self.exhaustiveness) \
                    + " -o {}".format(output_filename)\
                    + " -q --cpu {}".format(self.smina_cpu) 

        #os.system(my_command)
        try:
            temp = check_output(my_command.split(), timeout=60)
        except:
            print(my_command.splitlines())
            print("timeout occured, attempting again")
            self.run_docking(action=action, idx=idx)
        
        #check if autodock program output a blank pdbqt file
        f = open(output_filename, 'r')
        f_lines = f.readlines()
        f.close()

        if len(f_lines) == 0:
            
            print(my_command.splitlines())
            print("sim output blank file, attempting to run again")
            self.run_docking(action=action, idx=idx)
    
    def get_rmsd(self, worker_idx=None):

        #try: 
        if worker_idx is None:
            redock_fn = "./output/{}-redocking.pdbqt".format(self.ligand[0:4])
        else:
            redock_fn = "./output/{}-redocking{}.pdbqt".format(self.ligand[0:4], worker_idx)

        
        f = open(redock_fn, 'r')
        f_gt = open("./data/{}/ligands/{}".format(self.mode, self.ligand), 'r')
    
        stop = False
        rsd = 0.0
        count = 0

        gt_lines = f_gt.readlines()
        comp_lines= f.readlines() 

        
#            while ("ATOM" not in gt) or ('1' not in gt):
#                gt = f_gt.readline().split()
#            while ("ATOM" not in comp) or ('1' not in comp):
#                comp = f.readline().split()


        #while not stop:
        old_jj = 0
        for ii in range(len(gt_lines)): #gt_line, comp_line in zip(f_gt.readlines(), f.readlines()):

            try:
                if "ATOM" in gt_lines[ii]:
                    gt = gt_lines[ii].split()

                    for jj in range(old_jj, len(comp_lines)):
                        if "ATOM" in comp_lines[jj]:
                            comp = comp_lines[jj].split()

                            if comp[1] == gt[1]:
                                count += 1
                                coords_gt = np.array([float(elem) for elem in gt[5:8]])
                                coords_comp = np.array([float(elem) for elem in comp[5:8]])

                                rsd += np.sum(np.sqrt((coords_gt - coords_comp)**2))
                                old_jj = jj
                            elif jj == (len(comp_lines)-1):
                                print("matching atom not found in output file")
            except:
                import pdb; pdb.set_trace()
                    


#            if count > 0 and "TORSDOF" in gt:
#                stop = True

        if count > 0:
            rmsd = rsd / count
        else: 
            print("mismatch in output/reference ligands? rerun docking with default params") 
            print(self.ligand, redock_fn)
            self.run_docking(idx=worker_idx)

            rmsd = self.get_rmsd(worker_idx=worker_idx)
            #rmsd = 20.0

        f.close()
        f_gt.close()

        #except:
        #print("there's been a problem calculating rmsd")
        #rmsd = 10.0

        return rmsd

    def get_default_rmsd(self, mode="test"):

        rmsd = 0.0
        num_docks = 50

        num_samples = len(self.ligands_test_dir) if mode=="test" else len(self.ligands_dir)
        for ii in range(num_docks):
            
            #_ = self.reset(test=(mode == "test"))
            _ = self.reset(test=(mode == "test"), sample_idx=ii % num_samples)
            self.run_docking(action=None)

            rmsd_temp = self.get_rmsd()
            
            rmsd += rmsd_temp 


        rmsd /= num_docks

        print("Average rmsd over {} runs with default score weighting  = {:.3f}"\
                .format(num_docks, rmsd))

        return rmsd

    def get_bjerrum_rmsd(self, mode="test"):


        rmsd = 0.0
        num_docks = 50

        action = np.array([-0.0460161,\
                -0.000384274,\
                -0.00812176,\
                -0.431416,\
                0.366584,\
                0.0])

        num_samples = len(self.ligands_test_dir) if mode=="test" else len(self.ligands_dir)
        for ii in range(num_docks):
            
            #_ = self.reset(test=(mode == "test"))
            _ = self.reset(test=(mode == "test"), sample_idx=ii % num_samples)
            self.run_docking(action)

            rmsd_temp = self.get_rmsd()
            rmsd += rmsd_temp 

        rmsd /= num_docks

        print("Average rmsd over {} runs with weighting from (Bjerrum 2016) = {:.3f}"\
                .format(num_docks, rmsd))

        return rmsd


    def step(self, action, worker_idx=None):

        assert self.ligand is not None, "Must call env.reset() before env.step(action)"

        self.run_docking(action, idx=worker_idx)

        rmsd = self.get_rmsd(worker_idx=worker_idx)

        # an rmsd of less than 2.0 angstroms is generally considered "accurate"
        #reward = 10.0 if rmsd <= 2.0 else -rmsd
        reward = -rmsd

        # regularization/
        l1_reg = 0.0 #1e-1
        l2_reg = 0.0# 1e-1
        reward -= l1_reg * np.sum(np.abs(action)) + l2_reg * np.sum(np.abs(action**2))

        obs = np.append(action, reward)

        self.steps += 1

        if self.steps < self.max_steps:
            done = False
        else:
            done = True


        info = {"rmsd": rmsd}

        return obs, reward, done, info 
        
    def reset(self, test=False, sample_idx=None):

        self.steps = 0
        if test:
            self.mode = "test"
        else:
            self.mode = "train"

        if test:
            my_ligands_dir = self.ligands_test_dir
            my_receptors_dir = self.receptors_test_dir
        else: 
            my_ligands_dir = self.ligands_dir
            my_receptors_dir = self.receptors_dir

        if sample_idx is None:
            self.ligand = np.random.choice(my_ligands_dir, \
                    p=[1/len(my_ligands_dir) for elem in my_ligands_dir])
        else:
            self.ligand = my_ligands_dir[sample_idx]

        for receptor in my_receptors_dir:
            if self.ligand[0:4] in receptor:
                self.receptor = receptor
                break

        obs = np.zeros((7))

        return obs


if __name__ == "__main__":

    env = DockEnv()

    obs = env.reset()
    
    done = False
    reward_sum = 0.0

    env.get_bjerrum_rmsd()
    env.get_default_rmsd()

    import pdb; pdb.set_trace()
    while not done:
        obs, reward, done, info = env.step(np.random.randn(6))

        reward_sum += reward

    print("cumulative reward: {}".format(reward_sum))
