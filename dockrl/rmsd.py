import numpy as np

if __name__ == "__main__":

    gt_filename = "1OYT-FSN.pdbqt"
    compare_filename = "1OYT-redock_4.pdbqt"

    f = open(compare_filename)
    f_gt = open(gt_filename)
    
    stop = False
    rsd = 0.0
    count = 0

    gt = f_gt.readline().split()
    
    comp = f.readline().split() 

    while ("ATOM" not in gt) or ('1' not in gt):
        gt = f_gt.readline().split()
    while ("ATOM" not in comp) or ('1' not in comp):
        comp = f.readline().split()
        
    print(comp)
    print(gt)

    while not stop:

        gt = f_gt.readline().split()
        comp = f.readline().split()

        if len(gt) == 12:
            count += 1
            coords_gt = np.array([float(elem) for elem in gt[5:8]])
            coords_comp = np.array([float(elem) for elem in comp[5:8]])

            rsd += np.sum(np.sqrt((coords_gt - coords_comp)**2))

        if count > 0 and len(gt) < 12:
            stop = True

    rmsd = rsd / count

    print(rmsd)

