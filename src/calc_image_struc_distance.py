import numpy as np
from tqdm import tqdm 
import torch
import imggen_torch as igt
import os
import MDAnalysis as mda
import sys




directory = "/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/desres"
snrlabels = [
    "nonoise",
    "snr0",  ## SNR = 1e-0
    "snr1",  ## SNR = 1e-1
    "snr2",
    "snr3",
    "snr4",
    "snr5",
    "snr6",
    "snr7",
]
snrs = [
    np.infty,
    1,
    0.1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
]

n_clusters = int(sys.argv[1])

dataset = "replicagmm%dtodesres"%n_clusters

def mdau_to_pos_arr(u, frames=None):
    protein_CA = u.select_atoms("protein and name CA")
    if frames is None:
        pos = torch.zeros((len(u.trajectory), len(protein_CA), 3), dtype=float)
        for i, ts in enumerate(u.trajectory):
            pos[i] = torch.from_numpy(protein_CA.positions)
    else:
        pos = torch.zeros((len(frames), len(protein_CA), 3), dtype=float)
        for i, ts in enumerate(u.trajectory[frames]):
            print(ts.frame)
            pos[i] = torch.from_numpy(protein_CA.positions)
    pos -= pos.mean(1).unsqueeze(1)
    return pos

print("Reading trajectory...")

## Reading image trajectory
uImg = mda.Universe(
    "/mnt/home/wtang/ceph/desres_data/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein.pdb",
    "trj/rmsfit_md_run_desres_skip5.xtc")
coord_img = mdau_to_pos_arr(uImg)

## Using K-medoids centers
M = 20
uStr = mda.Universe("/mnt/home/wtang/ceph/test_MD/chignolin/gromacs_2022/data/gro/md_run_pbc_center.gro",
    "/mnt/home/wtang/ceph/test_MD/chignolin/replicas/data/xtc/rmsfit_md_run_all_pbc_center_skip10.xtc")
frames = np.loadtxt("km_medoids_120k_%d.txt"%M).astype(int)
coord = mdau_to_pos_arr(uStr, frames)
rot_mats_align = torch.from_numpy(np.load("rot_mats_120kkm%d_desres.npy"%M))

## Using GMM centers
# coord = torch.from_numpy(np.load("/mnt/home/wtang/Code/gmm_clustering/output/gmm_centers_%d.npy"%n_clusters).astype(float))
# coord -= coord.mean(1).unsqueeze(1)
# rot_mats_align = torch.from_numpy(np.load("rot_mats_120kgmm%d_desres.npy"%n_clusters))

n_struc = coord.shape[0]

n_batch = 10

for snr, snrlabel in zip(snrs, snrlabels):

    file_prefix = "npix256_ps015_s15_%s_skip5_n1"%snrlabel
    batch_start = 0
    n_batch = 10
    n_frame = len(uImg.trajectory) # pos.shape[0]
    batch_size = int(n_frame/n_batch)
    while batch_size*n_batch < n_frame:
        batch_size += 1

    print(file_prefix)

    for i_batch in range(n_batch):

        print("Batch %d"%i_batch)
        pos_batch = coord_img[i_batch*batch_size:(i_batch+1)*batch_size].cuda()
        rot_mats, ctfs, images = igt.generate_images(
                pos_batch,
                num_images_per_struc = 1,
                n_pixels = 256,  ## use power of 2 for CTF purpose
                pixel_size = 0.15,
                sigma = 1.5,
                snr = snr,
                ctf = True,
                batch_size = 20,
            )

        # np.save('%s/rot_mats_%s_batch%d.npy'%(directory, file_prefix, i_batch), rot_mats)
        # np.save('%s/ctf_%s_batch%d.npy'%(directory, file_prefix, i_batch), ctfs)
        # np.save('%s/images_%s_batch%d.npy'%(directory, file_prefix, i_batch), images)

        np.save('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/desres/rot_mats_npix256_ps015_s15_snr10_skip5_n1_batch%d.npy'%i, rot_mats)
        np.save('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/desres/ctf_npix256_ps015_s15_snr10_skip5_n1_batch%d.npy'%i, ctfs)
        np.save('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/desres/images_npix256_ps015_s15_snr10_skip5_n1_batch%d.npy'%i, images)

        print("Batch %d..."%i_batch)

        # rot_mats = torch.from_numpy(rot_mats)
        # ctfs = torch.from_numpy(ctfs)
        # images = torch.from_numpy(images)

        rot_mats = torch.from_numpy(np.load('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/desres/rot_mats_%s_batch%d.npy'%(file_prefix,i_batch)))
        ctfs = torch.from_numpy(np.load('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/desres/ctf_%s_batch%d.npy'%(file_prefix,i_batch)))
        images = torch.from_numpy(np.load('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/desres/images_%s_batch%d.npy'%(file_prefix,i_batch)))
        
        n_image = images.shape[0]
        batch_end = batch_start + n_image

        # if not os.path.exists('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/diff_%s_%s_batch%d.npy'%(dataset, snrlabel, i_batch)):

        diff = np.zeros((n_struc, n_image), dtype=float)
        for i in tqdm(range(n_struc)):
            aligned_coord = coord[i].unsqueeze(0).matmul(rot_mats_align[i,batch_start:batch_end]).matmul(rot_mats)
            diff[i] = igt.calc_struc_image_diff(
                aligned_coord, #.cuda(),
                n_pixels = 256,  ## use power of 2 for CTF purpose
                pixel_size = 0.15, 
                sigma = 1.5, 
                images = images,
                ctfs = ctfs,
                batch_size = 16,
                device = "cpu",
            )

        print("Saving...")
        np.save('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/diff_%s_%s_batch%d.npy'%(dataset, snrlabel, i_batch), diff)

        batch_start = batch_end