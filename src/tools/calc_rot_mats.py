# This script calculates the rotation matrices that aligns the image trajectories to the structure trajectories
# and return rotation matrices in .npy format, i.e. rot_mats_struc_image.npy

import argparse


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="calc_rot_mats",
        description="Calculate rotation matrices relating \
            a conformational ensemble with a cryo-EM \
            particles ensemble",
        # epilog = 'Text at the bottom of help'
    )

    parser.add_argument(
        "-ip",
        "--top_image",
        type=str,
        default="data/image.pdb",
        help="topology file for image-generating trajectory",
    )
    parser.add_argument(
        "-it",
        "--traj_image",
        type=str,
        default="data/image.xtc",
        help="trajectory file for image-generating trajectory",
    )
    parser.add_argument(
        "-sp",
        "--top_struc",
        type=str,
        default="data/struc.pdb",
        help="topology file for structure trajectory",
    )
    parser.add_argument(
        "-st",
        "--traj_struc",
        type=str,
        default="data/struc_m10.xtc",
        help="trajectory file for structure trajectory",
    )

    parser.add_argument(
        "-o", "--outdir", default="output/", help="directory for output files"
    )
    parser.add_argument(
        "-nb",
        "--n_batch",
        type=int,
        default=5,
        help="number of batch to separate the output files into for memory management",
    )
    return parser

######## ######## ######## ########


def main(args=None):
    import MDAnalysis as mda
    import numpy as np
    import os
    import torch
    from tqdm import tqdm
    from .mdau_to_pos_arr import mdau_to_pos_arr

    # This uses a faster torch SVD implementation from 
    # https://github.com/KinglittleQ/torch-batch-svd
    from torch_batch_svd import svd 

    if args is None:
        parser = _parse_args()
        args = parser.parse_args()

    top_image = args.top_image
    traj_image = args.traj_image
    top_struc = args.top_struc
    traj_struc = args.traj_struc

    outdir = args.outdir
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    n_batch = args.n_batch

    print("Reading trajectory files...")

    uImage = mda.Universe(top_image, traj_image)
    uStruc = mda.Universe(top_struc, traj_struc)

    posStruc = mdau_to_pos_arr(uStruc).cuda()
    posImage = mdau_to_pos_arr(uImage).cuda()

    batch_size = int(posStruc.shape[0]/n_batch)
    while batch_size * n_batch < posStruc.shape[0]:
        batch_size += 1

    rot_mats = torch.empty((posStruc.shape[0], posImage.shape[0], 3, 3), dtype=torch.float64, device="cpu")

    print("Calculating rotation matrices...")
    for i_batch in tqdm(range(n_batch)):
        batch_start  = i_batch*batch_size
        batch_end = (i_batch+1)*batch_size
        Hs = torch.einsum('nji,mjk->nmik', posImage, posStruc[batch_start:batch_end])
        u, s, v = svd(Hs.flatten(0,1))
        torch.cuda.empty_cache()
        R = torch.matmul(v, u.transpose(1,2))
        rot_mats[batch_start:batch_end] = R.cpu().reshape(Hs.shape).transpose(0,1)

    print("Saving files...")
    out_filename = outdir + "/rot_mats_struc_image.npy"
    np.save(out_filename, rot_mats)

    print("Complete!")


if __name__ == "__main__":
    main()