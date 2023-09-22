import torch

def mdau_to_pos_arr(u):
    protein_CA = u.select_atoms("protein and name CA")
    pos = torch.zeros((len(u.trajectory), len(protein_CA), 3), dtype=float)
    for i, ts in enumerate(u.trajectory):
        pos[i] = torch.from_numpy(protein_CA.positions)
    pos -= pos.mean(1).unsqueeze(1)
    return pos
