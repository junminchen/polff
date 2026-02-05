import argparse
import os
import numpy as np
import ase.io
from scipy.spatial.transform import Rotation
from bytemol.core import Molecule
from bytemol.utils import setup_default_logging

# Setup English logging
logger = setup_default_logging()

parser = argparse.ArgumentParser('Create dimer configurations for electrolyte studies')
parser.add_argument('--mol1', type=str, default='./ACT.xyz', help='Path to mol1 xyz file')
parser.add_argument('--mol2', type=str, default='./EC.xyz', help='Path to mol2 xyz file')
parser.add_argument('--save_dir', type=str, default='./dimer_output', help='Directory to save results')
parser.add_argument('--nconfs', type=int, default=100, help='Number of conformers to generate')
parser.add_argument('--min_dist', type=float, default=1.5, help='Minimum atom-to-atom distance in Angstrom')
parser.add_argument('--max_dist', type=float, default=10., help='Maximum translation displacement in Angstrom')
parser.add_argument('--noise_std', type=float, default=0.05, help='Standard deviation for coordinate noise')
args = parser.parse_args()

np.random.seed(42)

def calc_min_dist2(coords1, coords2):
    """Calculates the minimum squared distance between two sets of coordinates."""
    c11 = coords1[:, np.newaxis]
    c11 = np.tile(c11, (1, coords2.shape[0], 1))
    c21 = coords2[np.newaxis, ...]
    c21 = np.tile(c21, (coords1.shape[0], 1, 1))
    dist2 = np.square(c21 - c11).sum(-1).min()
    return dist2

def perturb_coords(coords1, coords2, nconfs, min_dist, max_dist):
    """Generates perturbed dimer configurations."""
    coords_new = []
    disps = []
    while len(coords_new) < nconfs:
        d = np.random.uniform(0, max_dist)
        disp = np.random.uniform(-1, 1, (1, 3))
        disp = disp / np.sqrt(np.sum(np.square(disp))) * d
        rot = Rotation.random()
        
        # Apply rotation, translation, and noise
        c2_new = rot.apply(coords2) + disp + np.random.normal(0, args.noise_std, coords2.shape)
        c1_new = coords1 + np.random.normal(0, args.noise_std, coords1.shape)
        
        if calc_min_dist2(c1_new, c2_new) < min_dist**2:
            continue
            
        disps.append(d)
        coords_new.append([c1_new.copy(), c2_new.copy()])
    
    # Sort by displacement distance
    ids = np.argsort(disps)
    return [coords_new[i] for i in ids]

if __name__ == '__main__':
    # 1. Load molecules
    mol1 = Molecule.from_xyz(args.mol1)
    mol2 = Molecule.from_xyz(args.mol2)
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Handle same-molecule case for independent files
    name1 = mol1.name
    name2 = mol2.name
    if name1 == name2:
        name1 = f"{name1}_1"
        name2 = f"{name2}_2"
    
    print(f"Generating Dimer: {name1} and {name2}")

    # 3. Generate configurations
    coords_list = perturb_coords(
        mol1.conformers[0].coords,
        mol2.conformers[0].coords,
        args.nconfs,
        args.min_dist,
        args.max_dist,
    )

    # 4. Save each molecule to its own file in each conf directory
    for ci, (c1, c2) in enumerate(coords_list):
        conf_dir = os.path.join(args.save_dir, f'conf_{ci}')
        os.makedirs(conf_dir, exist_ok=True)
        
        # Update atoms with new coordinates
        atoms1 = mol1.conformers[0].to_ase_atoms()
        atoms1.positions = c1
        
        atoms2 = mol2.conformers[0].to_ase_atoms()
        atoms2.positions = c2

        # Write independent files using ASE (avoids 'append' error)
        ase.io.write(os.path.join(conf_dir, f'{name1}.xyz'), atoms1)
        ase.io.write(os.path.join(conf_dir, f'{name2}.xyz'), atoms2)

    print(f"Success: Generated {args.nconfs} conformers in '{args.save_dir}'")
