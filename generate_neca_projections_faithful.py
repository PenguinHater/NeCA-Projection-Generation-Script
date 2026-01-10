import sys
import os
import numpy as np
import torch

# --------------------------------------------------
# Force Python to use the project root first
# --------------------------------------------------
PROJECT_ROOT = "/home/george/NeCA_Master/NeCA-main"
sys.path.insert(0, PROJECT_ROOT)

# Now imports will always use the same files as trainer.py
from src.render.ct_geometry_projector import ConeBeam3DProjector
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to

# --------------------------------------------------
# Paths
# --------------------------------------------------
INPUT_NPY  = os.path.join(PROJECT_ROOT, "data/CCTA_GT/1.npy")
OUTPUT_NPY = os.path.join(PROJECT_ROOT, "data/CCTA_test/data.npy")

# --------------------------------------------------
# Geometry (match config/CCTA.yaml)
# --------------------------------------------------
DSD = [993.0, 1055.0]
DSO = [757.06045586378, 756.9934313361]
DDE = [235.93954413621998, 298.0065686639]

nDetector = [512, 512]
dDetector = [0.278, 0.278]

nVoxel = [128, 128, 128]
dVoxel = [0.8, 0.8, 0.8]

first_projection_angle  = [29.7, 0.1]
second_projection_angle = [2.0, 29.0]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def rotation_matrix_to_axis_angle(m):
    angle = np.arccos((m[0,0] + m[1,1] + m[2,2] - 1) / 2)
    axis = np.array([
        m[2,1] - m[1,2],
        m[0,2] - m[2,0],
        m[1,0] - m[0,1],
    ])
    axis = axis / np.linalg.norm(axis)
    return axis, angle

def make_projector(angle_pair, idx):
    """Trainer-faithful CPU projector construction"""

    # Step 1, Conversion from [pitch, yaw] to [yaw, pitch]
    proj_angle = [-angle_pair[1], angle_pair[0]]

    # Step 2, Define reference vectors
    from_source_vec = (0, -DSO[idx], 0)
    from_rot_vec    = (-1, 0, 0)

    # Step 3, Perform yaw rotation
    to_source_vec = axis_rotation(
        (0, 0, 1),
        angle=proj_angle[0] / 180.0 * np.pi,
        vectors=from_source_vec
    )

    to_rot_vec = axis_rotation(
        (0, 0, 1),
        angle=proj_angle[0] / 180.0 * np.pi,
        vectors=from_rot_vec
    )

    # Step 4, Perform pitch rotation
    to_source_vec = axis_rotation(
        to_rot_vec[0],
        angle=proj_angle[1] / 180.0 * np.pi,
        vectors=to_source_vec[0]
    )

    # Step 5, Convert to axis-angle for ODL
    rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
    axis, angle = rotation_matrix_to_axis_angle(rot_mat)

    # Step 6, Contruct ConeBeam3DProjector
    return ConeBeam3DProjector(
        nVoxel,
        dVoxel,
        angle,
        axis,
        nDetector,
        dDetector,
        DDE[idx],
        DSO[idx]
    )

# --------------------------------------------------
# Main
# --------------------------------------------------
print("[INFO] Loading GT volume...")
vol = np.load(INPUT_NPY).astype(np.float32)     # (128,128,128)
vol = vol[None, None, ...]                      # (1,1,128,128,128)
vol_t = torch.from_numpy(vol)

print("[INFO] Creating projectors (trainer-faithful CPU)...")
proj1 = make_projector(first_projection_angle,  idx=0)
proj2 = make_projector(second_projection_angle, idx=1)

print("[INFO] Forward projecting...")
p1 = proj1.forward_project(vol_t)   # (1,1,1,512,512)
p2 = proj2.forward_project(vol_t)

p1 = p1.squeeze(2)  # remove depth dimension -> (1,1,512,512)
p2 = p2.squeeze(2)

projections = torch.cat([p1, p2], dim=1).cpu().numpy()  # (1,2,512,512)

# Normalize
projections -= projections.min()
projections /= (projections.max() + 1e-8)

# Save
os.makedirs(os.path.dirname(OUTPUT_NPY), exist_ok=True)
np.save(OUTPUT_NPY, projections)
print("[DONE] Saved projections:", projections.shape)

