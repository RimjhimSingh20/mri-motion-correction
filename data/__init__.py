from .motion_simulator import MotionSimulator, simulate_motion

# torch-dependent imports — only available when PyTorch is installed
try:
    from .dataset import MRIMotionDataset
    from .transforms import (
        RandomFlip3D,
        RandomRotation3D,
        SimulateMotion,
        Compose,
        build_train_transforms,
        build_val_transforms,
    )
except ImportError:
    pass

__all__ = [
    "MotionSimulator",
    "simulate_motion",
    "MRIMotionDataset",
    "RandomFlip3D",
    "RandomRotation3D",
    "SimulateMotion",
    "Compose",
    "build_train_transforms",
    "build_val_transforms",
]
