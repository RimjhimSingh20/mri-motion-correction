"""Classical motion-correction baselines: FSL FLIRT and SimpleITK rigid registration."""

import subprocess
from typing import Optional, Tuple

import numpy as np


class FSLFlirtRegistration:
    """
    Rigid-body registration using FSL FLIRT (6 DOF by default).

    Requires FSL installed and ``flirt`` on PATH.
    """

    def __init__(
        self,
        dof: int = 6,
        cost: str = "normcorr",
        interp: str = "spline",
    ):
        self.dof = dof
        self.cost = cost
        self.interp = interp

    def _check_fsl(self):
        result = subprocess.run(["which", "flirt"], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("flirt not found — install FSL and ensure it is on PATH.")

    def register(
        self,
        moving_path: str,
        reference_path: str,
        output_path: str,
        matrix_path: Optional[str] = None,
    ) -> str:
        self._check_fsl()
        cmd = [
            "flirt",
            "-in", moving_path,
            "-ref", reference_path,
            "-out", output_path,
            "-dof", str(self.dof),
            "-cost", self.cost,
            "-interp", self.interp,
        ]
        if matrix_path:
            cmd += ["-omat", matrix_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FLIRT failed:\n{result.stderr}")
        return output_path


class SimpleITKRegistration:
    """
    Rigid-body registration using SimpleITK (Mattes MI metric, gradient descent).

    Requires: ``pip install SimpleITK``
    """

    def register(
        self,
        moving: np.ndarray,
        fixed: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, object]:
        """
        Args:
            moving:  degraded volume (D, H, W) float32
            fixed:   reference / ground-truth volume (D, H, W) float32
            spacing: voxel spacing in mm

        Returns:
            (corrected volume, SimpleITK transform)
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK required: pip install SimpleITK")

        fixed_itk = sitk.GetImageFromArray(fixed.astype(np.float32))
        moving_itk = sitk.GetImageFromArray(moving.astype(np.float32))
        fixed_itk.SetSpacing(spacing)
        moving_itk.SetSpacing(spacing)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInitialTransform(
            sitk.CenteredTransformInitializer(
                fixed_itk,
                moving_itk,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        )
        reg.SetInterpolator(sitk.sitkLinear)

        transform = reg.Execute(fixed_itk, moving_itk)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_itk)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetTransform(transform)
        corrected = sitk.GetArrayFromImage(resampler.Execute(moving_itk))

        return corrected.astype(np.float32), transform
