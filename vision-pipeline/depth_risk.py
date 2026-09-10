"""
Physics-grounded risk estimate: turns a pixel-space drop into an impact-energy
number instead of a black-box "risky" label.

Calibration note: MiDaS gives *relative* depth, not metres. Rather than
fighting monocular metric calibration during a 4-day build, calibrate once per
camera angle: measure one known carton's real height (m) and its height in
pixels in a representative frame, set METERS_PER_PIXEL below. This is an
approximation, not lab-grade measurement — say so explicitly in the demo/pitch,
it is exactly the "responsible AI / don't overclaim precision" point the brief
itself asks for.
"""
from dataclasses import dataclass

G = 9.81  # m/s^2
DEFAULT_PACKAGE_MASS_KG = 5.0


@dataclass
class ImpactEstimate:
    drop_height_m: float
    impact_velocity_mps: float
    impact_energy_j: float


def estimate_impact(pixel_drop: float, meters_per_pixel: float, mass_kg: float = DEFAULT_PACKAGE_MASS_KG) -> ImpactEstimate:
    """pixel_drop: vertical centroid displacement in pixels during the drop event (positive = downward)."""
    drop_height_m = max(pixel_drop, 0.0) * meters_per_pixel
    impact_velocity_mps = (2 * G * drop_height_m) ** 0.5
    impact_energy_j = 0.5 * mass_kg * impact_velocity_mps ** 2
    return ImpactEstimate(round(drop_height_m, 3), round(impact_velocity_mps, 3), round(impact_energy_j, 3))


def calibrate_meters_per_pixel(known_object_height_m: float, known_object_height_px: float) -> float:
    return known_object_height_m / known_object_height_px


def load_midas(model_type: str = "MiDaS_small"):
    """Optional: relative depth map, useful for a nicer demo visual (heatmap
    overlay) or for a smarter meters-per-pixel-at-this-depth estimate later.
    Not required for the impact-energy calc above to work."""
    import torch

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform if "small" in model_type.lower() else transforms.default_transform
    return midas, transform


def get_relative_depth_map(midas, transform, frame_rgb):
    import torch

    input_batch = transform(frame_rgb)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()
