from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


ZFeatureOrder = Literal["lon_lat", "lat_lon"]


@dataclass(frozen=True)
class LatLonCropResult:
    lat_crops: np.ndarray
    lon_crops: np.ndarray
    crop_indices: list[tuple[int, int, int, int]]
    center_indices: list[tuple[int, int]]
    used_lons: np.ndarray


def normalize_lon_for_grid(lon2d: np.ndarray, target_lon: float) -> float:
    """Convert target lon to match either [-180, 180] or [0, 360] grid convention."""
    lon_use = float(target_lon)
    lon_max = float(np.nanmax(lon2d))

    if lon_max > 180.0 and lon_use < 0.0:
        return lon_use % 360.0
    if lon_max <= 180.0 and lon_use > 180.0:
        return ((lon_use + 180.0) % 360.0) - 180.0
    return lon_use


def find_center_index(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int, float]:
    """Find the nearest grid point to a target storm center lat/lon."""
    lon_use = normalize_lon_for_grid(lon2d, target_lon)
    dist2 = (lat2d - float(target_lat)) ** 2 + (lon2d - lon_use) ** 2
    iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return int(iy), int(ix), lon_use


def crop_around_center(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    center_y: int,
    center_x: int,
    size: int = 100,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Extract a square lat/lon crop centered as closely as possible on a grid point."""
    if lat2d.shape != lon2d.shape:
        raise ValueError(
            f"lat2d and lon2d must have the same shape, got {lat2d.shape} and {lon2d.shape}"
        )
    if lat2d.ndim != 2:
        raise ValueError(f"lat2d and lon2d must be 2D arrays, got ndim={lat2d.ndim}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    height, width = lat2d.shape
    if size > height or size > width:
        raise ValueError(f"size={size} exceeds source grid shape {lat2d.shape}")

    half = size // 2
    y0 = max(0, min(int(center_y) - half, height - size))
    x0 = max(0, min(int(center_x) - half, width - size))
    y1 = y0 + size
    x1 = x0 + size

    return lat2d[y0:y1, x0:x1], lon2d[y0:y1, x0:x1], (y0, y1, x0, x1)


def extract_latlon_crops_for_track(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lon_centers: Iterable[float],
    lat_centers: Iterable[float],
    size: int = 100,
) -> LatLonCropResult:
    """Extract one lat/lon crop for each storm center point."""
    lon_arr = np.asarray(list(lon_centers), dtype=float).reshape(-1)
    lat_arr = np.asarray(list(lat_centers), dtype=float).reshape(-1)
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(
            "lon_centers and lat_centers must have the same length, "
            f"got {lon_arr.size} and {lat_arr.size}"
        )

    lat_crops = []
    lon_crops = []
    crop_indices: list[tuple[int, int, int, int]] = []
    center_indices: list[tuple[int, int]] = []
    used_lons = []

    for target_lon, target_lat in zip(lon_arr, lat_arr):
        iy, ix, lon_use = find_center_index(lat2d, lon2d, target_lat, target_lon)
        lat_crop, lon_crop, crop_idx = crop_around_center(lat2d, lon2d, iy, ix, size=size)

        lat_crops.append(lat_crop)
        lon_crops.append(lon_crop)
        crop_indices.append(crop_idx)
        center_indices.append((iy, ix))
        used_lons.append(lon_use)

    return LatLonCropResult(
        lat_crops=np.stack(lat_crops),
        lon_crops=np.stack(lon_crops),
        crop_indices=crop_indices,
        center_indices=center_indices,
        used_lons=np.asarray(used_lons, dtype=float),
    )


def extract_centers_from_z(
    z: np.ndarray,
    sample_idx: int,
    frame_indices: Iterable[int],
    feature_order: ZFeatureOrder = "lon_lat",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lon/lat center arrays from a Z array.

    Current step2 writers save Z as [lon, lat, sin_doy, cos_doy].
    Use feature_order="lat_lon" only for older arrays with reversed columns.
    """
    z_arr = np.asarray(z)
    if z_arr.ndim != 3 or z_arr.shape[-1] < 2:
        raise ValueError(f"Expected Z shape [sample, frame, feature>=2], got {z_arr.shape}")

    frame_idx = np.asarray(list(frame_indices), dtype=int)
    centers = z_arr[int(sample_idx), frame_idx, :]

    if feature_order == "lon_lat":
        return centers[:, 0], centers[:, 1]
    if feature_order == "lat_lon":
        return centers[:, 1], centers[:, 0]
    raise ValueError(f"Unknown feature_order: {feature_order!r}")


def extract_latlon_crops_from_z(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    z: np.ndarray,
    sample_idx: int,
    frame_indices: Iterable[int],
    size: int = 100,
    feature_order: ZFeatureOrder = "lon_lat",
) -> LatLonCropResult:
    """Extract storm-centered lat/lon crops using center coordinates from Z."""
    lon_centers, lat_centers = extract_centers_from_z(
        z=z,
        sample_idx=sample_idx,
        frame_indices=frame_indices,
        feature_order=feature_order,
    )
    return extract_latlon_crops_for_track(
        lat2d=lat2d,
        lon2d=lon2d,
        lon_centers=lon_centers,
        lat_centers=lat_centers,
        size=size,
    )


def get_extent_from_latlon(lat_crop: np.ndarray, lon_crop: np.ndarray) -> list[float]:
    """Return matplotlib imshow extent as [lon_min, lon_max, lat_min, lat_max]."""
    return [
        float(np.nanmin(lon_crop)),
        float(np.nanmax(lon_crop)),
        float(np.nanmin(lat_crop)),
        float(np.nanmax(lat_crop)),
    ]
