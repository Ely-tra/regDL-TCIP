import argparse
import os

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process idealized WRF NetCDF files into X/Z NumPy arrays (CMIP6-style step2)"
    )
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        default="/N/slate/kmluong/PROJECT2/WRF/wrf_data",
        help="Directory containing input .nc files",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="/N/slate/kmluong/PROJECT2/WRF/wrf_data_lv2",
        help="Directory where output .npy files will be saved",
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        default=5,
        help="Number of consecutive frames to include",
    )
    parser.add_argument(
        "-vl",
        "--var_levels",
        type=str,
        nargs="+",
        default=[
            "U10m",
            "V10m",
            "SST",
            "LANDMASK",
            "U14",
            "V14",
            "U03",
            "V03",
            "T12",
            "QVAPOR05",
            "PHB05",
        ],
        help=(
            "List of variable-level codes. "
            "Format e.g. U01, V02, QVAPOR03 or PSFC for surface."
        ),
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="wrf_idealized_track_{frames}_dataset",
        help="Output filename prefix; may include {frames}",
    )
    return parser.parse_args()


def var_extract(ds: xr.Dataset, var_levels=None, frames: int = 2):
    if var_levels is None:
        var_levels = [
            ("U", 1),
            ("U", 2),
            ("U", 3),
            ("V", 1),
            ("V", 2),
            ("V", 3),
            ("T", 1),
            ("T", 2),
            ("T", 3),
            ("QVAPOR", 1),
            ("QVAPOR", 2),
            ("QVAPOR", 3),
            ("PSFC", None),
        ]

    t = ds["Time"]
    doy = t.dt.dayofyear.values
    is_leap = t.dt.is_leap_year.values
    yearlen = np.where(is_leap, 366, 365)
    theta = 2 * np.pi * doy / yearlen
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)

    lon_arr = ds["cen_lon"].values
    lat_arr = ds["cen_lat"].values
    year_arr = t.dt.year.values
    file_year = int(year_arr[0])

    var_das, var_names = [], []
    for name, lvl in var_levels:
        da = ds[name]
        if lvl is not None:
            lvl_dim = next(d for d in da.dims if "bottom_top" in d)
            if name == "PHB":
                da = da.isel({lvl_dim: lvl}) + ds["PH"].isel({lvl_dim: lvl})
            elif name == "PH":
                da = da.isel({lvl_dim: lvl}) + ds["PHB"].isel({lvl_dim: lvl})
            else:
                da = da.isel({lvl_dim: lvl})
        da = da.reset_coords(drop=True)
        var_das.append(da)
        var_names.append(f"{name}_{lvl}" if lvl is not None else name)

    n_time = ds.sizes["Time"]
    bases = np.arange(0, n_time, 4)
    x_list, z_list = [], []

    for base in bases:
        idx_hist = base + np.arange(frames)
        if idx_hist.max() >= n_time:
            continue

        hist_vars = []
        for da in var_das:
            h = da.isel(Time=idx_hist).rename({"Time": "frame"})
            hist_vars.append(h)
        sample_x = xr.concat(hist_vars, dim="var", coords="minimal", compat="override")
        sample_x = sample_x.assign_coords(var=var_names)
        sample_x = sample_x.transpose("frame", "var", "y", "x")
        sample_x = sample_x.assign_coords(frame=np.arange(sample_x.sizes["frame"]))
        x_list.append(sample_x.expand_dims({"sample": [base]}))

        zarr = np.array([lon_arr[base], lat_arr[base], sin_theta[base], cos_theta[base]])
        sample_z = xr.DataArray(
            zarr,
            dims=("feature",),
            coords={"feature": ["lon", "lat", "sin", "cos"]},
        )
        z_list.append(sample_z.expand_dims({"sample": [base]}))

    x = xr.concat(x_list, dim="sample").values
    x = np.transpose(x, (0, 1, 3, 4, 2))
    z = xr.concat(z_list, dim="sample").values
    return x, z, file_year


def process_data(indir, outdir, var_levels=None, frames: int = 2, prefix: str = "wrf_idealized_track_{frames}_dataset"):
    os.makedirs(outdir, exist_ok=True)
    x_list, z_list, exp_id_list = [], [], []

    for fname in os.listdir(indir):
        if not fname.endswith(".nc"):
            continue
        fpath = os.path.join(indir, fname)
        with xr.open_dataset(fpath) as ds:
            n_time = int(ds.sizes["Time"])
            if n_time < frames:
                print(f"Skipping {fname}: Time len {n_time} < frames {frames}")
                continue
            x, z, _ = var_extract(ds, var_levels, frames)
        x_list.append(x)
        z_list.append(z)
        sample_exp_id = os.path.splitext(fname)[0]
        exp_id_dtype = f"<U{max(1, len(sample_exp_id))}"
        exp_id_list.append(np.full(x.shape[0], sample_exp_id, dtype=exp_id_dtype))

    if not x_list:
        raise RuntimeError(f"No valid .nc files found in {indir}")

    x_all = np.concatenate(x_list, axis=0)
    z_all = np.concatenate(z_list, axis=0)
    exp_ids_all = np.concatenate(exp_id_list, axis=0).astype(str)

    out_prefix = prefix.format(frames=frames)
    np.save(os.path.join(outdir, f"{out_prefix}_X.npy"), x_all)
    np.save(os.path.join(outdir, f"{out_prefix}_Z.npy"), z_all)
    np.save(os.path.join(outdir, f"{out_prefix}_exp_ids.npy"), exp_ids_all)


if __name__ == "__main__":
    args = parse_args()
    var_levels = [
        (v[:-2], int(v[-2:])) if v[-2:].isdigit() else ((v[:-1], None) if v[-1:] == "m" else (v, None))
        for v in args.var_levels
    ]
    process_data(
        indir=args.indir,
        outdir=args.outdir,
        var_levels=var_levels,
        frames=args.frames,
        prefix=args.prefix,
    )
