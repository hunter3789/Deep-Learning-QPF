"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import h5netcdf

from load_dataset_regressor_predict import load_data
from models_regressor import load_model

def parse_case(case_str):
    return datetime.strptime(case_str, "%Y%m%d%H")

def visualize(
    case,
    epoch,
    model_name: str = "detector",
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    model = load_model(model_name, epoch, with_weights=True)
    model = model.to(device)
    #model.train()

    #mean_std_file = '../dataset/trainset_agg_mean_std.npz'
    mean_std_file = '../dataset/trainset_ssrd_agg_mean_std.npz'
    mean, std = np.load(mean_std_file)['mean'], np.load(mean_std_file)['std']

    topo_file = '../dataset/topo_norm.npz'
    topo = np.load(topo_file)['topo']

    label_mean_std_file = '../dataset/trainset_label_agg_mean_std.npz'
    label_mean, label_std = np.load(label_mean_std_file)['mean'], np.load(label_mean_std_file)['std']

    #val_data = load_data(mean=mean, std=std, topo=topo, case=case, dataset_path="../dataset/input/{case:%Y%m}/{case:%d}".format(case=case), shuffle=False, batch_size=1, num_workers=0, transform_pipeline="default")
    val_data = load_data(mean=mean, std=std, topo=topo, case=case, dataset_path="../dataset/input_ssrd/{case:%Y%m}/{case:%d}".format(case=case), shuffle=False, batch_size=1, num_workers=0, transform_pipeline="default")


    latlon_file = '../dataset/latlon.npz'
    lat, lon = np.load(latlon_file)['lat'], np.load(latlon_file)['lon']

    map_NX, map_NY, map_SX, map_SY = 576, 720, 560/2, 840/2

    cx, cy = map_SX, map_SY
    projy = np.array([2*(y-map_SY) for y in range(lat.shape[0])])
    projx = np.array([2*(x-map_SX) for x in range(lat.shape[1])])
    #print(projy.shape)
    #print(projx.shape)
    margin = 20

    tp = np.zeros_like(lat)

    model.eval()

    with torch.inference_mode():
        for data in val_data:
            #if case != args_case:
            #    continue

            img, label, mask, fhr = data['image'], data['label'], data['mask'], data['fhr']
            img = img.to(device)

            pred  = model(img).squeeze()
            label = label.squeeze()
            mask  = mask.squeeze()

            pred  = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            mask  = mask.detach().cpu().numpy()

            pred = pred*label_std + label_mean
            pred = np.power(10, pred) - 0.1
            pred = np.where(pred < 0, 0, pred)

            tp = tp + pred

            fhr = fhr.detach().cpu().numpy()[0]
            #print(fhr)

            #print(pred.shape)
            #np.savez_compressed('./output/{case:%Y%m}/{case:%d}/pred.regress.{case:%Y%m%d%H}00.f{fhr:03d}'.format(case=case, fhr=fhr), pred=pred, label=label, mask=mask)

            # Create a new NetCDF file
            nc_file = './output/{case:%Y%m}/{case:%d}/pred.regress.{case:%Y%m%d%H}00.f{fhr:03d}.nc'.format(case=case, fhr=fhr)
            with h5netcdf.File(nc_file, "w") as f:
                # Create dimensions
                f.dimensions["y"] = pred.shape[0] - margin*2
                f.dimensions["x"] = pred.shape[1] - margin*2
                f.dimensions["T"] = 1

                # Add **global attributes** (metadata for the file)
                f.attrs["Conventions"] = np.array("CF-1.6", dtype="S")

                # Create a variable
                var = f.create_variable(
                        "lat",
                        dtype=float,
                        dimensions=("y", "x"),
                        compression="gzip",  # Enables compression (zlib/GZIP)
                        compression_opts=5,  # Compression level (1-9, higher = more compression)
                        shuffle=True,  # Enables shuffle filter to improve compression efficiency
                    )                
                # Assign data
                var[:] = lat[margin:-margin,margin:-margin]

                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["units"] = np.array("degrees_north", dtype="S")
                var.attrs["long_name"] = np.array("latitude", dtype="S")
                var.attrs["standard_name"] = np.array("latitude", dtype="S")
                var.attrs["_FillValue"] = -999.0
                var.attrs["coordinates"] = np.array("lat lon", dtype="S")
                var.attrs["grid_mapping"] = np.array("grid_mapping", dtype="S")

                # Create a variable
                var = f.create_variable(
                        "lon",
                        dtype=float,
                        dimensions=("y", "x"),
                        compression="gzip",  # Enables compression (zlib/GZIP)
                        compression_opts=5,  # Compression level (1-9, higher = more compression)
                        shuffle=True,  # Enables shuffle filter to improve compression efficiency
                    )                
                # Assign data
                var[:] = lon[margin:-margin,margin:-margin]

                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["units"] = np.array("degrees_east", dtype="S")
                var.attrs["long_name"] = np.array("longitude", dtype="S")
                var.attrs["standard_name"] = np.array("longitude", dtype="S")
                var.attrs["_FillValue"] = -999.0
                var.attrs["coordinates"] = np.array("lat lon", dtype="S")
                var.attrs["grid_mapping"] = np.array("grid_mapping", dtype="S")

                # Create a variable
                var = f.create_variable(
                        "tp",
                        dtype=float,
                        dimensions=("y", "x"),
                        compression="gzip",  # Enables compression (zlib/GZIP)
                        compression_opts=5,  # Compression level (1-9, higher = more compression)
                        shuffle=True,  # Enables shuffle filter to improve compression efficiency
                    )                
                # Assign data
                var[:] = tp[margin:-margin,margin:-margin]

                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["units"] = np.array("mm", dtype="S")
                var.attrs["long_name"] = np.array("Total precipitation", dtype="S")
                var.attrs["standard_name"] = np.array("tp", dtype="S")
                var.attrs["_FillValue"] = -999.0
                var.attrs["lengthOfTimeRange"] = fhr
                var.attrs["coordinates"] = np.array("lat lon", dtype="S")
                var.attrs["grid_mapping"] = np.array("grid_mapping", dtype="S")

                # Create a variable
                var = f.create_variable(
                        "time",
                        dtype=int,
                        dimensions=("T"),
                    )                
                # Assign data
                var[:] = (case + timedelta(hours=int(fhr)) - datetime(1970, 1, 1, 0, 0)).total_seconds() / 3600

                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["units"] = np.array("hours since 1970-01-01 00:00:00 0:00", dtype="S")
                var.attrs["long_name"] = np.array("time", dtype="S")
                var.attrs["standard_name"] = np.array("time", dtype="S")
                var.attrs["axis"] = np.array("T", dtype="S")

                # Create a variable
                var = f.create_variable(
                        "forecast_reference_time",
                        dtype=int,
                        dimensions=("T"),
                    )                
                # Assign data
                var[:] = (case - datetime(1970, 1, 1, 0, 0)).total_seconds() / 3600

                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["units"] = np.array("hours since 1970-01-01 00:00:00 0:00", dtype="S")
                var.attrs["long_name"] = np.array("forecast_reference_time", dtype="S")
                var.attrs["standard_name"] = np.array("forecast_reference_time", dtype="S")

                # Create a variable
                var = f.create_variable(
                        "grid_mapping",
                        dtype=int,
                        data=0,
                    )                
     
                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["grid_mapping_name"] = np.array("lambert_conformal_conic", dtype="S")
                var.attrs["standard_parallel"] = np.array([30.,60.])
                var.attrs["longitude_of_central_meridian"] = float(126.)
                var.attrs["latitude_of_projection_origin"] = float(38.)
                var.attrs["false_easting"] = int(0)
                var.attrs["false_northing"] = int(0)
                var.attrs["GRIB_earth_shape"] = np.array("spherical", dtype="S")
                var.attrs["GRIB_earth_shape_code"] = int(6)

                # Create a variable
                var = f.create_variable(
                        "x",
                        dtype=float,
                        dimensions=("x"),
                        compression="gzip",  # Enables compression (zlib/GZIP)
                        compression_opts=5,  # Compression level (1-9, higher = more compression)
                        shuffle=True,  # Enables shuffle filter to improve compression efficiency
                    )                
                # Assign data
                var[:] = projx[margin:-margin]

                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["units"] = np.array("km", dtype="S")
                var.attrs["long_name"] = np.array("x coordinate of projection", dtype="S")
                var.attrs["standard_name"] = np.array("projection_x_coordinate", dtype="S")
                var.attrs["grid_spacing"] = np.array("2 km", dtype="S")
                var.attrs["axis"] = np.array("X", dtype="S")

                # Create a variable
                var = f.create_variable(
                        "y",
                        dtype=float,
                        dimensions=("y"),
                        compression="gzip",  # Enables compression (zlib/GZIP)
                        compression_opts=5,  # Compression level (1-9, higher = more compression)
                        shuffle=True,  # Enables shuffle filter to improve compression efficiency
                    )                
                # Assign data
                var[:] = projy[margin:-margin]

                # Add **variable attributes** (metadata for a specific variable)
                var.attrs["units"] = np.array("km", dtype="S")
                var.attrs["long_name"] = np.array("y coordinate of projection", dtype="S")
                var.attrs["standard_name"] = np.array("projection_y_coordinate", dtype="S")
                var.attrs["grid_spacing"] = np.array("2 km", dtype="S")
                var.attrs["axis"] = np.array("Y", dtype="S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="regressor")
    parser.add_argument("--case", type=parse_case)
    parser.add_argument('--epoch', dest='epoch', type=int, default=0)

    # pass all arguments to train
    visualize(**vars(parser.parse_args()))
