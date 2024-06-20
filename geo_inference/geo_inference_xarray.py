import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from tqdm import tqdm

import numpy as np
import dask
import rioxarray
import xarray as xr

from .config.logging_config import logger
from .geo_blocks import InferenceMerge, InferenceSampler, RasterDataset
from .utils.helpers import cmd_interface, get_device, get_directory, get_model
from .utils.polygon import gdf_to_yolo, mask_to_poly_geojson, geojson2coco

logger = logging.getLogger(__name__)


class GeoInference:
    """
    A class for performing geo inference on geospatial imagery using a pre-trained model.

    Args:
        model (str): The path or url to the model file
        work_dir (str): The directory where the model and output files will be saved.
        batch_size (int): The batch size to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        device (str): The device to use for inference (either "cpu" or "gpu").
        gpu_id (int): The ID of the GPU to use for inference (if device is "gpu").

    Attributes:
        batch_size (int): The batch size to use for inference.
        work_dir (Path): The directory where the model and output files will be saved.
        device (torch.device): The device to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        model (torch.jit.ScriptModule): The pre-trained model to use for inference.
        classes (int): The number of classes in the output of the model.

    """

    def __init__(self,
                 model: str = None,
                 work_dir: str = None,
                 batch_size: int = 1,
                 mask_to_vec: bool = False,
                 device: str = "gpu",
                 gpu_id: int = 0):
        self.gpu_id = int(gpu_id)
        self.batch_size = int(batch_size)
        self.work_dir: Path = get_directory(work_dir)
        self.device = get_device(device=device, 
                                 gpu_id=self.gpu_id)
        #model_path: Path = get_model(model_path_or_url=model, 
        #                             work_dir=self.work_dir)
        model_path = self.work_dir.joinpath(Path("model/4cls_RGB_5_1_2_3_scripted.pt"))
        self.mask_to_vec = mask_to_vec
        self.model = torch.jit.load(model_path, map_location=self.device)
        dummy_input = torch.ones((1, 3, 32, 32), device=self.device)
        with torch.no_grad():
            self.classes = self.model(dummy_input).shape[1]

    @torch.no_grad() 
    def __call__(self, tiff_image: str, bbox: str = None, patch_size: int = 512, stride_size: str = None) -> None:
        """
        Perform geo inference on geospatial imagery.

        Args:
            tiff_image (str): The path to the geospatial image to perform inference on.
            bbox (str): The bbox or extent of the image in this format "minx, miny, maxx, maxy"
            patch_size (int): The size of the patches to use for inference.
            stride_size (int): The stride to use between patches.

        Returns:
            None

        """
        mask_path = self.work_dir.joinpath(Path(tiff_image).stem + "_mask.tif")
        polygons_path = self.work_dir.joinpath(Path(tiff_image).stem + "_polygons.geojson")
        yolo_csv_path = self.work_dir.joinpath(Path(tiff_image).stem + "_yolo.csv")
        coco_json_path = self.work_dir.joinpath(Path(tiff_image).stem + "_coco.json")
        
        image_prefix = Path(tiff_image).parent
        r = rioxarray.open_rasterio(f"{image_prefix}/R.tif", chunks=patch_size)
        g = rioxarray.open_rasterio(f"{image_prefix}/G.tif", chunks=patch_size)
        b = rioxarray.open_rasterio(f"{image_prefix}/B.tif", chunks=patch_size)
        dataset = xr.concat([r,g,b], dim="band")
        #print(dataset)
        
        start_time = time.time()
        
        meta = np.array([[]], dtype="uint8")[:0]

        mask_array = dataset.data.map_overlap(
            copy_and_infer_chunked,
            meta=meta,
            drop_axis=0,
            model=self.model,
            name="predict",
            depth={1:patch_size, 2:patch_size}, 
            trim=True,
        )
        #print(mask_array)

        mask = xr.DataArray(
            mask_array,
            coords=dataset.drop_vars("band").coords,
            dims=("y", "x"),
        )
        mask.rio.to_raster(mask_path)

        if self.mask_to_vec:
            mask_to_poly_geojson(mask_path, polygons_path)
            gdf_to_yolo(polygons_path, mask_path, yolo_csv_path)
            geojson2coco(mask_path, polygons_path, coco_json_path)
            
        end_time = time.time() - start_time
        
        logger.info('Extraction Completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))

def infer_chip(data: torch.Tensor, model) -> torch.Tensor:
    # Input is GPU, output is GPU.
    with torch.no_grad():
        result = model(data).argmax(dim=1).to(torch.uint8)
    return result.to("cpu")

def copy_and_infer_chunked(tile, model, token=None):
    slices = dask.array.core.slices_from_chunks(dask.array.empty(tile.shape).chunks)
    out = np.empty(shape=tile.shape[1:], dtype="uint8")
    device = torch.device("cuda")

    for slice_ in slices:
        gpu_chip = torch.as_tensor(tile[slice_][np.newaxis, ...]).to(device)
        out[slice_[1:]] = infer_chip(gpu_chip, model).cpu().numpy()[0]
    return out

def main() -> None:
    arguments = cmd_interface()
    geo_inference = GeoInference(model=arguments["model"],
                                 work_dir=arguments["work_dir"],
                                 batch_size=arguments["batch_size"],
                                 mask_to_vec=arguments["vec"],
                                 device=arguments["device"],
                                 gpu_id=arguments["gpu_id"])
    geo_inference(tiff_image=arguments["image"], bbox=arguments["bbox"])
               
if __name__ == "__main__":
    main()