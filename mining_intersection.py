"""GIS post-processing for Camalan mining-pond change predictions.

Inputs: one full-region binary prediction mask and its downloaded region ID.
Outputs: one GeoDataFrame of predicted change polygons and their overlaps.
This module does not train a model or determine whether mining is illegal.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def analyze_prediction(prediction, region_id, project_dir="."):
    """Compare one Camalan prediction mask with protected areas and mining records.

    prediction: 2D NumPy array, original region dimensions, values 0 or 1.
        1 = predicted mining-pond change; 0 = no predicted change.
    region_id: Downloaded Camalan folder ID (not the published paper region ID).
    project_dir: Folder containing georeferenced_masks and environmental_data.

    Returns a GeoDataFrame in WGS 84 / UTM zone 19S (EPSG:32719).
    All area columns are in hectares. Boolean intersections include boundary
    touches; use the overlap-area columns to identify positive-area overlaps.
    """

    # 1. Identify the input files in our shared project folder.
    project_dir = Path(project_dir)
    reference_raster_path = (
        project_dir / "georeferenced_masks"
        / f"region_{region_id}_change_mask.tif"
    )
    environmental_dir = project_dir / "environmental_data"
    protected_areas_path = environmental_dir / "peru_protected_areas.gpkg"
    mining_cadastre_path = environmental_dir / "CATASTROMINERO_Clip.shp"

    # 2. Check the prediction and recover its geographic location.
    # Read ONLY the reference raster's metadata, not its ground-truth labels.
    prediction = np.asarray(prediction)
    with rasterio.open(reference_raster_path) as reference_raster:
        expected_shape = (reference_raster.height, reference_raster.width)
        raster_transform = reference_raster.transform
        raster_crs = reference_raster.crs
        raster_bounds = reference_raster.bounds

    # Do not stretch an individual 128 x 128 patch over an entire region.
    if prediction.shape != expected_shape:
        raise ValueError(
            f"Region {region_id} requires a full-region mask of shape "
            f"{expected_shape}, but received {prediction.shape}."
        )
    if not np.isin(prediction, [0, 1]).all():
        raise ValueError("Prediction must contain only binary values 0 and 1.")
    if raster_crs is None:
        raise ValueError("The reference raster is missing its CRS.")
    prediction = prediction.astype(np.uint8)

    # 3. Convert connected positive pixels into geographic polygons.
    # Four-neighbor connectivity means pixels sharing only a corner are separate.
    # The mask argument excludes the background (0) from polygonization.
    predicted_polygons = [
        shape(polygon_geometry)
        for polygon_geometry, pixel_value in shapes(
            prediction, mask=(prediction == 1),
            transform=raster_transform, connectivity=4
        )
        if pixel_value == 1
    ]

    # Keep region and polygon IDs so each result can be traced to its source.
    # This also supports a valid, empty result when the model predicts no change.
    mining_changes = gpd.GeoDataFrame(
        {"region_id": np.full(len(predicted_polygons), region_id, dtype=int),
         "polygon_id": np.arange(1, len(predicted_polygons) + 1)},
        geometry=predicted_polygons, crs=raster_crs
    )
    mining_changes["geometry"] = mining_changes.geometry.make_valid()

    # 4. Read the two independent geographic boundary datasets.
    protected_areas = gpd.read_file(protected_areas_path)
    mining_cadastre = gpd.read_file(mining_cadastre_path)

    # Keep the titled records separate from pending, blocked, and other records.
    # TITULADO is the actual status used in the supplied INGEMMET shapefile.
    concession_status = mining_cadastre["LEYENDA"].str.strip().str.upper()
    titled_concessions = mining_cadastre[concession_status == "TITULADO"].copy()

    # 5. Select boundary features near this region, then project into meters.
    # This is just a spatial prefilter, not a change to the source datasets.
    # UTM zone 19 South is appropriate for our Madre de Dios study regions.
    area_crs = "EPSG:32719"
    west, south, east, north = raster_bounds
    mining_changes = mining_changes.to_crs(area_crs)

    boundary_layers = {
        "protected": protected_areas,
        "cadastre": mining_cadastre,
        "titled": titled_concessions,
    }

    # 6. Intersect predicted polygons with each boundary dataset.
    # First merge the nearby boundaries to avoid double-counting overlap where
    # two protected areas or two mining records overlap one another.
    mining_changes["area_ha"] = mining_changes.geometry.area / 10_000

    for boundary_name, boundary_layer in boundary_layers.items():
        boundary_layer = boundary_layer.to_crs(raster_crs)
        nearby_boundaries = boundary_layer.cx[west:east, south:north]
        nearby_boundaries = nearby_boundaries.to_crs(area_crs)
        boundary_union = nearby_boundaries.geometry.make_valid().union_all()

        # True/False: does the predicted change touch or overlap this layer?
        mining_changes[f"intersects_{boundary_name}"] = (
            mining_changes.geometry.intersects(boundary_union)
        )

        # How much of the predicted change actually overlaps the boundary layer?
        # Divide square meters by 10,000 to convert to hectares.
        mining_changes[f"{boundary_name}_overlap_ha"] = (
            mining_changes.geometry.intersection(boundary_union).area / 10_000
        )

    # 7. Calculate the portion outside mapped titled concession records.
    # A polygon can be partly inside and partly outside a concession.
    # This is NOT a determination of illegal mining or permit validity.
    mining_changes["outside_titled_ha"] = (
        mining_changes["area_ha"] - mining_changes["titled_overlap_ha"]
    ).clip(lower=0)

    # 8. Return the polygons and attributes for further analysis or GIS export.
    # We do not alter either source layer or the input prediction mask.
    return mining_changes


# EXAMPLE FOR MODEL NOTEBOOKS (uncomment and replace the placeholders):
# results = analyze_prediction(prediction=YOUR_PREDICTED_MASK, region_id=YOUR_REGION_ID)
# print(results.head())
# results.to_file("model_intersection_results.gpkg", driver="GPKG")
