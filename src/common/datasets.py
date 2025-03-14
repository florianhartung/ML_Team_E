from typing import Iterator
import os

import pyarrow.parquet as pq
import pandas as pd
from src.common import PARAMS_HILLAS, PARAMS_STEREO, PARAMS_TRUE_SHOWER, NUM_PIXELS

BATCH_SIZE = 2**16  # = 65_536
DATA_GAMMAS_PATH = "../../data/magic-gammas-new-4.parquet"
DATA_PROTONS_PATH = "../../data/magic-protons.parquet"

PARAMETERS_TO_LOAD = PARAMS_HILLAS + PARAMS_STEREO + PARAMS_TRUE_SHOWER
IMAGES_TO_LOAD = ["clean_image_m1", "clean_image_m2"]


def read_gammas() -> pd.DataFrame:
    return read_dataset_and_flatten_images(
        DATA_GAMMAS_PATH, PARAMETERS_TO_LOAD, IMAGES_TO_LOAD
    )


def read_protons() -> pd.DataFrame:
    return read_dataset_and_flatten_images(
        DATA_PROTONS_PATH, PARAMETERS_TO_LOAD, IMAGES_TO_LOAD
    )


def read_dataset_and_flatten_images(
    parquet_file: str, parameter_columns: list[str], image_columns: list[str]
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(DATA_PROTONS_PATH, memory_map=True)

    data = []
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        batch = batch.to_pandas()

        image_dfs = []
        for column_name in image_columns:
            image = batch[column_name].apply(pd.Series).iloc[:, :NUM_PIXELS]
            image.columns = [f"{column_name}_{i}" for i in range(image.shape[1])]
            image_dfs.append(image)

        data.append(pd.concat([batch[parameter_columns]] + image_dfs, axis=1))

    return pd.concat(data)
