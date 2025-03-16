import pyarrow.parquet as pq
import pandas as pd
from src.common import NUM_PIXELS

BATCH_SIZE = 2**16  # = 65_536

def load_dataset_and_flatten_images(
    parquet_path,
    images_to_load: list[str],
    features_to_load: list[str],
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(parquet_path, memory_map=True)

    data = []
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        batch = batch.to_pandas()

        image_dfs = []
        for column_name in images_to_load:
            image = batch[column_name].apply(pd.Series).iloc[:, :NUM_PIXELS]
            image.columns = [f"{column_name}_{i}" for i in range(image.shape[1])]
            image_dfs.append(image)

        data.append(pd.concat([batch[features_to_load]] + image_dfs, axis=1))

    return pd.concat(data)
