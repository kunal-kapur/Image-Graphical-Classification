from dataloader import AnimalsDatasetImage
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import torch
import os


def animals_parquet(dist=15, nodes=20, classified=False, data_path="Animals"):
    data = AnimalsDatasetImage("Animals", distance=15)

    names = ['y', 'x'] + [f"feature_{i}" for i in range(128)]

    # Initialize PyArrow table writer
    parquet_file = f'animals_d{dist}_nodes{nodes}_classified{classified}.parquet'

    if os.path.exists(parquet_file):
        print("CHECK DATA IF DATA CAN BE DELTED")
        raise ValueError
    writer = None

    for i, sample in tqdm(enumerate(data)):
        loc, description, label, path = sample

        if loc is None:
            continue
        AMOUNT = nodes

        loc = loc[0:AMOUNT]
        description = description[0:AMOUNT]
        combined = torch.concat((loc, description), dim=1).detach().numpy()

        paths = [path] * combined.shape[0]
        labels = [data.mapping[label]] * combined.shape[0]
        
        # Create DataFrame for the current batch
        df = pd.DataFrame(data=combined, columns=names)
        df['label'] = labels
        df['path'] = paths

        # Convert DataFrame to PyArrow table and write in batches
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, table.schema)
        writer.write_table(table)

    # Close the writer
    if writer:
        writer.close()

    print(f"Data written to {parquet_file} in chunks.")
