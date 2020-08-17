from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf


def get_feature(
    x: pd.DataFrame, dimension=1
) -> Union[
    tf.python.feature_column.NumericColumn, tf.python.feature_column.EmbeddingColumn
]:
    if x.dtype == np.float32:
        return tf.feature_column.numeric_column(x.name)
    else:
        return tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                x.name, num_buckets=x.max() + 1, default_value=0
            ),
            dimension=dimension,
        )


def df_to_dataset(
    X: pd.DataFrame, y: pd.Series, shuffle=False, batch_size=50000
) -> tf.python.data.ops.dataset_ops.TensorSliceDataset:
    ds = tf.data.Dataset.from_tensor_slices((dict(X.copy()), y.copy()))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ds
