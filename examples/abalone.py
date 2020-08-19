# %%
import pandas as pd
from tensorflow.keras.layers import DenseFeatures
from tftabular.models import TabNetRegressor
from tftabular.utils import df_to_dataset, get_feature

# %%
names = ['Sex', 'Length', 'Diameter', 'Height', 'While weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
                   names = names,
                   dtype = {k: 'float32' for k in names if k != 'Sex'})


# %%
def get_labels(x: pd.Series) -> pd.Series:
    """
    Converts strings to unqiue ints for use in Pytorch Embedding
    """
    labels, levels = pd.factorize(x)
    return pd.Series(labels, name=x.name, index=x.index)

X, y = (data
        .drop(columns='Rings')
        .assign(Sex = lambda df: df.Sex
                                   .pipe(get_labels)
                                   .add(1)
                                   .astype('int32'))
        .rename(columns=lambda s: '_'.join(s.lower()
                                            .split())),
        data.Rings)
split = data.shape[0] // 2
X_train, X_valid, y_train, y_valid = X.iloc[:split, :], X.iloc[-split:, :], y.iloc[:split], y.iloc[-split:]
train, valid = df_to_dataset(X_train, y_train), df_to_dataset(X_valid, y_valid)

# %%
columns = [get_feature(f) for k, f in X_train.iteritems()]
feature_column = DenseFeatures(columns, trainable=True)

model = TabNetRegressor(feature_column=feature_column,
                        virtual_batch_size=None)
model.compile('adam', 'mse')
model.fit(train, epochs=5)

# %%
prediction = model.predict(valid)
explanations = model.explain(dict(X_valid))
features = model.transform(dict(X_valid))