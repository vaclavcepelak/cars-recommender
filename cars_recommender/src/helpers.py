import warnings
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sklearn
from countryinfo import CountryInfo


def get_feature_names(column_transformer: Callable) -> List[str]:
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
            return []
        if trans == "passthrough":
            if hasattr(column_transformer, "_df_columns"):
                if (not isinstance(column, slice)) and all(
                    isinstance(col, str) for col in column
                ):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ["x%d" % i for i in indices[column]]
        if not hasattr(trans, "get_feature_names_out"):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn(
                "Transformer %s (type %s) does not "
                "provide get_feature_names. "
                "Will return input column names if available"
                % (str(name), type(trans).__name__)
            )
            # For transformers without a get_features_names method,
            # use the input names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names_out()]

    # Start of processing
    feature_names = []

    # Allow transformers to be pipelines.
    # Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [
            (name, trans, None, None)
            for step, name, trans in column_transformer._iter()
        ]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    return feature_names


def get_country_data(x: str) -> dict[str, Any]:
    """
    Get country data from the `countryinfo` library
    """
    country_replacements = {"North Macedonia": "Republic of Macedonia"}
    if x in country_replacements.keys():
        x = country_replacements[x]
    country = CountryInfo(x)
    return {
        "country": x,
        "region": country.region(),
        "capital_gps_lat": country.capital_latlng()[0],
        "capital_gps_lng": country.capital_latlng()[1],
    }


def plot_hist(df: pl.DataFrame, col: str, bins: int = 50) -> plt.hist:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Histogram of {col}: plain vs. log-scale")
    ax1.hist(df[col], bins=bins)
    ax2.hist(df[col].apply(np.log), bins=bins)
    return fig
