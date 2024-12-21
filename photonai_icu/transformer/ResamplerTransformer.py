from typing import Callable, Literal

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from photonai.photonlogger import logger
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)


class ResamplerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def __init__(
        self, 
        frequency: str = "1H",
        groupby: str = "stay_id",
        method: str | Callable | dict[str, str | Callable] = "mean",
        default_value: float | None = None,
        error: Literal["raise", "ignore"] = "raise",
    ):
        """Resampler Transformer for time series data.

        Parameters
        ----------
        frequency: str, default="1H"
            Frequency to resample the data to.
        groupby: str, default="stay_id"
            Column to group the patients by.
        method: str, default="mean"
            Method to use for resampling. Possible values are: 'mean', 'sum', 'max', 'min', 'std', 'median', 'first', 'last'.
        default: float, default=None
            Default value to use for missing values.
        error: str, default="raise"
            Error handling strategy. Possible values are: 'raise', 'ignore'.

        Examples
        --------
        Usage with PHOTONAI Now
        ```python
            import stuff

            hp = Hyperpipe()
            hp += PipelineElement("ResamplerTransformer")
        ```

        Notes
        -----
        This Transformer is used to resample time series data.
        """
        self.frequency = frequency
        self.groupby = groupby

        self.method = method 
        self.default_value = default_value
        self.error = error

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            logger.error("ResamplerTransformer: X is not a pandas DataFrame")
            if self.error == "raise":
                raise ValueError("X is not a pandas DataFrame")
            return X
        
        if not any(
            is_datetime64_any_dtype(X.index.get_level_values(ix_name))
            for ix_name in X.index.names
        ):
            logger.error("ResamplerTransformer: X does not have a datetime index")
            if self.error == "raise":
                raise ValueError("X does not have a datetime index")
            return X
                
        resampeld_x = (
            X.reset_index(self.groupby)
            .groupby(self.groupby)
            .resample(self.frequency)
            .agg(self.method)
            .drop(columns=self.groupby, errors="ignore")
        )

        for column in resampeld_x.columns:
            resampeld_x.loc[resampeld_x[column] == 0, column] = pd.NA

        if self.default_value is None:
            return resampeld_x
        
        return resampeld_x.fillna(self.default_value)
    
