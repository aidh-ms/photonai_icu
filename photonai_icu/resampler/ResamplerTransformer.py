import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from photonai.photonlogger import logger
from sklearn.base import BaseEstimator, TransformerMixin


class ResamplerTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self, 
        frequency: str | None = None,
        method: str | None = None,
        groupby: str | None = None,
        default_value: float | None = None        
    ):
        """Resampler Transformer for time series data.

        Parameters
        ----------
        frequency: str,default=None
            Frequency to resample the data to.
        method: str,default=None
            Method to use for resampling. Possible values are: 'mean', 'sum', 'max', 'min', 'std', 'median', 'first', 'last'.
        default: float,default=None
            Default value to use for missing values.

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
        self.frequency = frequency if frequency is not None else "1H"
        self.method = method if method is not None else "mean"
        self.groupby = groupby if groupby is not None else "stay_id"
        self.default_value = default_value if default_value is not None else 0.0

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            logger.error("ResamplerTransformer: X is not a pandas DataFrame")
            return X  # TODO: raise error
        
        if not is_datetime64_any_dtype(X.index):
            logger.error("ResamplerTransformer: X does not have a datetime index")
            return X  # TODO: raise error
        
        resampeld_x = (
            X.groupby(self.groupby)
            .resample(self.frequency)
            .agg(self.method)
        )

        if self.default_value is None:
            return resampeld_x
        
        return resampeld_x.fillna(self.default_value)

