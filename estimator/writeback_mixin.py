from base.timer import Timer


class WritebackMixin:
    """Writeback template class.

    This class is used to write back the results of the estimators to the
    data source.
    Implementations of this class shall override the :meth:`writeback` method. 
    """

    def writeback(self, new_col: str = None) -> int:
        """Write back the results to the data source.

        Parameters
        ----------
        new_col : str, optional
          the name of the new collection to write the results to. If not provided, 
          the implementation shall write back to the original column with -e.

        Returns
        -------
        the number of records written back.
        """
        if new_col is None:
            new_col = self.y_column + '-e'
        timer = Timer(new_col, self.logger)
        timer.start()
        counter = 0
        for scale in self.scales.values():
            df = self.load_data(scale=scale)
            # fill missing values with mean
            df.fillna(df.mean())
            x_cols = scale.meta['x_cols']
            y = scale.meta['model'].predict(df[x_cols])
            df[new_col] = y
            self.data_source.writeback(
                col=new_col,
                y=df[new_col],
            )
            counter += len(y)
        timer.stop(counter)
        return counter
