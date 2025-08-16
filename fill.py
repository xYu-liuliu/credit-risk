import polars as pl
from polars import Int64, Float64, UInt64, Utf8

class Filler:
    def fill_by_suffix(df: pl.DataFrame) -> pl.DataFrame:
        fill_exprs = []
        numeric_types = {Int64, Float64, UInt64}
        schema = df.schema                  

        for col, dtype in schema.items():
            if "_" not in col:
                continue                   

            suffix = col.split("_")[-1][-1] 

            # -------- numeric 0‑fill --------
            if suffix in {"A", "P"} and dtype in numeric_types:
                fill_exprs.append(pl.col(col).fill_null(0))

            # -------- M: categorical --------
            elif suffix == "M":
                if "zip" in col.lower() and dtype in numeric_types:
                    fill_exprs.append(pl.col(col).fill_null(-1))   # zip code numeric
                elif dtype == Utf8:
                    fill_exprs.append(pl.col(col).fill_null("missing"))

        return df.with_columns(fill_exprs)

