### Rewrite ideas

- DF IR for pandas / polars / other df APIs
    - selections, filters, aggregations, joins, etc.
- operator fusion for pandas:
```python
cols = [...]
df[cols] = df.groupby("id")[cols].transform(some_udf)
df_agg = df.groupby("id").tail(1)
```