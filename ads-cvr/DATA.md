# Criteo Sponsored Search Data

The project uses the official Criteo Sponsored Search Conversion Log Dataset:

- Description and terms: <https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/>
- Download: <https://criteostorage.blob.core.windows.net/criteo-research-datasets/Criteo_Conversion_Search.tar.gz>

The Criteo page identifies the dataset license as CC BY-NC-SA 4.0. Review the
current terms before use. The raw data is not redistributed by this repository.

Extract the archive and place or link `CriteoSearchData` at:

```text
ads-cvr/data/criteo-sponsored-search/data.tsv
```

The 2026-09-20 experiment used:

- rows: `15,995,634`
- columns: `23`, tab-separated
- bytes: `6,426,808,162`
- SHA-256: `a2ee46feec6008f901ab1ce5e51a44b591b5194285ef86a141b5cc54dbde0567`

Each row is an ad click. `Sale` is the post-click conversion target;
`SalesAmountInEuro` and `time_delay_for_conversion` are not prediction inputs.

## Product-price audit

Although the official description lists `product_price` as the advertised
product price, the released file has a near-deterministic empirical relation
between this field and `Sale`. On the chronological test split, 90.85% of rows
have price 0, every row with a non-zero price is positive, and no negative row
has a non-zero price. Primary `clean` experiments therefore exclude
`product_price`; it is retained only for leakage diagnostics.
