# rank-propogation-game

This repository provides an implementation of synchronous Footrule-median updates on random graphs.

## Quick start

Install requirements:

```bash
pip install -r requirements.txt
```

Run a small demonstration (10 trials on ER(3,0.5)):

```bash
python -m footrule_sync.main --trials 10 --out-csv sample_small_results.csv --summary-csv sample_small_summary.csv
```

Run a larger experiment:

```bash
python -m footrule_sync.main --n 20 --m 5 --p 0.2 --trials 5000 --out-csv sample_large_results.csv --summary-csv sample_large_summary.csv
```

## Testing

Unit tests verify core dynamics. Run them with:

```bash
pytest -q
```

## Sample outputs

Two CSV outputs produced for demonstrations are included:

- `sample_small_results.csv` / `sample_small_summary.csv`
- `sample_large_results.csv` / `sample_large_summary.csv`
