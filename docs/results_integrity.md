# Results integrity manifest

`config/results_integrity.json` records SHA-256 digests for every committed file
under `results/`, including figures, result tables, split data, and layout files.
The quality gate and CI verify both the file set and each file's bytes.

After intentionally regenerating and reviewing result artifacts, update the
manifest with:

```powershell
python scripts/results_integrity.py --write
```

Review the manifest diff together with the regenerated files, then run:

```powershell
python scripts/results_integrity.py --check
```

Do not use `--write` to bless unexplained or accidental result changes.
