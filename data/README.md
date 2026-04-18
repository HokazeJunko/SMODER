# Data

This repository does not store large research datasets directly in GitHub.

## Expected input structure

A typical dataset layout is:

```text
data/
└── mousebrain_H3K27ac/
   ├── sc_mousebrain_processed.h5ad
   ├── RNA.h5ad
   └── peak.h5ad
```

## Required files

The current validated mouse brain example uses:

- `sc_mousebrain_processed.h5ad`: single-cell reference RNA
- `RNA.h5ad`: spatial RNA
- `peak.h5ad`: spatial peak data

## Notes

- Large datasets are not committed directly to the repository.
- Users should download or prepare the required files separately.
- The documentation site provides additional information on expected data organization.