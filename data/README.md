# Data

The current validated example workflow uses a mouse brain RNA + peak dataset.

## Required input files

The current example expects three input files:

- `sc_mousebrain_processed.h5ad`: single-cell reference RNA
- `RNA.h5ad`: spatial RNA
- `peak.h5ad`: spatial peak data

## Example directory layout

```text
data/
└── mousebrain_H3K27ac/
   ├── sc_mousebrain_processed.h5ad
   ├── RNA.h5ad
   └── peak.h5ad
```

## How to obtain the data

The validated example dataset is not currently distributed through a public download link.

Please contact the maintainer to obtain the example files.

## Notes

After obtaining the required files, place them into a local data directory matching the expected structure.

The file paths can then be configured in the pipeline configuration.