# Data

The tracked file in `sample/` is a compact project sample copied from the OBSEA wave-observation export. It is intended for demonstrating the CSV schema and the inference command, not as a replacement for the complete research datasets.

The associated paper evaluates three sources:

- Tarragona buoy observations from Puertos del Estado.
- Barcelona buoy observations from Puertos del Estado.
- EMSO-OBSEA observations collected by the underwater observatory.

The original working directory contains multiple raw and derived CSV exports, including duplicate downloads and a very large summary export. Those files are intentionally ignored by Git to keep the repository reviewable and below common hosting limits. Before redistributing full raw datasets, check the terms of the original data providers and retain their metadata/provenance.

## Expected feature schema

For inference, a CSV must contain at least these numeric columns:

`VDMR` or `VMDR`, `VTPK`, `VZMX`, and `VTZA`.

The optional `time` and `VHM0` columns are preserved in the sample but are not required to generate a one-step prediction.
