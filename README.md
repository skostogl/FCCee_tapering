# FCCee_tapering studies

by Delphine and Sofia

coarse tapering via dipole main circuits, quads and sextupoles with ideal tapering in insertions
Then two scenarios:
1. orbit correctors for more refined tapering
2. or trim in one dipole of 3 dipoles in the cell (1/2 in the dispersion suppressor cells)

For each scenario, the number of dipoles tapering circuits can be defined to compare different tapering granularity. 
Change the variable N_DIPOLES_CIRCUIT_PER_SECTOR in the first cell of Circuit Definition part in the notebook.

To choose which dipole you want to use for the trim, one can use TRIM_POSITION_IN_CELL.

To do:

- refine customized tapering strategy
- rematch chroma and tunes
- scan both scenarios for more refined coarse powering of main dipoles
- provide mT m values to magnet team for both Z and ttbar


### Environment

Python >= 3.11.

```bash
conda create -n fccee-tapering python=3.13
conda activate fccee-tapering
pip install -e ".[notebook]"
```



