# Project Notes for Coding Agents

## Test environment

- This project is commonly tested on Windows with PowerShell.
- The local Python/Tk installation is incomplete. Matplotlib's default Tk backend
  fails with errors about `init.tcl`, `tk.tcl`, or an unusable Tk installation.
- Always select Matplotlib's non-interactive `Agg` backend before running tests or
  scripts that may import `matplotlib.pyplot`.
- In PowerShell, run the test suite with:

  ```powershell
  $env:MPLBACKEND = 'Agg'
  python -m pytest -q
  ```

- Apply the environment setting on the first test run; do not wait for a Tk error.
- Treat a Tk backend failure as an environment/configuration issue, not as a product
  code failure. Do not modify plotting behavior solely to work around this local Tk
  installation unless the user explicitly requests such a code change.

## Project entry points

- `ProjectControl_Loop.py` orchestrates attacks and imports their implementations
  from `Codes/attacks_new.py`.
- Focused attack-pipeline tests are in `tests/test_attack_pipeline.py`.
