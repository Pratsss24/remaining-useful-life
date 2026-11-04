# Where this stands

Built over several sessions rather than one. Both estimators, the evaluation,
the censoring counterfactual, the charts, the report and the README are done
and committed; 80 tests pass. The analysis is complete as specified.

Published at https://github.com/Pratsss24/remaining-useful-life.

Worth doing if the project is picked up again:

* **FD002 and FD004 lead runs.** `run.py --subset FD002` works and the
  all-four-fleets table is computed every run, but the report's narrative
  pages are written around FD001 as the lead. Nothing is wrong on the other
  fleets; the prose simply names FD001's numbers.
* **The 50–75 band.** It is the one band where the age law beats the
  degradation signal on FD001 (20.2 against 25.2) and the reason is not
  established beyond "near the cap". A diagnosis would be a real addition.

Run it with:

    C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe run.py
    C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests/ -q

The data layer fetches about 12 MB on first use and caches it in `data/`,
which is gitignored and reproducible; `src/data.py` pins the archive's
SHA-256 for both zip layers.
