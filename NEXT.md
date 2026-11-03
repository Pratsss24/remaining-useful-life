# Where this stands

Work in progress, built over several sessions rather than one. Two commits so
far: the data layer and the first of two planned estimators. No report yet, and
no README with the headline result, because there is no headline result yet --
the second estimator is what the project is actually about.

Working now, with 46 tests passing:

* `src/data.py` -- fetches and caches the NASA C-MAPSS turbofan run-to-failure
  archive, slices it into per-engine records, and recovers the operating-regime
  count from the settings rather than assuming it.
* `src/lifetimes.py` -- lifetime records, and the four ways of treating the
  censored units.
* `src/survival.py` -- Kaplan-Meier and non-parametric mean residual life, as
  the distribution-free check on the parametric work.
* `src/agelaw.py` -- estimator A: remaining life from the fleet's lifetime law,
  given nothing but an engine's age.
* `src/simulate.py` -- planted-truth fleets under three censoring schemes.

Still to come: the sensor-driven estimator that estimator A is the floor for,
the evaluation module that reports error against true remaining life and
measured interval coverage, the sensor-selection analysis, the charts, and the
PDF report.

Run the tests with:

    C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests/ -q

The data layer fetches about 12 MB on first use and caches it in `data/`, which
is gitignored and reproducible; `src/data.py` pins the archive's SHA-256.
