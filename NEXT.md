# Where this stands

Built over several sessions rather than one. Both estimators, the evaluation,
the censoring counterfactual, the charts, the report and the README are done
and committed; 80 tests pass. The analysis is complete as specified.

Not done, deliberately:

* **Not published.** The repo is local. Publishing to github.com/Pratsss24
  with topics is the last step and is waiting on a decision, not on work.

Worth doing if the project is picked up again:

* **Port the `paper.py` table-break fix back to the sibling repos.** The
  vendored copy here now moves a short table whole to the next page rather
  than orphaning one or two rows onto a near-empty continuation (`PAGE_ROWS`
  and the `orphaned` test in `section`). The other fifteen repos carry the
  older copy and have the same latent fault.
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
