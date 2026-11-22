# Convenience targets.  The tool itself is a plain Python package; these are
# just shorthand.

.PHONY: suite test quick serve clean

# Full acceptance suite: four services over a full day, tuning, reverse
# calibration, HTML report in reports/.
suite:
	python3 -m gcsim suite --outdir reports

# Mechanism tests + cohort-vs-oracle validation.
test:
	python3 -m pytest -q

# A quick subset (skip the slow object-by-object oracle) for iteration.
quick:
	python3 -m pytest -q -k "not oracle"

# Interactive browser UI at http://127.0.0.1:8000
serve:
	python3 -m gcsim serve --results reports/results.json

clean:
	rm -f reports/index.html reports/results.json reports/summary.md
	rm -rf reports/logs __pycache__ gcsim/__pycache__ tests/__pycache__
