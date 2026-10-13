# Thin wrappers; the Python scripts are the primary interface.
PYTHON ?= python

.PHONY: run validate all serve test clean

all: run validate

run:
	$(PYTHON) run_pipeline.py

validate:
	$(PYTHON) validate.py

serve:
	$(PYTHON) serve.py

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "from rag.jsonio import ArtifactStore; from rag import config; ArtifactStore(config.DEFAULT_ARTIFACTS_DIR).clean()"
