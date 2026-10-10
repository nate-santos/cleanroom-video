.PHONY: install test demo lint clean

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest -q

demo:
	cleanroom demo

lint:
	python -m compileall -q src tests

clean:
	rm -rf assets/work/* .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
