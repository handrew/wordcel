.PHONY: sync
sync:
	poetry install --with dev

.PHONY: format
format: 
	poetry run ruff format
	poetry run ruff check --fix

.PHONY: lint
lint: 
	poetry run ruff check

.PHONY: mypy
mypy: 
	poetry run mypy .

.PHONY: tests
tests: 
	poetry run pytest 

.PHONY: coverage
coverage:
	poetry run coverage run -m pytest
	poetry run coverage xml -o coverage.xml
	poetry run coverage report -m --fail-under=95

.PHONY: snapshots-fix
snapshots-fix: 
	poetry run pytest --inline-snapshot=fix 

.PHONY: snapshots-create 
snapshots-create: 
	poetry run pytest --inline-snapshot=create 