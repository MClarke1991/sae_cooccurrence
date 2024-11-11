format:
	poetry run ruff format

check-linting:
	poetry run ruff check
	poetry run ruff format --check

check-linting-exclude:
	poetry run ruff check --exclude temp,src/example_clusters
	poetry run ruff format --check --exclude temp,src/example_clusters

check-type:
	poetry run pyright .

check-type-exclude:
	poetry run pyright . #--exclude "temp/**" "src/example_clusters/**"

test:
	poetry run pytest

check-ci:
	make check-linting
	make check-type
	make test

check-ci-exclude:
	make check-linting-exclude
	make check-type-exclude
	make test
