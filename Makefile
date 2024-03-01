.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using pyenv and poetry"
	@poetry install	
	@ poetry run pre-commit install
	@poetry shell

.PHONY: yapf
yapf: ## Run yapf on main module and tests
	@echo "🚀 Formatting files with YAPF"
	@poetry run yapf -i --recursive --parallel pfl/ tests/

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking Poetry lock file consistency with 'pyproject.toml': Running poetry check --lock"
	@poetry check --lock
	@echo "🚀 Linting code: Running pre-commit"
	@poetry run pre-commit run -a

.PHONY: cov
cov: ## Run test coverage
	@echo "🚀 Checking test coverage"
	@pytest --doctest-modules tests --cov --cov-config=pyproject.toml --cov-report=term  --disable_horovod

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@poetry run pytest -svx tests/

.PHONY: build
build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@poetry build

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: docs
docs: ## Build documentation
	@echo "🚀 Building documentation"
	@poetry run sphinx-build -b html docs/source docs/build
	@echo "🚀 Compiled documentation available in docs/build/"

.PHONY: docs-and-publish
docs-and-publish: 
	@./build_scripts/publish_docs.sh

.PHONY: publish
publish: ## publish a release to pypi.
	@echo "🚀 Publishing."
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	@poetry publish

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
