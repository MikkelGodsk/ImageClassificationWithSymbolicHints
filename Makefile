.PHONY: clean data lint requirements run_imagenet run_cmplaces

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = ImageClassificationWithSymbolicHints
ENV_NAME = Image_classification_with_symbolic_hints  # Is also set in environment.yml file
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
$(error Could not find conda installation)
else
CONDA_LOC := $(shell which conda | rev | cut -d/ -f3- | rev) #~/miniconda3
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.ONESHELL:
requirements: test_environment
	-conda env create -f environment.yml
	source $(strip $(CONDA_LOC))/bin/activate
	conda activate $(ENV_NAME)
	$(PYTHON_INTERPRETER) setup.py install

## Make Dataset
.ONESHELL:
data: requirements
	source $(strip $(CONDA_LOC))/bin/activate
	conda activate $(ENV_NAME)
	$(PYTHON_INTERPRETER) src/data/make_dataset.py hydra.verbose=[__main__]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Run experiments
.ONESHELL:
run_imagenet: requirements
	source $(strip $(CONDA_LOC))/bin/activate
	conda activate $(ENV_NAME)
	$(PYTHON_INTERPRETER) src/experiments/main.py --dataset=imagenet

.ONESHELL:
run_cmplaces: requirements
	source $(strip $(CONDA_LOC))/bin/activate
	conda activate $(ENV_NAME)
	$(PYTHON_INTERPRETER) src/experiments/main.py --dataset=cmplaces

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
