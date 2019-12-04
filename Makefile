ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(ARGS):;@:)

init:
	pip install pipenv
	pipenv install --dev
test:
	pipenv run python -m pytest $(ARGS)
