black:
	black *.py mixin/*.py mixin/*/*.py

lint:
	pyflakes *.py mixin/*.py mixin/*/*.py | grep -v '__init__.py'

fmt: black

.PHONY: black fmt
