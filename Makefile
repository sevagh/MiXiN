black:
	black *.py primalx/*.py primalx/*/*.py

lint:
	pyflakes *.py primalx/*.py primalx/*/*.py | grep -v '__init__.py'

fmt: black

.PHONY: black fmt
