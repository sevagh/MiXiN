black:
	black *.py primalx/*.py primalx/*/*.py

fmt: black

.PHONY: black fmt
