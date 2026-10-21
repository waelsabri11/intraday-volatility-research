.PHONY: setup data daily hourly sizing figures all test
setup:
	uv sync
data:
	uv run ivr download && uv run ivr process
daily:
	uv run ivr daily
hourly:
	uv run ivr hourly
sizing:
	uv run ivr sizing
figures:
	uv run ivr figures
all:
	uv run ivr all
test:
	uv run pytest -q
