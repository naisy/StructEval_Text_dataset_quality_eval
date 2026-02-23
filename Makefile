.PHONY: install eval filter

install:
	python -m pip install -r requirements.txt

eval:
	python -m dataset_eval.run_eval --config configs/eval.yaml

filter:
	python -m dataset_eval.run_filter --config configs/filter.yaml
