VENV=.venv
PY=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip

.PHONY: install sanity streaming predict all clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install pandas scikit-learn joblib

sanity:
	$(PY) src/sanity_check.py

streaming:
	$(PY) src/train_streaming.py

predict:
	$(PY) src/predict.py

all: sanity streaming predict

clean:
	rm -rf models metrics
