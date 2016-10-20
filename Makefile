init:
	conda install --file requirements.txt

notebook:
	jupyter notebook notebooks/

test:
	py.test tests

test-verbose:
	py.test -s tests

tensorboard:
	python -m tensorflow.tensorboard --logdir=logs/
