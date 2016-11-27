deps-mac:
	# installs deps for execution and report generation on mac
	# provides `dot` command for diagrams
	brew install graphviz
	# provides `pdflatex` and `biber` commands for latex to pdf conversion
	brew cask install mactex
	# provides `inkscape` command for svg to latex conversion
	brew cask install inkscape
	pip install cython
	# PLEASE NOTE this installs for py3. You can change this...
	pip install --ignore-installed https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py3-none-any.whl
	pip install -r requirements35.txt

deps-virtual:
	brew cask install virtualbox
	brew cask install vagrant

deps-linux:
	# installs deps for execution and report generation on ubuntu trusty or debian jessie
	sudo apt-get install graphviz python-pip python-numpy python-matplotlib python-pandas python-zmq -y
	sudo pip install --upgrade pip
	sudo pip install --ignore-installed https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
	sudo pip install -r requirements27.txt

notebook:
	jupyter notebook notebooks/

run:
	python -m digits.main $(ARGS)

fetch:
	mkdir -p data
	mkdir -p pickled
	mkdir -p logs
	mkdir -p results
	@$(MAKE) run ARGS="fetch_mnist"
	@$(MAKE) run ARGS="fetch_svhn"
	#@$(MAKE) run ARGS="fetch_svhn_img"

results-gen:
	@$(MAKE) run ARGS="notebooks"

clean:
	mkdir -p data
	mkdir -p pickled
	mkdir -p logs
	mkdir -p results
	rm -rf pickled/*
	rm -rf logs/*
	rm -rf results/*

nuke: clean
	rm -rf data/*

train:
	@$(MAKE) run ARGS="drive --model=tf --variant=mnist"
	@$(MAKE) run ARGS="drive --model=tf --variant=crop-huge"
	@$(MAKE) run ARGS="drive --model=vote --variant=mnist"
	@$(MAKE) run ARGS="drive --model=vote --variant=crop-huge"
	@$(MAKE) run ARGS="drive --model=baseline --variant=mnist"
	@$(MAKE) run ARGS="drive --model=baseline --variant=crop-big"

test:
	py.test tests

test-verbose:
	py.test -s tests --fulltrace

tensorboard:
	python -m tensorflow.tensorboard --logdir=logs/

report-clean:
	cd report && rm -f report.{aux,log,out,toc,bcf,bbl,blg,run.xml}

report-gen: report-clean
	cd report && pdflatex --shell-escape report.tex && biber report && pdflatex --shell-escape report.tex && pdflatex --shell-escape report.tex

report-preview: report-gen
	cd report && open report.pdf

all: test-verbose results-gen
