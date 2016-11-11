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
	pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py3-none-any.whl
	pip install -r requirements35.txt

deps-virtual:
	brew cask install virtualbox
	brew cask install vagrant

deps-jessie:
	# installs deps for execution and report generation on debian jessie
	sudo apt-get install graphviz texlive inkscape python-pip python-numpy python-matplotlib python-pandas python-zmq -y
	sudo pip install --upgrade pip
	sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
	sudo pip install -r requirements27.txt

notebook:
	jupyter notebook notebooks/

run:
	python -m digits.main $(ARGS)

fetch:
	@$(MAKE) run ARGS="fetch_mnist"
	@$(MAKE) run ARGS="fetch_svhn"

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

all: clean fetch test-verbose results-gen report-gen
