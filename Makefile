init:
	# deps for report generation
	# provides `dot` command
	brew install graphviz
	# provides `pdflatex` and `biber` commands
	brew cask install mactex
	# provides `inkscape` command
	brew cask install inkscape
	# you may need to pip install some of these
	conda install --file requirements.txt

notebook:
	jupyter notebook notebooks/

run:
	python -m digits.main $(ARGS)

fetch:
	@$(MAKE) run ARGS="fetch_mnist"
	@$(MAKE) run ARGS="fetch_svhn"

clean:
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
