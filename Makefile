init:
	conda install --file requirements.txt

notebook:
	jupyter notebook notebooks/

run:
	python -m digits.main $(ARGS)

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
	cd report && pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex

report-preview: report-gen
	cd report && open report.pdf
