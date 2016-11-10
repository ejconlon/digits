digits
======

a.k.a. "Classifying Street View House Numbers with Deep Learning"

Project dependencies can be handled through `pip` or `conda` using the provided `requirements.txt`. To generate
report artifacts and the report itself you will need `dot`, `pdflatex`, and `biber`, which can be easily
installed through package managers.  (See `make init` for an example.)

Most other tasks can be accomplished with `make`.  From a clean checkout, you can fetch datasets with

    make fetch

run a simple sanity check with

    make test

run the final models and generate report assests with

    make gen

and with LaTeX installed you can generate the report with

    make report-gen

(or you can just read the checked-in copy `report/report.pdf`).
