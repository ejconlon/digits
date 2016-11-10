digits
======

a.k.a. "Classifying Street View House Numbers with Deep Learning"

Project dependencies can be handled through `pip` or `conda` using the provided `requirements.txt`.

Most other tasks can be accomplished with `make`.  From a clean checkout, you can fetch datasets with

    make fetch

run a simple sanity check with

    make test

run the final models and generate report assests with

    make gen

and with LaTeX installed you can generate the report with

    make report-gen

(or you can just read the checked-in copy `report/report.pdf`).
