digits
======

a.k.a. "Classifying Street View House Numbers with Deep Learning"

Read `report/report.pdf` for an explanation of some simple approaches to classification of SVHN and MNIST datasets with both conventional and deep learning methods.

Building and running
--------------------

This project has been built and tested on macOS with Homebrew, Ubuntu Trusty, and Debian Jessie. Python dependencies are handled through `pip`. There are `make` targets to help fetch dependencies (`make deps-mac` and `make deps-linux`). However, the easiest way to run it is to use Vagrant and VirtualBox (or another VM provider) with the included Vagrantfile to bootstrap a known-good environment. On macOS you can setup a VM and run the whole application like

    # Install Vagrant and VirtualBox with `brew cask`
    make deps-virtual
    # Provision and boot the VM
    vagrant up
    # SSH into the running VM
    vagrant ssh
    # Now on the virtual machine cd into the project dir
    cd /digits
    # Fetch data
    make fetch
    # Run the application
    make clean all
    # Exit the VM
    exit
    # Now on your machine again, remove the VM
    vagrant destroy -f
    # Optionally remove the data you've downloaded and any results
    make clean nuke

PLEASE NOTE: The `make fetch` step will download about 1.5 GB of SVHN and MNIST data!

The macOS build is currently only configured to use Python 3, but you can massage that (see the Makefile). You would need to change tensorflow versions and use a different set of requirements. Linux builds (as setup with Vagrant above) use Python 2.

`make all` does a few things things: First, it runs a test suite that checks basic functionality. Then it trains several models on several datasets based on tuned parameters.  Finally, it evaluates all the IPython notebooks in the repo to generate some explorable reports and artifacts. The `results` directory might be worth exploring!

Generating the report
---------------------

In all likelihood you don't want to generate the report.  Instead you probably just want to read what's already generated and checked-in. The generation process makes use of `make all` output (specifically `make results-gen`) to include various facts and figures. If you manage to get TeX Live 2016 installed (see `make deps-mac`), you can generate the report with `make report-gen`.
