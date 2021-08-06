The accompanying codes reproduce all figures and statistics presented in
"Cumulative differences between subpopulations" by Mark Tygert. This repository
also provides the LaTeX and BibTeX sources required for replicating the paper.

The main files in the repository are the following:

``tex/paper.pdf``
PDF version of the paper

``tex/paper.tex``
LaTeX source for the paper

``tex/paper.bib``
BibTeX source for the paper

``codes/disjoint.py``
Functions for plotting differences between two subpops. with disjoint scores

``codes/disjoint_weighted.py``
Functions for plotting differences between two subpops. with weighted samples

``codes/imagenet.py``
Python script for processing ImageNet using a pre-trained ResNet-18

``codes/imagenet_classes.txt``
Text file containing a dictionary of the names of the classes in ImageNet

``codes/acs.py``
Python script for processing the American Community Survey

``codes/psam_h06.csv``
Microdata from the 2019 American Community Survey of the U.S. Census Bureau

Regenerating all the figures requires running in the directory ``codes`` every
Python file there. All but ``codes/imagenet.py`` need only CPUs, whereas
``codes/imagenet.py`` runs on a GPU simultaneously with running on CPU cores.

********************************************************************************

License

This fbcddisgraph software is licensed under the LICENSE file (the MIT license)
in the root directory of this source tree.
