# ToeplitzLDA

This is an anonymized version of this repository without `.git`

Code for the ToeplitzLDA classifier proposed in (TODO insert reference).
The classifier conforms sklearn and can be used as a drop-in replacement for other LDA classifiers.
For usage refer to the learning from label proportions (LLP) example or the benchmark scripts.

## Learning from label proportions

Use the run_llp.py script to apply ToeplitzLDA in the LLP scenario and create results file that can then be visualized using visualize_llp.py to create the plots shown in the publication at TODO.
Note that the two datasets will be downloaded automatically and are approximately 16GB in size.

## ERP benchmark

This is not yet available.

Note this benchmark will take quite a long time if you do not have access to a computing cluster.
The public datasets (including the LLP datasets) total a size of approximately 35GB.

BLOCKING TODO: How should we handle the private datasets?
