ICGauge
=======

This project seeks to automatically measure integrative complexity (cognitive complexity) in English texts.  

Briefly, integrative complexity is a measurement of to what extent the writer is entertaining multiple perspectives as valid.  It is measured at the paragraph level using a 7-point scale in which 1 indicates "very little complexity" (more formally, only a single idea is entertained) and 7 indicates "extremely complexity" (more formally, at least two potentially conflicting ideas are entertained, developed, and acknowledged to be simultaneously true with a discussion of how they can both be true simultaneously).

This project has four main directories:
* experiments: Experiment scripts and notes, one experiment per file.
* icgauge: Modeling package.  Contains experiment_frameworks (currently consisting only of experiment_features, a framework that expects a hand-provided set of feature functions), as well as support methods separated into their types: data_readers, feature_extractors, label_transformers, training_functions, and utils.
* sample_data: Sample data supporting the framework.  Contains examples of each level of complexity.
* tests: Contains an abuse of the testing framework to verify all data has the expected format, plus any actual tests that might get written.

To run all the experiments, you'll want to have the following environmental variables set:
* `GLV_HOME`: location of GloVe vector .txt files
* `STANFORD_NLP_HOME`: location of Stanford NLP .jar files

To run the toy experiment, use the following at this top-level directory:
    `python -m experiments.toy`
