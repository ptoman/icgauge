ICGauge
=======

This project raises state-of-the-art performance on automatically measure integrative complexity (cognitive complexity) in English texts from Pearson's *r* = 0.57 to *r* = 0.73.

From the [abstract](https://github.com/ptoman/icgauge/blob/master/deliverables/evaluating-level-semantic-complexity-final.pdf):
> Integrative complexity is a construct from political psychology that measures semantic complexity in discourse. Although this metric has been shown useful in predicting violence and understanding elections, it is very time-consuming for analysts to assess. We describe a theory-driven automated system that improves the state-of-the-art for this task from Pearson's *r* = 0.57 to *r* = 0.73 through framing the task as ordinal regression, leveraging dense vector representations of words, and developing syntactic and semantic features that go beyond lexical phrase matching. Our approach is less labor-intensive and more transferable than the previous state-of-the-art for this task. The success of this system demonstrates the usefulness of word vectors in transferring context into new problems with limited available data.

Briefly, integrative complexity is a measurement of to what extent the writer is entertaining multiple perspectives as valid.  It is measured at the paragraph level using a 7-point scale in which 1 indicates "very little complexity" (more formally, only a single idea is entertained) and 7 indicates "extremely complexity" (more formally, at least two potentially conflicting ideas are entertained, developed, and acknowledged to be simultaneously true with a discussion of how they can both be true simultaneously).

This project has five main directories:
* icgauge: Modeling package.  Contains experiment_frameworks (currently consisting only of experiment_features, a framework that expects a hand-provided set of feature functions), as well as support methods separated into their types: data_readers, feature_extractors, label_transformers, training_functions, and utils.
* experiments: Experiment scripts and notes, one experiment per file.
* lstm: Modeling files for the LSTM developed as part of this work.
* sample_data: Sample data supporting the framework.  Contains examples of each level of complexity.
* tests: Contains a stretch of the testing framework to verify all data have the expected format.

Documentation in the form of papers can be found in the deliverables folder.

To run all the experiments, you'll want to have the following environmental variables set:
* `GLV_HOME`: location of GloVe vector .txt files
* `STANFORD_NLP_HOME`: location of Stanford NLP .jar files

To run the toy experiment, use the following at this top-level directory:
    `python -m experiments.toy`

License
-------
MIT license: If you use this code, you must attribute it.  Please also contribute your improvements back into the codebase.
