Integrative complexity: Sample data
===========

The data are UTF-8 files in JSON format, currently containing only ASCII characters.  Each file contains an array of paragraph-and-meta-data objects. The only required field is `paragraph`.

The file `test/data_tests.py` verifies that all data are in the right format.

Possible fields
----------------
* `paragraph`: String containing the paragraph to score.
* `score`: Integer in range [1,7], reflecting the human-generated assessment of complexity, or the string "NA", reflecting an unscorable paragraph.  Values of 1 indicate "not complex" and 7 indicate "highly complex".

File descriptions
----------------
* `toy.json`: Contains annotated toy development data sourced from [the Baker-Brown et al. coding manual](http://www2.psych.ubc.ca/~psuedfeld/MANUAL.pdf) available on Peter Suedfeld's [Complexity Materials](http://www2.psych.ubc.ca/~psuedfeld/Download.html) site. This data has human scores for each paragraph.  It contains 68 examples, with higher levels of complexity underrepresented.
* `unscorable.json`: Contains unscorable data sourced from [the Baker-Brown et al. coding manual](http://www2.psych.ubc.ca/~psuedfeld/MANUAL.pdf) available on Peter Suedfeld's [Complexity Materials](http://www2.psych.ubc.ca/~psuedfeld/Download.html) site. It contains 4 examples.
