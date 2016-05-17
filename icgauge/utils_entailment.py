# Parses the SNLI entailment data
#
# Resulting format is complexity.jsonl, with the properties:
#   label -- integer in {0,1}
#        1 indicates more complexity, 0 indicates less complexity
#   example -- string
#        the text itself
#
import json
import codecs

with codecs.open('complexity.jsonl', 'w', 'utf-8') as f_out:
    for source_file in ['snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl', 'snli_1.0_train.jsonl']:
        with codecs.open(source_file,'rU','utf-8') as f_in:
            for line in f_in:
                unit = json.loads(line)
                if unit['gold_label'] == 'entailment':
                    # Sentence1 entails Sentence2 means that 1 is more complex than 2
                    out = json.dumps({'label': 1, 'example': unit['sentence1']})
                    f_out.write(out + '\n')
                    out = json.dumps({'label': 0, 'example': unit['sentence2']})
                    f_out.write(out + '\n')


