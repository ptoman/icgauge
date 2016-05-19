# Parses the entailment data
#
# Resulting format is complexity.jsonl, with the properties:
#   label -- integer in {0,1}
#        1 indicates more complexity, 0 indicates less complexity
#   example -- string
#        the text itself
#
import json
import codecs


# SNLI
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


# SICK
with codecs.open('complexity.jsonl', 'w', 'utf-8') as f_out:
    for source_file in ['SICK.txt']:
        with codecs.open(source_file,'rU','utf-8') as f_in:
            for line in f_in:
                _, sentA, sentB, _, _, atob, btoa, _, _, _, _, _ = line.split('\t')
                if atob == "A_entails_B" and btoa == "B_neutral_A":
                    # A is more complex than B
                    out = json.dumps({'label': 1, 'example': sentA})
                    f_out.write(out + '\n')
                    out = json.dumps({'label': 0, 'example': sentB})
                    f_out.write(out + '\n')
                if btoa == "B_entails_A" and atob == "A_neutral_B":
                    # B is more complex than A
                    out = json.dumps({'label': 1, 'example': sentB})
                    f_out.write(out + '\n')
                    out = json.dumps({'label': 0, 'example': sentA})
                    f_out.write(out + '\n')
                    

# PPDB
# This code just shows examples, illustrating that this dataset is on the
# whole quite bidirectional.
ctr = 0
with codecs.open('complexity.jsonl', 'w', 'utf-8') as f_out:
    for source_file in ['ppdb-1.0-s-all']:
        with codecs.open(source_file,'rU','utf-8') as f_in:
            for line in f_in:
                _, sentA, sentB, _, _ = line.split(' ||| ')
                combo = sentA + "___" + sentB
                comboRev = sentB + "___" + sentA
                ctr += 1
                if ctr % 10000 == 0:
                    print sentA
                    print sentB
                    print
