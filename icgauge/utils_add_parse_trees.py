#!/usr/bin/python

# This file appends parse trees to all the datasets if the data 
# do not currently have trees 
#
# Usage: python -m icgauge.utils_add_parse_trees

import os
import json

from utils_parsing import get_trees_given_paragraph

def add_parse_trees():
    for dirname in ["sample_data", "data"]:
        for fn in os.listdir(dirname):
            if fn.endswith(".json"):
                print "  Checking", fn
                with open(os.path.join(dirname,fn)) as json_file:
                    needs_new_file = False
                    dataset = json.load(json_file)
                    revised_items = []
                    for item in dataset:
                        if "parse" not in item:
                            print item
                            needs_new_file = True
                            tree_strings = [str(tree) for tree in get_trees_given_paragraph(item['paragraph'])]
                            item["parse"] = tree_strings
                        revised_items.append(item)
                    if needs_new_file:
                        with open(os.path.join(dirname, fn+"_parsed.json"), 'w') as to_file:
                            json.dump(revised_items, fp=to_file, indent=4)

if __name__ == '__main__':
    add_parse_trees()
