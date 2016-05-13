#!/usr/bin/python
##
## Usage: python -m unittest tests.data_tests
##

import unittest
import os
import json

class DataTests(unittest.TestCase):
    """
    Abuse of testing framework to verify that all
    data are in the expected format.
    """
    
    def setUp(self):
        """ No setup required """
        pass
    
    def tearDown(self):
        """ No cleanup required """
        pass
    
    def testFormatOfSampleData(self):
        """ Test the format against files in sample_data """
        print "DataTest:testFormatOfSampleData"
        self.helper_testFormat("sample_data")
        
    def testFormatOfData(self):
        """ Test the format against files in sample_data """
        print "DataTest:testFormatOfData"
        self.helper_testFormat("data")

    def testFormatOfSampleData(self):
        """ Test the format against files in sample_data """
        print "DataTest:testFormatOfSampleData"
        self.helper_testAscii("sample_data")
        
    def testFormatOfData(self):
        """ Test the format against files in sample_data """
        print "DataTest:testFormatOfData"
        self.helper_testAscii("data")
                
    def helper_testFormat(self, dirname):
        """ 
        Ensure each .json file in the dirname directory is:
          - an array of objects 
          - contains a populated `paragraph` field
          - if 'score' appears, it is an integer 1-7 or "NA"
        """
        for fn in os.listdir(dirname):
            if fn.endswith(".json"):
                print "  Checking", fn
                with open(os.path.join(dirname,fn)) as json_file:
                    dataset = json.load(json_file)
                    for item in dataset:
                        self.assertIn("paragraph", item, "The `paragraph` property is missing")
                        self.assertIsNotNone(item["paragraph"], "A `paragraph` content is missing")
                        self.assertGreater(len(item["paragraph"]), 0, "A `paragraph` content has no length")
        
                        if "score" in item:
                            self.assertIsNotNone(item["score"], "A `score` content is missing")
                            self.assertIn(item["score"], [1, 2, 3, 4, 5, 6, 7, "NA"], "A `score` value is invalid")
    
    def helper_testAscii(self, dirname):
        """
        Ensure each .json file in the dirname directory
        contains only ASCII characters.
        """
        for fn in os.listdir(dirname):
            if fn.endswith(".json"):
                print "  Checking", fn
                with open(os.path.join(dirname,fn)) as json_file:
                    for line in json_file:
                        for c in line:
                            self.assertTrue(ord(c) < 128, "File contains a non-ASCII character")

if __name__ == '__main__':
    unittest.main()
