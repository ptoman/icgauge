# -*- coding: utf-8 -*-
#!/usr/bin/python

def identity_class_func(y):
    """Leave the data without transformation, as in qualitative
       work and Conway.""" 
    return y
        
def ternary_class_func(y):
    """Transform the data into a ternary task, as in Kannan Ambili."""    
    if y in (1,2):
        return 1
    elif y in (3,4,5):
        return 2
    elif y in (6,7):
        return 3
    else:
        raise ValueError("The input value " + y + " is invalid")
