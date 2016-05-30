import json

with open('toy.json', 'r') as f:
    json_data = json.load(f)

def remove_punc(par):
	puncs = [',', '.', ';', '-', '?', '!']	# Remove commas and periods
	out = "".join([c for c in par if c not in puncs])
	return out.lower() + '.'

for i in range(len(json_data)):
	par = json_data[i]['paragraph']
	del json_data[i]["parse"]
	new_par = remove_punc(par)
	json_data[i]["paragraph"] = new_par
	print new_par


with open('toy_unpunctuated.json', 'w') as f:
    json.dump(json_data, f)

"""    
    json_data['b'] = "9"

with open('toy_unpuctuated.json', 'w') as f
    f.write(json.dumps(json_data))
"""