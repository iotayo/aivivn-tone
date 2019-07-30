import re


infile = "words.txt"

LEGAL = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`bcfghjklmnpqrstvwxz{|}~"

with open(infile, 'r') as f:
    lines = f.readlines()

vc = set()
for line in lines:
    line = list(line.lower().strip())
    line = ['-' if c in LEGAL else c for c in line]
    line = re.split('-', ''.join(line))
    for c in line:
        if len(c) > 1:
            vc.add(c)

infile = "test.txt"

with open(infile, 'r') as f:
    lines = f.readlines()

for line in lines:
    line = list(line[4:].lower().strip())
    line = ['-' if c in LEGAL else c for c in line]
    line = re.split('-', ''.join(line))
    for c in line:
        if len(c) > 1:
            vc.add(c)

for c in sorted(vc):
    if c.isalpha():
        print(c)
