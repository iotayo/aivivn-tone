import re

LEGAL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

infile = "test.txt"

with open(infile, 'r') as f:
    lines = f.readlines()

abbrev = set()
for line in lines:
    line = list(line[4:].strip())
    line = ['-' if c not in LEGAL else c for c in line]
    line = re.split('-', ''.join(line))
    for c in line:
        if len(c) > 1:
            abbrev.add(c)

for c in sorted(abbrev):
    if c.isalpha():
        print(c)
