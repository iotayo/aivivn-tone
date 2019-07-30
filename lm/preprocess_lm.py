import re


infile = "corpus-full-0.2-shuffled.txt"
outfile = "corpus-full-0.2-shuffled-wplm-v2.txt"

j = 0
with open(infile, 'r') as fr, open(outfile, 'w') as fw:
    for line in fr:
        line = ' '.join(re.findall(r"\w+|[^\w\s]", line.strip().lower(), re.UNICODE))

        fw.write(line)
        fw.write('\n')

        j += 1
        if j % 100000 == 0:
            print(str(j) + ' ' + line)
