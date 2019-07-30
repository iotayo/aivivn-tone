import unidecode


train_data = "data/train.txt"
train_tgt = "data/train_full.tgt"
train_src = "data/train_full.src"

vn = 'aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵdđ'
aeiouyd = ['a', 'e', 'i', 'o', 'u', 'y', 'd']
legal = ' !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ{|}~'
punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def preprocess(line):
    line = line.strip().lower()
    line = ''.join(c if c not in punct else '-' for c in line)  # replace all punctuations with '-'
    line = ''.join(c if c in legal else '?' for c in line)  # replace unknown characters with '?'
    return line


with open(train_data, 'r') as f:
    lines = f.readlines()

with open(train_tgt, 'w') as ft, open(train_src, 'w') as fs:
    for line in lines:
        line = line.strip().lower()
        line = ''.join(c if c not in punct else '-' for c in line)  # replace all punctuations with '-'
        line = ''.join(c if c in legal else '?' for c in line)  # replace unknown characters with '?'
        line_no_tone = unidecode.unidecode(line)

        if len(line) <= 300:
            ft.write(line)
            ft.write('\n')
            fs.write(line_no_tone)
            fs.write('\n')
