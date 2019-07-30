import kenlm
import re


class KenLM:
    def __init__(self, binary_path):
        self.model = kenlm.Model(binary_path)

    def preprocess_word(self, s):
        # split a sentence into words and punctuations
        s = ' '.join(re.findall(r"\w+|[^\w\s]", s.strip().lower(), re.UNICODE))
        return s

    def score(self, s, preprocess=True):
        if preprocess:
            s = self.preprocess_word(s)
        return self.model.score(s)
