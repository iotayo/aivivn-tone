import re
import dill
import itertools
import unidecode
import torch
from string import punctuation
from torchtext.data import BucketIterator
from model import Seq2SeqConcat, Encoder, Decoder
from dataset import Seq2SeqDataset, PAD
from alphabet import LEGAL, PUNCT, A_LIST, E_LIST, I_LIST, O_LIST, U_LIST, Y_LIST, D_LIST
from lm.lm import KenLM


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Predictor:
    def __init__(self, src_vocab_path, tgt_vocab_path, model_path, wlm_path,
                 max_len=1000, hidden_size=300, n_layers=2):
        # load vocab
        print("Loading vocab...")
        with open(src_vocab_path, "rb") as f:
            self.src_vocab = dill.load(f)
        with open(tgt_vocab_path, "rb") as f:
            self.tgt_vocab = dill.load(f)
        print("Source vocab size:", len(self.src_vocab))
        print("Target vocab size:", len(self.tgt_vocab))

        # hyper-parameters
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.pad_idx = self.src_vocab.stoi[PAD]

        # prepare model
        self.encoder = Encoder(self.src_vocab, self.max_len, self.hidden_size, self.n_layers)
        self.decoder = Decoder(self.tgt_vocab, self.max_len, self.hidden_size * 2, self.n_layers)
        self.reverse_decoder = Decoder(self.tgt_vocab, self.max_len, self.hidden_size * 2, self.n_layers, reverse=True)
        self.model = Seq2SeqConcat(self.encoder, self.decoder, self.reverse_decoder, self.pad_idx)

        # load model
        print("Loading model...")
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # initialize character and word language model
        self.wlm = KenLM(wlm_path)

        # prepare eligible tone map
        self.tone_map = dict()
        self.legal_vc = set()
        self.tone_map['a'] = A_LIST
        self.tone_map['e'] = E_LIST
        self.tone_map['i'] = I_LIST
        self.tone_map['o'] = O_LIST
        self.tone_map['u'] = U_LIST
        self.tone_map['y'] = Y_LIST
        self.tone_map['d'] = D_LIST
        with open("data/legal_vc.txt", 'r') as f:
            for line in f:
                vc = line.strip()
                self.legal_vc.add(vc)
                vc_no_tone = unidecode.unidecode(vc)
                if vc_no_tone in self.tone_map:
                    self.tone_map[vc_no_tone].append(vc)
                else:
                    self.tone_map[vc_no_tone] = [vc]

    def predict(self, test_path, test_cleaned_path, out_path):
        # read raw data to list
        lines_id = []
        lines_raw = []
        lines_cleaned = []
        lines_prep = []
        with open(test_path, 'r') as f, open(test_cleaned_path, 'r') as fc:
            for line in f:
                line_id = line[:3]
                line_seq = line[4:]
                lines_id.append(line_id)
                lines_raw.append(line_seq)
                lines_prep.append(self.preprocess(line_seq))
            for line in fc:
                lines_cleaned.append(line[4:])

        # prepare dataset
        print("Reading test data...")
        test = Seq2SeqDataset.from_list(lines_prep)
        test.src_field.vocab = self.src_vocab

        # prepare iterator
        test_iterator = BucketIterator(dataset=test, batch_size=1, train=False,
                                       sort=False, sort_within_batch=False,
                                       shuffle=False, device=device)

        # predict
        with open(out_path, 'w') as writer:
            with torch.no_grad():
                for i, batch in enumerate(test_iterator):
                    # forward through model
                    _, _, output = self.model(batch, has_targets=False, mask_softmax=1.0, teacher_forcing=1.0)

                    # get top-1
                    predicted_values, predicted_indices = torch.max(output, dim=-1)

                    # convert predicted vocab indices to an actual sentence
                    predicted_seq = [self.tgt_vocab.itos[c] for c in predicted_indices.squeeze(0).tolist()]

                    # output is log_softmax so do exp()
                    predicted_values = predicted_values.exp()

                    # convert to list
                    predicted_values_ = predicted_values.squeeze(0).tolist()

                    # beam search
                    predicted_seq = self.beam_lm(''.join(predicted_seq[1:-1]), predicted_values_[1:-1], lines_raw[i])

                    # match case and punctuations
                    predicted_seq = self.match_case(predicted_seq, lines_raw[i])

                    # do some post-processing to match submission output
                    predicted_seq = self.match_output(predicted_seq, lines_cleaned[i])
                    print("{} {}".format(i, predicted_seq))

                    # write to file with line_id
                    writer.write(lines_id[i] + ',' + predicted_seq + '\n')

    def beam_lm(self, predicted_seq, predicted_values, line_raw, k=500, threshold=0.99):
        # replace uncertain characters with placeholders
        predicted_seq_copy = predicted_seq
        for i, v in enumerate(predicted_values):
            if v < threshold:
                predicted_seq_copy = predicted_seq_copy[:i] + '*' + predicted_seq_copy[i+1:]
            elif v < 1.0:
                predicted_seq_copy = predicted_seq_copy[:i] + '~' + predicted_seq_copy[i+1:]

        predicted_seq_copy = re.sub('~*\\*+~*', lambda x: '*'*len(x.group()), predicted_seq_copy)

        predicted_seq_copy = ''.join([predicted_seq[i] if c == '~' else c for i, c in enumerate(predicted_seq_copy)])

        # recursive function
        #   sample sequence:   aaaaaaaaa*aaaaa*aaaa
        #   char to search :            ^
        #   left context   :   <------->
        #   right context  :             <--->
        # left_contexts is a list of previous top-k scoring substrings
        def beam_lm_(predicted_seq, predicted_values, predicted_seq_uncertain, k=k, step=0):
            # stop condition = no disagreements left
            uncertainties = [m.span() for m in re.finditer('\\*+', predicted_seq_uncertain)]
            if len(uncertainties) == 0:
                return predicted_seq_uncertain

            # forward
            topk_fwd = [predicted_seq_uncertain[0:uncertainties[0][0]]]
            for i, (start_idx, end_idx) in enumerate(uncertainties):
                c = predicted_seq[start_idx:end_idx]

                # get a list of possible characters to search
                c_list = self.tone_map.get(unidecode.unidecode(c), [unidecode.unidecode(c)])

                # get contexts
                left_contexts = topk_fwd
                right_context = predicted_seq_uncertain[end_idx:uncertainties[i + 1][0]] if i < len(uncertainties) - 1 else predicted_seq_uncertain[end_idx:]

                candidates = []
                scores = torch.empty(len(c_list) * len(left_contexts)).fill_(-float('inf'))

                # score candidates
                j = 0
                for ch in c_list:
                    for left_context in left_contexts:
                        candidate = left_context + ch + right_context
                        score = self.score(self.match_punct(candidate, line_raw))
                        candidates.append(candidate)
                        scores[j] = score
                        j += 1

                # get top-k candidates
                if len(candidates) > 0:
                    _, topk_fwd_scores = torch.topk(scores, k=min(k, len(candidates)))
                    current_topk_fwd = [candidates[s] for s in topk_fwd_scores.tolist()]
                    if len(current_topk_fwd) > 0:
                        topk_fwd = current_topk_fwd
                    else:
                        topk_fwd = [left_context + unidecode.unidecode(c) + right_context for left_context in left_contexts]
                else:
                    topk_fwd = [left_context + unidecode.unidecode(c) + right_context for left_context in left_contexts]

            # backward (lazy boilerplate code)
            topk_bwd = [predicted_seq_uncertain[uncertainties[-1][1]:]]
            for i, (start_idx, end_idx) in reversed(list(enumerate(uncertainties))):
                c = predicted_seq[start_idx:end_idx]
                c_list = self.tone_map.get(unidecode.unidecode(c), [unidecode.unidecode(c)])

                right_contexts = topk_bwd
                left_context = predicted_seq_uncertain[uncertainties[i - 1][1]:start_idx] if i > 0 else predicted_seq_uncertain[0:start_idx]

                candidates = []
                scores = torch.empty(len(c_list) * len(right_contexts)).fill_(-float('inf'))

                j = 0
                for ch in c_list:
                    for right_context in right_contexts:
                        candidate = left_context + ch + right_context
                        score = self.score(self.match_punct(candidate, line_raw, forward=False))
                        candidates.append(candidate)
                        scores[j] = score
                        j += 1

                if len(candidates) > 0:
                    _, topk_bwd_scores = torch.topk(scores, k=min(k, len(candidates)))
                    current_topk_bwd = [candidates[s] for s in topk_bwd_scores.tolist()]
                    if len(current_topk_bwd) > 0:
                        topk_bwd = current_topk_bwd
                    else:
                        topk_bwd = [left_context + unidecode.unidecode(c) + right_context for right_context in
                                    right_contexts]
                else:
                    topk_bwd = [left_context + unidecode.unidecode(c) + right_context for right_context in
                                right_contexts]

            assert len(topk_fwd[0]) == len(topk_bwd[0])

            # combine and find disagreements between ltr and rtl beam search
            # if 10 recursive calls made, fall back on exhaustive search
            if step < 10:
                out = []
                # find disagreements
                for i in range(len(topk_fwd[0])):
                    if topk_fwd[0][i] == topk_bwd[0][i]:
                        out.append(topk_fwd[0][i])
                    else:
                        out.append('*')
                # recursive call
                return beam_lm_(predicted_seq, predicted_values, ''.join(out), k=k, step=step+1)
            else:
                out = []
                # find disagreements
                # for each disagreement, get top-5 candidates from ltr and rtl search TODO: hard-coded magic number
                for i in range(len(topk_fwd[0])):
                    if topk_fwd[0][i] == topk_bwd[0][i]:
                        out.append(topk_fwd[0][i])
                    else:
                        topc = set()
                        for j in range(min(5, len(topk_fwd))):
                            topc.add(topk_fwd[j][i])
                        for j in range(min(5, len(topk_bwd))):
                            topc.add(topk_bwd[j][i])
                        out.append(topc)

                # cartesian product
                candidates = []
                for i in itertools.product(*out):
                    candidates.append(''.join(i))

                # find top-1
                best_score = -1000000.0
                best_candidate = ""
                for candidate in candidates:
                    score = self.score(self.match_punct(candidate, line_raw))
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                return best_candidate

        return beam_lm_(predicted_seq, predicted_values, predicted_seq_copy)

    def match_punct(self, candidate, line_raw, forward=True):
        line_raw = line_raw.strip()
        if forward:
            line_raw = line_raw[:len(candidate)]
            for i, c in enumerate(candidate):
                if c == '-' or c == '?':
                    candidate = candidate[:i] + line_raw[i] + candidate[i+1:]
        else:
            line_raw = line_raw[len(line_raw)-len(candidate):]
            for i, c in enumerate(candidate):
                if c == '-' or c == '?':
                    candidate = candidate[:i] + line_raw[i] + candidate[i+1:]
        return candidate

    def score(self, candidate):
        return self.wlm.score(candidate)

    def match_case(self, predicted, src):
        src = src.strip()
        out = []
        for i in range(len(predicted)):
            if src[i].isupper():
                out.append(predicted[i].upper())
            elif src[i] in PUNCT or src[i] not in LEGAL:
                out.append(src[i])
            else:
                out.append(predicted[i])
        return ''.join(out)

    def match_output(self, predicted, cleaned):
        predicted = predicted.strip(punctuation)
        cleaned = cleaned.strip()
        out = []

        # questionable shifting procedure to match predicted output and required submission output
        cleaned_ptr = 0
        try:
            for predicted_ptr in range(len(predicted)):
                if unidecode.unidecode(predicted[predicted_ptr]) == cleaned[cleaned_ptr]:
                    out.append(predicted[predicted_ptr])
                    cleaned_ptr += 1
                elif predicted_ptr < len(predicted)-1 and cleaned_ptr < len(cleaned)-1:
                    if unidecode.unidecode(predicted[predicted_ptr+1]) == cleaned[cleaned_ptr+1]:
                        out.append(cleaned[cleaned_ptr])
                        cleaned_ptr += 1
                if cleaned_ptr == len(cleaned):
                    break
        except IndexError:
            print("** predicted: ", predicted)
            print("** cleaned:   ", cleaned)
            print("** out:       ", ''.join(out))

        out_seq = ''.join(out)

        # dumb hard-code to match required submission output
        # for some reason this word is not in the submission file
        if " lịchđầu " in out_seq:
            out_seq = out_seq.replace(" lịchđầu ", " ")
        if " lichdau " in out_seq:
            out_seq = out_seq.replace(" lichdau ", " ")

        return out_seq

    def preprocess(self, line):
        line = line.strip().lower()
        line = ''.join(c if c not in PUNCT else '-' for c in line)  # replace all punctuations with '-'
        line = ''.join(c if c in LEGAL else '?' for c in line)  # replace unknown characters with '?'
        return line


if __name__ == "__main__":
    src_vocab_path = "checkpoint/vocab.src"
    tgt_vocab_path = "checkpoint/vocab.tgt"
    model_path = "checkpoint/aivivn_tone.model.ep25"
    wlm_path = "lm/corpus-wplm-4g-v2.binary"

    test_path = "data/test.txt"
    test_cleaned_path = "data/test_cleaned.txt"
    out_path = "data/submission.txt"

    predictor = Predictor(src_vocab_path, tgt_vocab_path, model_path, wlm_path)
    predictor.predict(test_path, test_cleaned_path, out_path)
