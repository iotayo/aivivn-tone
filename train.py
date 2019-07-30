import os
import gc
import random
import torch
import dill
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torchtext.data import BucketIterator
from dataset import Seq2SeqDataset, PAD, tgt_field_name
from model import Encoder, Decoder, Seq2SeqConcat
from cyclic_lr import CyclicLR
from visualization import Visualization


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Trainer:
    def __init__(self, src_vocab, tgt_vocab,
                 max_len=300, hidden_size=300, n_layers=2, clip=5, n_epochs=30):
        # hyper-parameters
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.clip = clip
        self.n_epochs = n_epochs

        # vocab
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_idx = self.src_vocab.stoi[PAD]

        # prepare model
        self.encoder = Encoder(self.src_vocab, self.max_len, self.hidden_size, self.n_layers)
        self.decoder = Decoder(self.tgt_vocab, self.max_len, self.hidden_size * 2, self.n_layers)
        self.reverse_decoder = Decoder(self.tgt_vocab, self.max_len, self.hidden_size * 2, self.n_layers, reverse=True)
        self.model = Seq2SeqConcat(self.encoder, self.decoder, self.reverse_decoder, self.pad_idx)
        self.model.to(device)
        print(self.model)
        print("Total parameters:", sum([p.nelement() for p in self.model.parameters()]))

        # initialize weights
        for name, param in self.model.named_parameters():
            if "lstm.bias" in name:
                # set lstm forget gate to 1 (Jozefowicz et al., 2015)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
            elif "lstm.weight" in name:
                nn.init.xavier_uniform_(param)

        # prepare loss function; don't calculate loss on PAD tokens
        self.criterion = nn.NLLLoss(ignore_index=self.pad_idx)

        # prepare optimizer and scheduler
        self.optimizer = Adam(self.model.parameters())
        self.scheduler = CyclicLR(self.optimizer, base_lr=0.00001, max_lr=0.00005,
                                  step_size_up=4000, step_size_down=4000,
                                  mode="triangular", gamma=1.0, cycle_momentum=False)

        # book keeping vars
        self.global_iter = 0
        self.global_numel = []
        self.global_loss = []
        self.global_acc = []

        # visualization
        self.vis_loss = Visualization(env_name="aivivn_tone", xlabel="step", ylabel="loss", title="loss (mean per 300 steps)")
        self.vis_acc = Visualization(env_name="aivivn_tone", xlabel="step", ylabel="acc", title="training accuracy (mean per 300 steps)")

    def train(self, train_iterator, val_iterator, start_epoch=0, print_every=100):
        for epoch in range(start_epoch, self.n_epochs):
            self._train_epoch(epoch, train_iterator, train=True, print_every=print_every)
            self.save(epoch)

            # evaluate on validation set after each epoch
            with torch.no_grad():
                self._train_epoch(epoch, val_iterator, train=False, print_every=print_every)

    def train_in_parts(self, train_parts, val, val_iterator, batch_size, start_epoch=0, print_every=100):
        for epoch in range(start_epoch, self.n_epochs):
            # shuffle data each epoch
            random.shuffle(train_parts)

            for train_src_, train_tgt_ in train_parts:
                # create train dataset
                print("Training part [{}] with target [{}]...".format(train_src_, train_tgt_))
                train_ = Seq2SeqDataset.from_file(train_src_, train_tgt_, share_fields_from=val)

                # create iterator
                train_iterator_ = BucketIterator(dataset=train_, batch_size=batch_size,
                                                 sort=False, sort_within_batch=True,
                                                 sort_key=lambda x: len(x.src),
                                                 shuffle=True, device=device)
                # train
                self._train_epoch(epoch, train_iterator_, train=True, print_every=print_every)

                # clean
                del train_
                del train_iterator_
                gc.collect()

            # save
            self.save(epoch)

            # evaluate on validation set after each epoch
            with torch.no_grad():
                self._train_epoch(epoch, val_iterator, train=False, print_every=print_every)

    def resume(self, train_iterator, val_iterator, save_path):
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        self.train(train_iterator, val_iterator, start_epoch)

    def resume_in_parts(self, train_parts, val, val_iterator, batch_size, save_path):
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        self.train_in_parts(train_parts, val, val_iterator, batch_size, start_epoch=start_epoch)

    def _train_epoch(self, epoch, batch_iterator, train=True, print_every=100):
        if train:
            self.model.train()
        else:
            self.model.eval()
            print("***Evaluating on validation set***")

        total_loss = 0
        total_correct = 0
        total_numel = 0
        total_iter = 0
        num_batch = len(batch_iterator)

        for i, batch in enumerate(batch_iterator):
            # forward propagation
            # (batch, seq_len, tgt_vocab_size)
            if train:
                # crude annealing teacher forcing
                teacher_forcing = 0.5
                if epoch == 0:
                    teacher_forcing = max(0.5, (num_batch - total_iter) / num_batch)
                output, reverse_output, combined_output = self.model(batch, mask_softmax=0.5, teacher_forcing=teacher_forcing)
            else:
                output, reverse_output, combined_output = self.model(batch, mask_softmax=1.0, teacher_forcing=1.0)

            # (batch, seq_len)
            target = getattr(batch, tgt_field_name)

            # reshape to calculate loss
            output = output.view(-1, output.size(-1))
            reverse_output = reverse_output.view(-1, reverse_output.size(-1))
            combined_output = combined_output.view(-1, combined_output.size(-1))
            target = target.view(-1)

            # calculate loss
            loss = self.criterion(output, target) + self.criterion(reverse_output, target) + self.criterion(combined_output, target)

            # backprop
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.scheduler.step()

            # calculate accuracy
            correct = output.argmax(dim=-1).eq(target).sum().item()
            r_correct = reverse_output.argmax(dim=-1).eq(target).sum().item()
            c_correct = combined_output.argmax(dim=-1).eq(target).sum().item()

            # summarize for each batch
            total_loss += loss.item()
            total_correct += c_correct
            total_numel += target.numel()
            total_iter += 1

            # add to global summary
            if train:
                self.global_iter += 1
                self.global_numel.append(target.numel())
                self.global_loss.append(loss.item())
                self.global_acc.append(c_correct)

                # visualize
                if self.global_iter == 1:
                    self.vis_loss.plot_line(self.global_loss[0], 1)
                    self.vis_acc.plot_line(self.global_acc[0]/total_numel, 1)

                # update graph every 10 iterations
                if self.global_iter % 10 == 0:
                    # moving average of most recent 300 losses
                    moving_avg_loss = sum(self.global_loss[max(0, len(self.global_loss)-300):]) / min(300.0, self.global_iter)
                    moving_avg_acc = sum(self.global_acc[max(0, len(self.global_acc) - 300):]) / sum(self.global_numel[max(0, len(self.global_numel) - 300):])

                    # visualize
                    self.vis_loss.plot_line(moving_avg_loss, self.global_iter)
                    self.vis_acc.plot_line(moving_avg_acc, self.global_iter)

            # print
            if i % print_every == 0:
                template = "epoch = {}  iter = {}  loss = {:5.3f}  correct = {:6.3f}  r_correct = {:6.3f}  c_correct = {:6.3f}"
                print(template.format(epoch,
                                      i,
                                      loss.item(),
                                      correct / target.numel() * 100.0,
                                      r_correct / target.numel() * 100.0,
                                      c_correct / target.numel() * 100.0))

        # summarize for each epoch
        template = "EPOCH = {}  AVG_LOSS = {:5.3f}  AVG_CORRECT = {:6.3f}\n"
        print(template.format(epoch,
                              total_loss / total_iter,
                              total_correct / total_numel * 100.0))

    def save(self, epoch, save_path="checkpoint"):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_iter": self.global_iter
        }, os.path.join(save_path, "aivivn_tone.model.ep{}".format(epoch)))


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(train_src, train_tgt, val_src, val_tgt, batch_size=64, save_path="checkpoint"):
    # prepare dataset
    print("Reading data...")
    train = Seq2SeqDataset.from_file(train_src, train_tgt)

    print("Building vocab...")
    train.build_vocab(max_size=300)

    val = Seq2SeqDataset.from_file(val_src, val_tgt, share_fields_from=train)

    src_vocab = train.src_field.vocab
    tgt_vocab = train.tgt_field.vocab

    # save vocab
    with open(os.path.join(save_path, "vocab.src"), "wb") as f:
        dill.dump(src_vocab, f)
    with open(os.path.join(save_path, "vocab.tgt"), "wb") as f:
        dill.dump(tgt_vocab, f)

    print("Source vocab size:", len(src_vocab))
    print("Target vocab size:", len(tgt_vocab))

    # data iterator
    # keep sort=False and shuffle=False to speed up training and reduce memory usage
    train_iterator = BucketIterator(dataset=train, batch_size=batch_size,
                                    sort=False, sort_within_batch=True,
                                    sort_key=lambda x: len(x.src),
                                    shuffle=False, device=device)
    val_iterator = BucketIterator(dataset=val, batch_size=batch_size, train=False,
                                  sort=False, sort_within_batch=True,
                                  sort_key=lambda x: len(x.src),
                                  shuffle=False, device=device)

    return src_vocab, tgt_vocab, train_iterator, val_iterator


def load_data_in_parts(train_src, train_tgt, val_src, val_tgt, batch_size=64, save_path="checkpoint"):
    # prepare dataset
    print("Reading data...")
    val = Seq2SeqDataset.from_file(val_src, val_tgt)

    print("Building vocab...")
    val.build_vocab(max_size=300)

    src_vocab = val.src_field.vocab
    tgt_vocab = val.tgt_field.vocab

    # save vocab
    with open(os.path.join(save_path, "vocab.src"), "wb") as f:
        dill.dump(src_vocab, f)
    with open(os.path.join(save_path, "vocab.tgt"), "wb") as f:
        dill.dump(tgt_vocab, f)

    print("Source vocab size:", len(src_vocab))
    print("Target vocab size:", len(tgt_vocab))

    # data iterator
    # keep sort=False and shuffle=False to speed up training and reduce memory usage
    val_iterator = BucketIterator(dataset=val, batch_size=batch_size, train=False,
                                  sort=False, sort_within_batch=True,
                                  sort_key=lambda x: len(x.src),
                                  shuffle=False, device=device)

    return src_vocab, tgt_vocab, list(zip(train_src, train_tgt)), val, val_iterator, batch_size


if __name__ == "__main__":
    train_src = ["data/train.src.0", "data/train.src.1", "data/train.src.2", "data/train.src.3"]
    train_tgt = ["data/train.tgt.0", "data/train.tgt.1", "data/train.tgt.2", "data/train.tgt.3"]
    val_src = "data/val.src"
    val_tgt = "data/val.tgt"
    # src_vocab_path = "checkpoint/vocab.src"
    # tgt_vocab_path = "checkpoint/vocab.tgt"

    # set random seeds
    set_seeds(420)

    # load vocab
    # with open(src_vocab_path, "rb") as f:
    #     src_vocab = dill.load(f)
    # with open(tgt_vocab_path, "rb") as f:
    #     tgt_vocab = dill.load(f)

    # load data
    src_vocab, tgt_vocab, train_parts, val, val_iterator, batch_size = load_data_in_parts(train_src, train_tgt, val_src, val_tgt)

    # prepare trainer
    trainer = Trainer(src_vocab, tgt_vocab)

    # train
    trainer.train_in_parts(train_parts, val, val_iterator, batch_size)
    # trainer.resume_in_parts(train_parts, val, val_iterator, batch_size, save_path="checkpoint/aivivn_tone.model.ep19")
