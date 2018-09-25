from torch.nn import LSTM

from .imports import *
from .torch_imports import *
from .dataset import *
from .learner import *


class ColumnarSeqDataset(Dataset):
    def __init__(self, seqs_lim, cats, conts, y, is_reg, is_multi):
        self.seqs_lim = seqs_lim
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats  else np.zeros((n, 1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n, 1))
        self.y = np.zeros((n, 1)) if y is None else y
        if is_reg:
            self.y = self.y[:, None]
        self.is_reg = is_reg
        self.is_multi = is_multi

        if is_reg:
            y_type = np.float32 if is_reg else np.int64
            y_nan_val = np.nan if is_reg else -1
        else:
            y_type = np.int64
            y_nan_val = -1

        self.y_nan = np.asarray([[y_nan_val] * self.y.shape[1]] * 12, dtype=y_type)

    def __len__(self): return len(self.seqs_lim) - 1

    def __getitem__(self, idx):
        beg = self.seqs_lim[idx]
        end = beg + 12
        return [self.cats[beg - 12:end], self.conts[beg - 12:end], np.concatenate((self.y_nan, self.y[beg:end]))]

    @classmethod
    def from_data_frames(cls, seqs_lim, df_cat, df_cont, y=None, is_reg=True, is_multi=False):
        cat_cols = [c.values for n, c in df_cat.items()]
        cont_cols = [c.values for n, c in df_cont.items()]
        return cls(seqs_lim, cat_cols, cont_cols, y, is_reg, is_multi)

    @classmethod
    def from_data_frame(cls, seqs_lim, df, cat_flds, y=None, is_reg=True, is_multi=False):
        return cls.from_data_frames(seqs_lim, df[cat_flds], df.drop(cat_flds, axis=1), y, is_reg, is_multi)


class ColumnarSeqModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, bs, test_ds=None, shuffle=True):
        test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=1) if test_ds is not None else None
        train_dl = DataLoader(trn_ds, bs, shuffle=shuffle, num_workers=1)
        val_dl = DataLoader(val_ds, bs, shuffle=False, num_workers=1)
        super().__init__(path, train_dl, val_dl, test_dl)

    # @classmethod
    # def from_arrays(cls, path, val_idxs, xs, y, is_reg=True, is_multi=False, bs=64, test_xs=None, shuffle=True):
    #     ((val_xs, trn_xs), (val_y, trn_y)) = split_by_idx(val_idxs, xs, y)
    #     test_ds = PassthruDataset(*(test_xs.T), [0] * len(test_xs), is_reg=is_reg, is_multi=is_multi) if test_xs is not None else None
    #     return cls(path, PassthruDataset(*(trn_xs.T), trn_y, is_reg=is_reg, is_multi=is_multi),
    #                PassthruDataset(*(val_xs.T), val_y, is_reg=is_reg, is_multi=is_multi),
    #                bs=bs, shuffle=shuffle, test_ds=test_ds)

    @classmethod
    def from_data_frames(cls, path, trn_seqs_lim, trn_df, trn_y, val_seqs_lim, val_df, val_y, cat_flds, bs=1,
                         is_reg=True, is_multi=False, test_seqs_lim=None, test_df=None, shuffle=True):
        trn_ds = ColumnarSeqDataset.from_data_frame(trn_seqs_lim, trn_df, cat_flds, trn_y, is_reg, is_multi)
        val_ds = ColumnarSeqDataset.from_data_frame(val_seqs_lim, val_df, cat_flds, val_y, is_reg, is_multi)
        test_ds = ColumnarSeqDataset.from_data_frame(test_seqs_lim, test_df, cat_flds, None, is_reg,
                                                     is_multi) if test_df is not None else None
        return cls(path, trn_ds, val_ds, bs, test_ds, shuffle=shuffle)

    # @classmethod
    # def from_data_frame(cls, path, val_idxs, df, y, cat_flds, bs=64, is_reg=True, is_multi=False, test_df=None, shuffle=True):
    #     ((val_df, trn_df), (val_y, trn_y)) = split_by_idx(val_idxs, df, y)
    #     return cls.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg, is_multi, test_df=test_df, shuffle=shuffle)

    def get_learner(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                    lstm_hidden_size, lstm_num_layers, lstm_dropout,
                    y_range=None, use_bn=False, **kwargs):
        model = MixedInputSeqModel(emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                                   lstm_hidden_size, lstm_num_layers, lstm_dropout,
                                   y_range, use_bn, self.is_reg, self.is_multi)
        return StructuredSeqLearner(self, StructuredSeqModel(to_gpu(model)), opt_fn=optim.Adam, **kwargs)


def emb_init(x):
    x = x.weight.data
    sc = 2 / (x.size(1) + 1)
    x.uniform_(-sc, sc)


class MixedInputSeqModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                 lstm_hidden_size, lstm_num_layers, lstm_dropout,
                 y_range=None, use_bn=False, is_reg=True, is_multi=False):
        """

        :param emb_szs: list with the sizes of the embeddings for the categorical features.
        :param n_cont: number of continuous features.
        :param emb_drop: embedding dropout rate.
        :param out_sz: output size.
        :param szs: sizes (and number) of the linear layers between the input features and the LSTM layers.
        :param drops: dropout rates for the linear layers.
        :param lstm_hidden_size: size of the LSTM hidden states.
        :param lstm_num_layers: number of LSTM layers.
        :param lstm_dropout: dropout rate between LSTM layers (not included after the last LSTM layer).
        :param y_range:
        :param use_bn: whether to use batch normalization.
        :param is_reg: whether dealing with a regression problem.
        :param is_multi: whether dealing with a multi-label problem.
        """
        super().__init__()
        for i, (c, s) in enumerate(emb_szs): assert c > 1, f"cardinality must be >=2, got emb_szs[{i}]: ({c},{s})"
        if is_reg == False and is_multi == False: assert out_sz >= 2, "For classification with out_sz=1, use is_multi=True"

        # Embeddings.
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_szs])
        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont

        # Linear layers.
        szs = [n_emb + n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i + 1]) for i in range(len(szs) - 1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: kaiming_normal(o.weight.data)

        # LSTM layer
        self.lstm = LSTM(input_size=szs[-1], hidden_size=lstm_hidden_size,
                         num_layers=lstm_num_layers, dropout=lstm_dropout,
                         batch_first=True)

        self.lstm_drop = nn.Dropout(lstm_dropout)

        # Output weights.
        self.outp = nn.Linear(lstm_hidden_size, out_sz)
        kaiming_normal(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn, self.y_range = use_bn, y_range
        self.is_reg = is_reg
        self.is_multi = is_multi

    def forward(self, x_cat, x_cont):
        """

        :param x_cat: categorical features, shape=(batch, seq, cat_ftr)
        :param x_cont: continuous features, shape=(batch, seq, contin_ftr)
        :return: sequence of predictions, shape=(batch, seq, ?). The third dim is non-existent for regression problems,
            but for multilabel problem it is num_labels.
        """
        if self.n_emb != 0:
            x = [e(x_cat[:, :, i]) for i, e in enumerate(self.embs)]
            x = torch.cat(x, 2)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont) if self.use_bn else x_cont
            x = torch.cat([x, x2], 2) if self.n_emb != 0 else x2
        for l, d, b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x, _ = self.lstm(x)
        x = self.lstm_drop(x)
        x = self.outp(x)
        if not self.is_reg:
            if self.is_multi:
                x = torch.sigmoid(x)
            else:
                x = F.log_softmax(x)
        elif self.y_range:
            x = torch.sigmoid(x)
            x = x * (self.y_range[1] - self.y_range[0])
            x = x + self.y_range[0]
        return x


def mse_loss_nan(input, target):
    # Mask out nan values.
    m = torch.isnan(target) ^ 1
    return F.mse_loss(torch.masked_select(input, m), torch.masked_select(target, m))


def bce_nan(input, target):
    """
    Binary cross entropy that ignores input values equal to -1.

    :param input:
    :param target:
    :return:
    """
    # Mask out -1 values.
    m = torch.eq(target, -1) ^ 1
    input = torch.masked_select(input, m)
    target = torch.masked_select(target, m)
    return F.binary_cross_entropy(input, target)


class StructuredSeqLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data):
        return mse_loss_nan if data.is_reg else bce_nan if data.is_multi else F.nll_loss

    def predict_array(self, x_cat, x_cont):
        self.model.eval()
        return to_np(self.model(to_gpu(V(T(x_cat))), to_gpu(V(T(x_cont)))))

    def summary(self):
        x = [torch.ones(1, 3, self.data.trn_ds.cats.shape[1]).long(), torch.rand(1, 3, self.data.trn_ds.conts.shape[1])]
        return model_summary(self.model, x)


class StructuredSeqModel(BasicModel):
    def get_layer_groups(self):
        m = self.model
        return [m.embs, children(m.lins) + children(m.bns), m.lstm, m.outp]
