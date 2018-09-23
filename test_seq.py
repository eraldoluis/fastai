from fastai.column_seq_data import ColumnarSeqModelData
from fastai.structured import *

path = '/home/eraldo/lia/src/kddbr-2018/input/'

# Train data.
df_clean = pd.read_csv(os.path.join(path, 'all_clean.csv'), index_col=0)

print('Shape:', df_clean.shape)

print(df_clean.head(2))

# Categorical features.
cat_ftrs = ['field', 'age', 'type', 'harvest_month']
# Continuous features.
contin_ftrs = [f for f in df_clean.columns if f not in (['production', 'Id', 'index'] + cat_ftrs)]
print('Contin ftrs:', contin_ftrs)

# Remove unused features.
df_clean = df_clean[cat_ftrs + contin_ftrs + ['production', 'Id']]
print(df_clean.head(2))

val_year = 2011
test_year = 2012

train_mask = (df_clean.harvest_year < val_year)
val_mask = ((df_clean.harvest_year >= val_year) & (df_clean.harvest_year < test_year))
test_mask = (df_clean.harvest_year >= test_year)

df_train = df_clean[train_mask]
df_val = df_clean[val_mask]
df_test = df_clean[test_mask]

def getSeqs(df):
    df = df.reset_index(drop=True)
    df['year_plant'] = df.harvest_year - df.age
    seqs_lim = [0]
    row = df.iloc[0]
    vals = (row.field, row.type, row.year_plant)
    for i, row in df.iterrows():
        if (row.field, row.type, row.year_plant) != vals:
            seqs_lim.append(i)
            vals = (row.field, row.type, row.year_plant)
    return seqs_lim + [len(df)]

# Compute list of sequence limits.
seqs_train = getSeqs(df_train)
seqs_val = getSeqs(df_val)
seqs_test = getSeqs(df_test)

def convType(df, cat_ftrs, contin_ftrs):
    # Inform pandas which features are categorical ...
    for v in cat_ftrs:
        df.loc[:,v] = df[v].astype('category').cat.as_ordered()
    # ... and which are continuous.
    for v in contin_ftrs:
        df.loc[:,v] = df[v].astype('float32')

for df in [df_train, df_val, df_test, df_clean]:
    convType(df, cat_ftrs, contin_ftrs)

# Compute list of embedding sizes.
cat_sz = [(c, len(df_clean[c].cat.categories) + 1) for c in cat_ftrs]
print(cat_sz)
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
print(emb_szs)

df_train, y_train, nas, mapper = proc_df(df_train, y_fld='production', do_scale=True, skip_flds=['Id'])

df_val, y_val, nas, mapper = proc_df(df_val, y_fld='production', do_scale=True, mapper=mapper, na_dict=nas, skip_flds=['Id'])

test_ids = df_test.Id

df_test, _, nas, mapper = proc_df(df_test, y_fld='production', do_scale=True, mapper=mapper, na_dict=nas, skip_flds=['Id'])

# def groupBySeqs(*dfs):
#     grps = []
#     for df in dfs:
#         df["year_plant"] = df["harvest_year"] - df["age"]
#         gs = df.groupby(['field', 'type', 'year_plant'], sort=False)
#         df.drop(columns=['year_plant'], inplace=True)
#         grps.append([g for _, g in gs])
#     return grps
#
# dfs_train, dfs_val, dfs_test = groupBySeqs(df_train, df_val, df_test)

# def splitY(dfs):
#     ys = [df['production'] for df in dfs]
#     for df in dfs:
#         df.drop(columns=['production'], inplace=True)
#     return dfs, ys
#
# dfs_train, ys_train = splitY(dfs_train)
# dfs_val, ys_val = splitY(dfs_val)
# dfs_test, ys_test = splitY(dfs_test)

# path, trn_dfs, val_dfs, trn_ys, val_ys, cat_flds, bs=64, is_reg=True, is_multi=False, test_dfs=None, shuffle=True

md = ColumnarSeqModelData.from_data_frames(path, # path for data saving
                                           seqs_train,
                                           df_train, # training set
                                           y_train,
                                           seqs_val,
                                           df_val, # validation set
                                           y_val, # output variable for the validation set
                                           cat_flds=cat_ftrs, # categorical features
                                           is_reg=True, # not regression
                                           is_multi=False, # multi-label problem
                                           test_seqs_lim=seqs_test,
                                           test_df=df_test)

# dropout rate
dr = 0.3

n_cont = len(df_train.columns) - len(cat_ftrs)

learner_params = {
    "emb_szs": emb_szs, # embedding sizes
    "n_cont": n_cont, # num continuous inputs
    "emb_drop": dr, # embeddings dropout probability
    "out_sz": 1, # output size
    "szs": [100, 100], # sizes of fully-connected layers
    "drops": [dr, dr], # dropout probabilities after each FC layer
    "lstm_hidden_size": 100, # size of the LSTM hidden states
    "lstm_num_layers": 2, # number of LSTM layers
    "lstm_dropout": dr, # LSTM dropout
    "use_bn": False # batch normalization
}

def mae(y_true, y_pred):
    return metrics.mean_absolute_error(y_true.squeeze(), y_pred.squeeze())

m = md.get_learner(**learner_params)

m.summary()

#m.summary()
#m.lr_find()
#m.sched.plot()
lr=1e-3
m.fit(lr, n_cycle=10, cycle_len=3, metrics=[mae])
