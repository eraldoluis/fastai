import seaborn as sns

sns.set(style="whitegrid")

from fastai.structured import *
from fastai.column_seq_data import *

path = '/home/eraldo/lia/src/kddbr-2018/input'

# Train data.
df_all = pd.read_csv(os.path.join(path, 'all_clean.csv'), index_col=0)

print('Shape:', df_all.shape)

# Read field data.
df_field = pd.read_csv(os.path.join(path, 'field-0.csv'))
df_field['field'] = 0
for i in range(1, 28):
    _df_field = pd.read_csv(os.path.join(path, 'field-{}.csv'.format(i)))
    _df_field['field'] = i
    df_field = pd.concat([df_field, _df_field])

df_soil = pd.read_csv(os.path.join(path, 'soil_data.csv'))

df_all.drop(columns=['production'], inplace=True)

train_year = 2004
val_year = 2017

train_mask = ((df_all.harvest_year >= train_year) & (df_all.harvest_year < val_year))
val_mask = (df_all.harvest_year >= val_year)

df_train = df_all[train_mask].copy()
df_val = df_all[val_mask].copy()

def getSeqs(df):
    df = df.copy()

    df['plant_year'] = df.harvest_year - df.age

    grps = df.groupby(['field', 'type', 'plant_year'])
    first_hys = grps.min()['harvest_year']
    last_hys = grps.max()['harvest_year']

    dfs = []

    seqs_lim = []
    seq_idx = 0

    for ((field_, type_, plant_year_), grp), first_hy, last_hy in zip(grps, first_hys, last_hys):
        # Generate data (even without production variable) from one year before the first year of the group
        # up to the last year present in the group.
        hys = np.arange(first_hy - 1, last_hy + 1)

        hyears = grp.harvest_year.unique()

        # Add limits of the sequences in this group. Do not add years that have no production value.
        # However, if a year has more than zero production values, it is added to the sequences.
        #   seq_idx is the next index (len of the current df).
        #   12 is due to the addition of the previous year of the first year in this sequence.
        lims = [seq_idx + 12 + (hy - first_hy) * 12 for hy in hyears]
        seqs_lim += lims

        df1 = pd.DataFrame(data={'field': field_,
                                 'type': type_,
                                 'plant_year': plant_year_,
                                 'harvest_year': hys})
        df2 = pd.DataFrame(data={'field': field_,
                                 'harvest_month': np.arange(1, 13)})
        df3 = pd.merge(df1, df2, on='field', how='outer')

        df_grp = pd.merge(grp, df3, on=['field', 'type', 'plant_year', 'harvest_year', 'harvest_month'], how='outer',
                          sort=True)

        seq_idx += len(df_grp)

        dfs.append(df_grp)

    df = pd.concat(dfs)

    df['age'] = df.harvest_year - df.plant_year

    df.drop(columns=['plant_year'], inplace=True)

    df = pd.merge(df, df_field,
                  left_on=['field', 'harvest_year', 'harvest_month'],
                  right_on=['field', 'year', 'month'],
                  how='left')

    df.drop(columns=['year', 'month'], inplace=True)

    df = pd.merge(df, df_soil, on='field', how='left').reset_index(drop=True)

    return df, seqs_lim


df_train_ftrs, seqs_train = getSeqs(df_train)
df_val_ftrs, seqs_val = getSeqs(df_val)


# Categorical features.
cat_ftrs = ['field', 'age', 'type', 'harvest_month']
# Continuous features.
contin_ftrs = [f for f in df_train_ftrs.columns if f not in (['Id'] + cat_ftrs)]
print(contin_ftrs)


def convType(df, cat_ftrs, contin_ftrs):
    # Inform pandas which features are categorical ...
    for v in cat_ftrs:
        df[v] = df[v].astype('category').cat.as_ordered()
    # ... and which are continuous.
    for v in contin_ftrs:
        df[v] = df[v].astype('float32')


df_all_ftrs = pd.concat([df_train_ftrs, df_val_ftrs])

convType(df_all_ftrs, cat_ftrs, contin_ftrs)

# Compute list of embedding sizes.
cat_sz = [(c, len(df_all_ftrs[c].cat.categories) + 1) for c in cat_ftrs]
print(cat_sz)
emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in cat_sz]
print(emb_szs)

df_all_ftrs, _, _, _ = proc_df(df_all_ftrs, y_fld=None, do_scale=True, skip_flds=['Id'])

n_cont = len(df_all_ftrs.columns) - len(cat_ftrs)

len_train = len(df_train_ftrs)

df_train_ftrs = df_all_ftrs.iloc[:len_train].copy()
df_val_ftrs = df_all_ftrs.iloc[len_train:].copy()


# In[11]:


def gen_splits(cols, df, y, seqs):
    df_splits = []
    y_splits = []
    seq_splits = []

    idx_offset = 0

    for col in cols:
        # Create a new split for the categorical feature 'col'.
        df_col = df.copy()
        # Remove categorical feature value for this split.
        df_col[col] = 0
        # Add to list of splits.
        df_splits.append(df_col)
        # Copy the output variables.
        y_splits.append(y.copy())

        # Copy the sequences (add the index offset).
        seq_splits += [s + idx_offset for s in seqs]

        idx_offset += len(df_col)

    # Concatenate all splits.
    y = pd.concat(y_splits)
    df = pd.concat(df_splits)

    return df, y, seq_splits


def get_dummies(df_train_, seq_train, df_val_, seq_val):
    cols = ['field', 'age', 'type', 'harvest_month']

    df = pd.concat([df_train_, df_val_])

    # Create dummy values for categorical features.
    y = pd.get_dummies(df.loc[:, cols], columns=cols)

    y_train = y[:len(df_train_)]
    y_val = y[len(df_train_):]

    df_train_, y_train, seq_train = gen_splits(cols, df_train_, y_train, seq_train)
    df_val_, y_val, seq_val = gen_splits(cols, df_val_, y_val, seq_val)

    return df_train_, y_train, seq_train, df_val_, y_val, seq_val


# In[12]:


df_train_dum, y_train_dum, seqs_train_dum, df_val_dum, y_val_dum, seqs_val_dum = \
    get_dummies(df_train_ftrs, seqs_train, df_val_ftrs, seqs_val)


md = ColumnarSeqModelData.from_data_frames(path,  # path for data saving
                                           seqs_train_dum,  # limits of training sequences
                                           df_train_dum,  # training set
                                           y_train_dum.astype(np.float32),  # output variable for the training set
                                           seqs_val_dum,  # limits of validation sequences
                                           df_val_dum,  # validation set
                                           y_val_dum.astype(np.float32),  # output variable for the validation set
                                           cat_flds=cat_ftrs,  # categorical features
                                           is_reg=False,  # not regression
                                           is_multi=True,  # multi-label problem
                                           # test_seqs_lim=seqs_test,  # limits of test sequences
                                           # test_df=df_test_proc,  # test set
                                           bs=4)  # batch size


# In[15]:


def mae(y_pred, y_true):
    y_true = y_true.view((-1)).cpu().numpy()
    y_pred = y_pred.view((-1)).cpu().numpy()
    # Mask out nan values.
    m = np.invert(np.isnan(y_true))
    return metrics.mean_absolute_error(y_true[m], y_pred[m])


# In[18]:


# dropout rate
dr = 0.0

learner_params = {
    "emb_szs": emb_szs,  # embedding sizes
    "n_cont": n_cont,  # num continuous inputs
    "emb_drop": dr,  # embeddings dropout probability
    "out_sz": y_train_dum.shape[-1],  # output size
    "szs": [300],  # sizes of fully-connected layers
    "drops": [dr],  # dropout probabilities after each FC layer
    "lstm_hidden_size": 100,  # size of the LSTM hidden states
    "lstm_num_layers": 2,  # number of LSTM layers
    "lstm_dropout": dr,  # LSTM dropout
    "use_bn": False,  # batch normalization
    "y_range": [0.0, 1.0]
}

m = md.get_learner(**learner_params)

m.lr_find()  # start_lr=1e-4, end_lr=1e20)
m.sched.plot()

# In[22]:


# dropout rate
dr = 0.5

learner_params = {
    "emb_szs": emb_szs,  # embedding sizes
    "n_cont": n_cont,  # num continuous inputs
    "emb_drop": dr,  # embeddings dropout probability
    "out_sz": 1,  # output size
    "szs": [300],  # sizes of fully-connected layers
    "drops": [dr],  # dropout probabilities after each FC layer
    "lstm_hidden_size": 100,  # size of the LSTM hidden states
    "lstm_num_layers": 2,  # number of LSTM layers
    "lstm_dropout": dr,  # LSTM dropout
    "use_bn": False,  # batch normalization
    "y_range": [0.0, 1.0]
}

m = md.get_learner(**learner_params)

lr = 1e-3
m.fit(lr, n_cycle=10, cycle_len=3, metrics=[mae])

# In[17]:


# dropout rate
dr = 0.1

learner_params = {
    "emb_szs": emb_szs,  # embedding sizes
    "n_cont": n_cont,  # num continuous inputs
    "emb_drop": dr,  # embeddings dropout probability
    "out_sz": 1,  # output size
    "szs": [300],  # sizes of fully-connected layers
    "drops": [dr],  # dropout probabilities after each FC layer
    "lstm_hidden_size": 100,  # size of the LSTM hidden states
    "lstm_num_layers": 2,  # number of LSTM layers
    "lstm_dropout": dr,  # LSTM dropout
    "use_bn": False,  # batch normalization
    "y_range": [0.0, 1.0]
}

m = md.get_learner(**learner_params)

lr = 1e-3
m.fit(lr, n_cycle=10, cycle_len=3, metrics=[mae])

# In[33]:


# dropout rate
dr = 0.3

learner_params = {
    "emb_szs": emb_szs,  # embedding sizes
    "n_cont": n_cont,  # num continuous inputs
    "emb_drop": dr,  # embeddings dropout probability
    "out_sz": 1,  # output size
    "szs": [100, 50],  # sizes of fully-connected layers
    "drops": [dr, dr],  # dropout probabilities after each FC layer
    "lstm_hidden_size": 30,  # size of the LSTM hidden states
    "lstm_num_layers": 2,  # number of LSTM layers
    "lstm_dropout": dr,  # LSTM dropout
    "use_bn": False,  # batch normalization
    "y_range": [0.0, 1.0]
}

m = md.get_learner(**learner_params)

lr = 1e-3
m.fit(lr, n_cycle=20, cycle_len=3, metrics=[mae])

# ## Submission file

# In[ ]:


# dropout rate
dr = 0.3

learner_params = {
    "emb_szs": emb_szs,  # embedding sizes
    "n_cont": len(df.columns) - len(cat_ftrs),  # num continuous inputs
    "emb_drop": dr,  # embeddings dropout probability
    "out_sz": 1,  # output size
    "szs": [300, 100],  # sizes of fully-connected layers
    "drops": [dr, dr],  # dropout probabilities after each FC layer
    "use_bn": False  # batch normalization
}

m = md.get_learner(**learner_params)

m.load('finetune-all-clean-trn011')

# In[ ]:


from datetime import datetime

# Make prediction.
pred = m.predict(is_test=True).squeeze()

now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

submission_file = '../submissions/{}.submission.csv'.format(now)
model_file = '../submissions/{}.model'.format(now)

# Create a submission file.
with open(submission_file, 'w') as f:
    f.write("Id,production\n")
    for _id, _pred in zip(test_ids.values, pred):
        f.write("{},{}\n".format(_id, min(1.0, max(0.0, _pred))))

# Save model.
save_model(m.model, model_file)
