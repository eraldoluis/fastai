# coding: utf-8

# # LSTM Modeling
# Use a LSTM model.
# Each group of (field, type, year_plant), where year_plant = harvest_year - age, is considered a sequence.


import seaborn as sns

sns.set(style="whitegrid")

# In[3]:


from fastai.structured import *
from fastai.column_seq_data import *

# ## Train and test datasets
#
# Basic data containing palm tree information

# In[4]:


path = '/home/eraldo/lia/src/kddbr-2018/input'

# Train data.
df_all = pd.read_csv(os.path.join(path, 'all_clean.csv'), index_col=0)

print('Shape:', df_all.shape)

# ## Field Data

# In[5]:


# Read field data.
df_field = pd.read_csv(os.path.join(path, 'field-0.csv'))
df_field['field'] = 0
for i in range(1, 28):
    _df_field = pd.read_csv(os.path.join(path, 'field-{}.csv'.format(i)))
    _df_field['field'] = i
    df_field = pd.concat([df_field, _df_field])

# # Merge with given data.
# df = pd.merge(df, df_field, left_on=['harvest_year', 'harvest_month','field'],
#               right_on=['year', 'month', 'field'], how='inner').reset_index()


# ## Soil Data

# In[6]:


df_soil = pd.read_csv(os.path.join(path, 'soil_data.csv'))
# df = pd.merge(df, df_soil, on='field', how='inner')


# In[7]:


val_year = 2010
test_year = 2012

train_mask = (df_all.harvest_year < val_year)
val_mask = ((df_all.harvest_year >= val_year) & (df_all.harvest_year < test_year))
test_mask = (df_all.harvest_year >= test_year)

df_train = df_all[train_mask].copy()
df_val = df_all[val_mask].copy()
df_test = df_all[test_mask].copy()


# In[8]:


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

        years = grp.harvest_year.unique()

        # Add limits of the sequences in this group. Do not add years that have no production value.
        # However, if a year has more than zero production values, it is added to the sequences.
        #   seq_idx is the next index (len of the current df).
        #   12 is due to the addition of the previous year of the first year in this sequence.
        lims = [seq_idx + 12 + (y - first_hy)*12 for y in years]
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
df_test_ftrs, seqs_test = getSeqs(df_test)

# In[9]:


# def getSeqs(df):
#     df = df.reset_index(drop=True)
#     df['year_plant'] = df.harvest_year - df.age
#     seqs_lim = [0]
#     row = df.iloc[0]
#     vals = (row.field, row.type, row.year_plant)
#     for i, row in df.iterrows():
#         if (row.field, row.type, row.year_plant) != vals:
#             seqs_lim.append(i)
#             vals = (row.field, row.type, row.year_plant)
#     return seqs_lim + [len(df)]

# def getSeqsPerYear(df):
#     df = df.reset_index(drop=True)
#     seqs_lim = [0]
#     row = df.iloc[0]
#     vals = (row.field, row.type, row.harvest_year, row.age)
#     for i, row in df.iterrows():
#         if (row.field, row.type, row.harvest_year, row.age) != vals:
#             seqs_lim.append(i)
#             vals = (row.field, row.type, row.harvest_year, row.age)
#     return seqs_lim + [len(df)]

# # Compute list of sequence limits.
# seqs_train = getSeqsPerYear(df_train)
# seqs_val = getSeqsPerYear(df_val)
# seqs_test = getSeqsPerYear(df_test)


# In[10]:


# Categorical features.
cat_ftrs = ['field', 'age', 'type', 'harvest_month']
# Continuous features.
contin_ftrs = [f for f in df_train_ftrs.columns if f not in (['production', 'Id'] + cat_ftrs)]
print(contin_ftrs)


# In[11]:


def convType(df, cat_ftrs, contin_ftrs):
    # Inform pandas which features are categorical ...
    for v in cat_ftrs:
        df[v] = df[v].astype('category').cat.as_ordered()
    # ... and which are continuous.
    for v in contin_ftrs:
        df[v] = df[v].astype('float32')


df_all_ftrs = pd.concat([df_train_ftrs, df_val_ftrs, df_test_ftrs])

for df in [df_all_ftrs, df_train_ftrs, df_val_ftrs, df_test_ftrs]:
    convType(df, cat_ftrs, contin_ftrs)

# In[12]:


# Compute list of embedding sizes.
cat_sz = [(c, len(df_all_ftrs[c].cat.categories) + 1) for c in cat_ftrs]
print(cat_sz)
emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in cat_sz]
print(emb_szs)

# In[13]:


_, _, nas, mapper = proc_df(df_all_ftrs, y_fld='production', do_scale=True, skip_flds=['Id'])

df_train_proc, y_train, nas, mapper = proc_df(df_train_ftrs, y_fld='production', do_scale=True,
                                              mapper=mapper, na_dict=nas, skip_flds=['Id'])

df_val_proc, y_val, nas, mapper = proc_df(df_val_ftrs, y_fld='production', do_scale=True,
                                          mapper=mapper, na_dict=nas, skip_flds=['Id'])

test_ids = df_test.Id

df_test_proc, _, nas, mapper = proc_df(df_test_ftrs, y_fld='production', do_scale=True,
                                       mapper=mapper, na_dict=nas, skip_flds=['Id'])

n_cont = len(df_train_proc.columns) - len(cat_ftrs)



# df_train_proc = df_val_proc
# seqs_train = seqs_val
# y_train = y_val

# torch.manual_seed(1)

# # seqs_train = seqs_train[len(seqs_train)//2:]
# seqs_train = seqs_train[:len(seqs_train)//2]

# #seqs_train = seqs_train[len(seqs_train)//2:]
# seqs_train = seqs_train[:len(seqs_train)//2]
#
# # seqs_train = seqs_train[len(seqs_train)//2:]
# seqs_train = seqs_train[:len(seqs_train)//2]
#
# seqs_train = seqs_train[len(seqs_train)//2:]
# #seqs_train = seqs_train[:len(seqs_train)//2]
#
# # seqs_train = seqs_train[len(seqs_train)//2:]
# seqs_train = seqs_train[:len(seqs_train)//2]
#
# #seqs_train = seqs_train[len(seqs_train)//2:]
# seqs_train = seqs_train[:len(seqs_train)//2]
#
# seqs_train = seqs_train[len(seqs_train)//2:]
# #seqs_train = seqs_train[:len(seqs_train)//2]


md = ColumnarSeqModelData.from_data_frames(path,  # path for data saving
                                           seqs_train,  # limits of training sequences
                                           df_train_proc,  # training set
                                           y_train,  # output variable for the training set
                                           seqs_val,  # limits of validation sequences
                                           df_val_proc,  # validation set
                                           y_val,  # output variable for the validation set
                                           cat_flds=cat_ftrs,  # categorical features
                                           is_reg=True,  # not regression
                                           is_multi=False,  # multi-label problem
                                           test_seqs_lim=seqs_test,  # limits of test sequences
                                           test_df=df_test_proc)  # test set

# In[ ]:


# dropout rate
dr = 0.1

learner_params = {
    "emb_szs": emb_szs,  # embedding sizes
    "n_cont": n_cont,  # num continuous inputs
    "emb_drop": dr,  # embeddings dropout probability
    "out_sz": 1,  # output size
    "szs": [300, 100],  # sizes of fully-connected layers
    "drops": [dr, dr],  # dropout probabilities after each FC layer
    "lstm_hidden_size": 100,  # size of the LSTM hidden states
    "lstm_num_layers": 2,  # number of LSTM layers
    "lstm_dropout": dr,  # LSTM dropout
    "use_bn": False,  # batch normalization
    "y_range": [0.0, 1.0]
}

m = md.get_learner(**learner_params)


def mae(y_pred, y_true):
    y_true = y_true.view((-1)).numpy()
    y_pred = y_pred.view((-1)).numpy()
    # Mask out nan values.
    m = np.invert(np.isnan(y_true))
    return metrics.mean_absolute_error(y_true[m], y_pred[m])


lr = 1e-5
m.fit(lr, n_cycle=10, metrics=[mae])
