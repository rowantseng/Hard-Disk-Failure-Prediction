import pandas as pd
import numpy as np

# Load dataframe
df = pd.read_csv('/backblaze-hard-drive/data/merge/ST4000DM000.csv')
print ('Size of original:', df.shape)
print (df.head())

# Drop 'normalized' columns
columns_names = list(df)
columns_to_drop = [columns_name for columns_name in columns_names if 'normalized' in columns_name]
df = df.drop(columns_to_drop, axis=1)

# Fill in zeros to numerical columns
disk_df = df.iloc[:,0:4]
smart_df = df.iloc[:,4:].fillna(0)
smart_df = smart_df.astype('int64')
smart_df = smart_df.apply(pd.to_numeric, downcast='integer')
reduced_df = disk_df.join(smart_df)
reduced_df = reduced_df.dropna(axis=1)

# Transform datatime and drop model column
reduced_df['date'] = pd.to_datetime(reduced_df['date'])
reduced_df = reduced_df.sort_values(by=['serial_number', 'date'], axis=0, ascending=True)
reduced_df = reduced_df.drop(columns=['model'])
reduced_df = reduced_df.reset_index(drop=True)
print ('Size of reduced df:', reduced_df.shape)

# Aggregate multiple columns
agg_df = (reduced_df[['date','serial_number','failure']].groupby('serial_number', as_index=True)
          .agg({'date':'count', 'failure':'sum'})
          .rename(columns={'date': 'date_count', 'failure': 'failure_sum'})
          .sort_values(by=['failure_sum'], axis=0, ascending=False))
print (agg_df.head(5))

n_disk = agg_df.shape[0]
n_fail = agg_df[agg_df.iloc[:, 1] >= 1].shape[0]
print ('Num of disks:', n_disk)
print ('Num of failed disks:', n_fail)
print ('Percentage of broken disks:', n_fail / n_disk * 100, '%')

# How many healthy disks are added to dataset
n_ok_delta = 0  ## changable
index_fail = agg_df[agg_df['failure_sum'] > 0].index.values
index_ok = agg_df[agg_df['failure_sum'] == 0].sample(n=(index_fail.shape[0] + n_ok_delta)).index.values
selected_disks = np.concatenate((index_fail, index_ok), axis=0)
df_reduced = reduced_df.loc[reduced_df.serial_number.isin(selected_disks)]
print ('Size of reduced df:', df_reduced.shape)
print ('Dataset size reduction:', int(100 * ((reduced_df.shape[0] - df_reduced.shape[0]) / reduced_df.shape[0])), '%')
df_reduced.to_pickle('/backblaze-hard-drive/data/merge/ST4000DM000_n_ok_{}.csv'.format(n_ok_delta))