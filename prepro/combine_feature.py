import pandas as pd
import numpy as np
import random


input_csv = '/backblaze-hard-drive/data/merge/ST4000DM000_n_ok_0.csv'

# Load dataframe
df = pd.read_pickle(input_csv)
df = df.sort_values(by=['serial_number','date'], axis=0, ascending=True)

# Init columns
df.insert(1, 'weekday', df['date'].dt.dayofweek)
df.insert(5, 'fails_soon', np.nan)
df.insert(6, 'seq_id', np.nan)
df.insert(7, 'work_day', np.nan)
df.insert(8, 'max_work_day', np.nan)
df.insert(9, 'final_failure', np.nan)

df = df.sort_values(by=['serial_number', 'date'], axis=0, ascending=True)

# Deal with 'seq_id' and 'work_day'
curr_serial_number = None
prev_failure = 0
observ_seq_id = 0
counter = 0
for idx, data in df.iterrows():
    if (data['serial_number'] != curr_serial_number) or (prev_failure == 1):
        curr_serial_number = data['serial_number']
        curr_work_day = 1
        observ_seq_id += 1
    else:
        curr_work_day += 1

    df.at[idx, 'seq_id'] = observ_seq_id
    df.at[idx, 'work_day'] = curr_work_day
    prev_failure = data['failure'] 

    if (counter%500000==0): print (int(counter*100/df.shape[0]),'%',end =" > ")
    counter += 1

# Deal with 'max_work_day', 'final_failure' and 'fails_soon'
df = df.sort_values(by=['seq_id', 'work_day'], axis=0, ascending=[True, False])

sample_days = 5
predict_failure_days = 5
assert predict_failure_days >= sample_days

prev_observ_seq_id = None
counter = 0
remove_list = []
for idx, data in df.iterrows():
    # new disk input, keep 1. work_day and 2. failure 
    # as 1. max_work_day and 2. final_failure
    if (prev_observ_seq_id != data['seq_id']):
        max_working_day = data['work_day']
        final_failure = data['failure']

    # remove data that are shorter than sample_days
    if max_working_day >= sample_days:
        df.at[idx, 'max_work_day'] = max_working_day
        df.at[idx, 'final_failure'] = final_failure
        if (final_failure == 1) and (data['work_day'] > (max_working_day - predict_failure_days)):
            df.at[idx, 'fails_soon'] = 1
        else:
            df.at[idx, 'fails_soon'] = 0
    else:
        remove_list.append(idx)

    prev_observ_seq_id = data['seq_id']

    if (counter%500000==0): print (int(counter*100/df.shape[0]),'%',end =" > ")
    counter += 1

df = df.drop(remove_list)
df[['fails_soon','seq_id','work_day','max_work_day','final_failure']] = \
df[['fails_soon','seq_id','work_day','max_work_day','final_failure']].astype(int)
print ("Completed all new feature creation.")

# Gather failed samples with sample days
df = df.sort_values(by=['seq_id', 'work_day'], axis=0, ascending=[True, True])
df_failed_samples = pd.DataFrame()
failed_samples_list = []
counter = 0
for idx, data in df.iterrows():
    if (data['fails_soon'] == 1) and (data['work_day'] >= sample_days):
        int_loc = df.index.get_loc(idx)
        failed_sample = df.iloc[int_loc:int_loc + sample_days]
        failed_samples_list.append(failed_sample)

    if (counter%500000==0): print (int(counter*100/df.shape[0]),'%',end =" > ")
    counter += 1

df_failed_samples = df_failed_samples.append(failed_samples_list)
df_failed_samples = df_failed_samples.reset_index(drop=True)
df_failed_samples = df_failed_samples.sort_index(axis=0, ascending=False)

print ('Num of failed samples:', len(df_failed_samples))
print ("Completed failed samples creation.")

# Gather healthy samples with sample days
failed_len = int(df_failed_samples.shape[0]/sample_days)
df_len = df.shape[0]
df_ok_samples = pd.DataFrame()
ok_samples_list = []
counter = 0
used_disks_list = []
used_offset_list = []
max_same_disks = 2
for i in range(failed_len):
    selected_ok = False

    while selected_ok == False:
        ok_offset = random.randint(0, df_len-sample_days)
        check_row = df.iloc[ok_offset]
        if (check_row['fails_soon'] == 0) \
        and (check_row['work_day'] >= sample_days)\
        and (used_disks_list.count(check_row['serial_number']) <= max_same_disks)\
        and (ok_offset not in used_offset_list):
            used_disks_list.append(check_row['serial_number'])
            used_offset_list.append(ok_offset)
            ok_sample = df.iloc[ok_offset:ok_offset+sample_days]
            ok_samples_list.append(ok_sample)
            selected_ok = True

    if (counter%5000==0): print (int(counter*100/failed_len),'%',end =" > ")
    counter += 1

df_ok_samples = df_ok_samples.append(ok_samples_list)
df_ok_samples = df_ok_samples.reset_index(drop=True)
df_ok_samples = df_ok_samples.sort_index(axis=0,ascending=False)
print ("Completed healthy samples creation.")

df_samples = pd.concat([df_failed_samples, df_ok_samples])
df_samples = df_samples.reset_index(drop=True)
df_samples.to_pickle('disk.csv')