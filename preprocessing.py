import os
import pandas as pd
import numpy as np
from utils.sliding_windows import sliding_window
from concurrent.futures import ProcessPoolExecutor


def process_subject(sub_id):
    df = pd.read_csv(f'annotations-rpoints/mesa-sleep-{sub_id}-rpoint.csv')
    if not np.all(np.isin(df['stage'].values, [0, 1, 2, 3, 4, 5])):
        print(f'Invalid labels for subject {sub_id}')
        return
    # convert stage 4 to stage 3
    df['stage'] = df['stage'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3})
    df = df[['seconds', 'stage']]
    df['IBI'] = df['seconds'].diff()
    df['IBI'] = df['IBI'].fillna(method='bfill')
    # Remove IBI values > 5 standard deviations from the mean
    df = df[np.abs(df['IBI'] - df['IBI'].mean()) <= (5 * df['IBI'].std())]
    df['IHR'] = 60 / df['IBI']
    # Normalize IHR by subtracting mean and dividing by standard deviation
    df['IHR'] = (df['IHR'] - df['IHR'].mean()) / df['IHR'].std()
    df['seconds'] = pd.to_datetime(df['seconds'], unit='s')
    df.set_index('seconds', inplace=True)
    # resample to 2Hz preserve the labels as integers
    df = df.resample('500L').mean()
    df['IHR'] = df['IHR'].interpolate(method='linear')
    df['stage'] = df['stage'].interpolate(method='ffill').astype(int)
    # create a new column for labels
    df = df[['IHR', 'stage']]
    #reset the index
    df.reset_index(inplace=True, drop=True)
    if len(df) >= 72000:
        # if so, truncate the df to 72000
        df = df.iloc[:72000]
    elif len(df) < 72000:
        # pad the df with zeros
        df = df.reindex(range(72000), fill_value=0)
    # check is there is a df folder in the current directory
    if not os.path.exists('data'):
        os.mkdir('data')
    df.to_csv(f'data/{sub_id}_processed.csv', index=False)

def main():
    sub_ids = [x.split('-')[2] for x in os.listdir('annotations-rpoints')]
    sub_ids = list(set(sub_ids))

    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(process_subject, sub_ids)

if __name__ == '__main__':
    main()


