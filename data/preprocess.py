import pandas as pd

df = pd.read_csv('raw/rev02.csv')
test_split = int(0.8*df.shape[0])
df.sample(frac=1)
train = df[:test_split].reset_index(inplace=False)
test = df[test_split:].reset_index(inplace=False)

train.to_csv('raw/train_rev02.csv')
test.to_csv('raw/test_rev02.csv')
