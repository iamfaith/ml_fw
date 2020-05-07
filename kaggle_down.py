from google.colab import drive
drive.mount('/content/gdrive')


import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"
# /content/gdrive/My Drive/Kaggle is the path where kaggle.json is present in the Google Drive


df[df.isin([np.nan, np.inf, -np.inf]).any(1)]
~df.isin(somewhere)



new_df = my_df.loc[(my_df['A_count'] == 0.0) & (my_df['B_count'] == 0.0)]

my_df.query('A_count == 0 & B_count == 0')
sell_train_val_total.query('item_id == "HOBBIES_1_001" |  item_id == "HOBBIES_1_002"')


A variation on the .agg() function; provides the ability to (1) persist type DataFrame, (2) apply averages, counts, summations, etc. and (3) enables groupby on multiple columns while maintaining legibility.

df.groupby(['att1', 'att2']).agg({'att1': "count", 'att3': "sum",'att4': 'mean'})
using your values...

df.groupby(['Name', 'Fruit']).agg({'Number': "sum"})

df1 = df.groupby('dow', as_index=False).sum()
df1 = df.groupby('dow').sum().reset_index()


dataframe[['column1','column2']]
to select by iloc and specific columns with index number:

dataframe.iloc[:,[1,2]]
with loc column names can be used like

dataframe.loc[:,['column1','column2']]



for index, row in df.iterrows():
    print(row['c1'], row['c2'])
    
    
tmp1 = tmp[tmp.isin(b.index.to_list()).any(1)].drop(['total'], 1).reset_index(drop=True)



# Import widgets
from ipywidgets import widgets, interactive, interact
import ipywidgets as widgets
from IPython.display import display

train_sales = sell_train_val
calendar_df = cal

days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]

ids = np.random.choice(train_sales['id'].unique().tolist(), 1000)

series_ids = widgets.Dropdown(
    options=ids,
    value=ids[0],
    description='series_ids:'
)

def plot_data(series_ids):
    df = train_sales.loc[train_sales['id'] == series_ids][time_series_columns]
    df = pd.Series(df.values.flatten())

    df.plot(figsize=(20, 10), lw=2, marker='*')
    df.rolling(7).mean().plot(figsize=(20, 10), lw=2, marker='o', color='orange')
    plt.axhline(df.mean(), lw=3, color='red')
    plt.grid()


w = interactive(
    plot_data,
    series_ids=series_ids
)
display(w) 