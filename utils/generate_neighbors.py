import bokeh.models as bmo
import hdbscan
import numpy as np
import pandas as pd
from bokeh.palettes import d3
from bokeh.plotting import figure, show, ColumnDataSource, output_file
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#
# def parse_args():
#     parser = ArgumentParser('Script to generate dataframe with user embeddings')
#     parser.add_argument('--input')


# globals
CUTOFF = 15
SEED = 20150101

# select columns
user_df = pd.read_csv('../Datasets/users_NY_2015-2019.csv', index_col=0)
unused = ['sample_checklist', 'geometry', 'mean_interval']
user_df = user_df.iloc[:, [idx for idx, ele in enumerate(user_df.columns) if ele not in unused]]

# filter users with too little entries
user_df = user_df.loc[user_df.n_checklists >= 10]
user_df['since'] = 2019 - user_df['since']

# get a random subset of users
#indices = np.random.choice(len(user_df), 5000, replace=False)
#user_df = user_df.iloc[indices]

# preprocess data for TSNE
user_df = user_df.fillna(0)
normalized_users = user_df.values
min_max_scaler = preprocessing.MinMaxScaler()
normalized_users = min_max_scaler.fit_transform(normalized_users)
pca = PCA(n_components=4)
print(pca.fit(normalized_users).explained_variance_)
normalized_users = pca.fit_transform(normalized_users)

# run TSNE
tsne = TSNE(random_state=SEED, perplexity=50, n_iter=5000, n_jobs=-1)
users_proj = tsne.fit_transform(normalized_users)

# run hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
cluster_labels = [str(lbl) for lbl in clusterer.fit_predict(users_proj)]

# stage for plotting
plot_df = user_df.copy()
plot_df['x'], plot_df['y'], plot_df['label'] = users_proj[:, 0], users_proj[:, 1], cluster_labels
source = ColumnDataSource(plot_df)

# tooltips for plot
TOOLTIPS = [
    ("#species", "@n_species"),
    ("bird common", "@median_distance"),
    ("bird size", "@percent_complete"),
    ("#checklists", "@n_checklists"),
    ("group size", "@mean_group_size"),
    ("starting year", "@since")
]

# plot users
palette = d3['Category20'][len(plot_df['label'].unique())]
color_map = bmo.CategoricalColorMapper(factors=plot_df['label'].unique(),
                                       palette=palette)

# create figure and plot
birders_plot = figure(title=' ', tooltips=TOOLTIPS)
birders_plot.circle('x', 'y', color='black', fill_color={'field': 'label', 'transform': color_map}, size=6,
                    alpha=0.2, fill_alpha=0.6, source=source)
birders_plot.toolbar.logo = None
birders_plot.toolbar_location = None

# save output
output_file('all_users.html')
show(birders_plot)
