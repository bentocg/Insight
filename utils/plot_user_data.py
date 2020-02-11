import bokeh.models as bmo
import hdbscan
import pandas as pd
from bokeh.palettes import d3
from bokeh.plotting import figure, show, ColumnDataSource, output_file
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# globals
CUTOFF = 15
SEED = 20150101

# select columns
user_all = pd.read_csv('../Datasets/users_NA_relDec-2010.csv', index_col=0)
user_df = pd.read_csv('../Datasets/users_NY_2015-2019.csv', index_col=0)
print(len(user_df))
user_df = user_all.loc[user_df.index]
print(len(user_df))
unused = ['sample_checklist', 'geometry', 'mean_interval']
user_df = user_df.iloc[:, [idx for idx, ele in enumerate(user_df.columns) if ele not in unused]]

# filter users with too little entries
user_df = user_df.loc[user_df.n_checklists >= 10]

# get a random subset of users
# indices = np.random.choice(len(user_df), 5000, replace=False)
# user_df = user_df.iloc[indices]

# preprocess data for TSNE
user_df = user_df.fillna(0)
normalized_users = user_df.values
min_max_scaler = preprocessing.MinMaxScaler()
normalized_users = min_max_scaler.fit_transform(normalized_users)
pca = PCA(n_components=4)
normalized_users = pca.fit_transform(normalized_users)
print(pca.explained_variance_ratio_)

# run TSNE
tsne = TSNE(random_state=SEED, perplexity=50, n_iter=5000, n_jobs=-1, learning_rate=500)
users_proj = tsne.fit_transform(normalized_users)

# run hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
cluster_labels = [str(lbl) for lbl in clusterer.fit_predict(users_proj)]

# stage for plotting
plot_df = user_df.copy()
plot_df['x'], plot_df['y'], plot_df['label'] = users_proj[:, 0], users_proj[:, 1], cluster_labels
source = ColumnDataSource(plot_df)
plot_df['since'] = [2019 - ele for ele in plot_df.since]

# tooltips for plot
TOOLTIPS = [
    ("#species", "@n_species"),
    ("bird common", "@species_common"),
    ("travel distance", "@median_travel_distance"),
    ("#checklists", "@n_checklists"),
    ("group size", "@mean_group_size"),
    ("starting year", "@since"),
    ("median_start", "@median_start"),
    ("percent_travel", "@percent_travel")
]

# plot users
palette = d3['Category20'][max(3, len(plot_df['label'].unique()))]
color_map = bmo.CategoricalColorMapper(factors=plot_df['label'].unique(),
                                       palette=palette)

# create figure and plot
birders_plot = figure(title=' ', tooltips=TOOLTIPS)
birders_plot.circle('x', 'y', color='black', fill_color={'field': 'label', 'transform': color_map}, size=6,
                    alpha=0.2, fill_alpha=0.6, source=source)
birders_plot.toolbar.logo = None
birders_plot.toolbar_location = None

# save output
output_file('users_NY_2005-2019.html')
show(birders_plot)
