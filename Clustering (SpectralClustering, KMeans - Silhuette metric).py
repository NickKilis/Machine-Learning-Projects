'''
Clustering (SpectralClustering, KMeans - Silhuette metric)

From sklearn, we will import:
    'datasets'        : for loading data
    'model_selection' : package, which will help validate our results
    'metrics'         : package, for measuring scores
    'cluster'         : package, for importing the corresponding clustering algorithm
    'preprocessing'   : package, for rescaling ('normalizing') our data
'''

# =============================================================================
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering, KMeans

# =============================================================================
# Load a dataset.
myData = datasets.load_breast_cancer(return_X_y=False)
# Get samples from the data, and keep only the features that you wish
numberOfFeatures = 30
X = myData.data[:, :numberOfFeatures]
y = myData.target
# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y,stratify=y, random_state = 0)

# =============================================================================
# Most clustering methods are sensitive to the magnitudes of the features' values. Since we already
# split our dataset into 'train' and 'test' sets, we must rescale them separately (but with the same scaler)
# So, we rescale train data to the [0,1] range using a 'MinMaxScaler()' from the 'preprocessing' package,
# from which we shall then call 'fit_transform' with our data as input.
# Note that, test data should be transformed ONLY (not fit+transformed), since we require the exact
# same scaler used for the train data. 
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(x_train)
x_train = minMaxScaler.transform(x_train)
x_test = minMaxScaler.transform(x_test)

# =============================================================================
# It's time to create our clustering algorithm. Most algorithms share many common
# hyperparamters, but depending on their nature they tend to be tuned by different
# ones as well. A basic guide on the most important (/unique) hyperparameters of
# each clustering algorithm can be found here:
# https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
# Model parameters
n_init = [5, 10, 25]
affinities = ['nearest_neighbors', 'rbf']
n_neighbors = [3, 5, 10]
# ADD COMMAND TO CREAETE CLUSTERING METHOD HERE
for n in n_init:
    for affinity in affinities:
        if affinity == 'nearest_neighbors':
            for neighbors in n_neighbors:
                # Create Spectral Clustering model
                sc = SpectralClustering(n_clusters=2, affinity=affinity, n_init=n, assign_labels='discretize', n_neighbors=neighbors, random_state=0)
                # Train model
                sc.fit(x_train)
                # predict test data
                y_predicted = sc.fit_predict(x_test)
                # Calculate silhouette_score
                result = metrics.silhouette_score(x_test, y_predicted)
                print('# =============================================================================')
                print('Cluster:                             SpectralClustering')
                print('Kmeans runs:                         ' + str(n))
                print('Affinity:                            ' + affinity)
                print('Number of neighbors:                 ' + str(neighbors))
                print('Silhouette score:                    ' + str(result))
        else:
            # Create Spectral Clustering model
            sc = SpectralClustering(n_clusters=2, affinity=affinity, n_init=n, assign_labels='discretize', random_state=0)
            # Train model
            sc.fit(x_train)
            # predict test data
            y_predicted = sc.fit_predict(x_test)
            
# For this project, the 'Silhuette' metric is suitable for the model's evaluation. 
# ==========================================================================================================================
            # Calculate silhouette_score
            result = metrics.silhouette_score(x_test, y_predicted)
            print('# =============================================================================')
            print('Cluster:                             SpectralClustering')
            print('Kmeans runs:                         ' + str(n))
            print('Affinity:                            ' + affinity)
            print('Silhouette score:                    ' + str(result))
# Run KMeans
for n in n_init:
    km = KMeans(n_clusters=2, n_init=n, random_state=0)
    # Train model
    km.fit(x_train)
    # predict test data
    y_predicted = km.fit_predict(x_test)
    
# For this project, the 'Silhuette' metric is suitable for the model's evaluation. 
# ==========================================================================================================================
    # Calculate silhouette_score
    result = metrics.silhouette_score(x_test, y_predicted)
    print('# =============================================================================')
    print('Cluster:                             KMeans')
    print('Kmeans runs:                         ' + str(n))
    print('Silhouette score:                    ' + str(result))
# ==========================================================================================================================