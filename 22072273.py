import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file_path):
    """
    Read data from a CSV file using pandas.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Data loaded from the CSV file.
    """
    df = pd.read_csv(file_path)
    # Filters the data to include only indicators of interest
    indicators_to_filter = ['EN.ATM.GHGT.KT.CE', 'EN.ATM.CO2E.KT', 'EG.FEC.RNEW.ZS', 'AG.LND.PRCP.MM'
                            , 'AG.LND.FRST.ZS', 'AG.LND.ARBL.ZS']
    countries_to_filter = ['Brazil', 'China', 'United Kingdom', 'India', 'Russian Federation', 
                           'United States']

    # Filtering the rows based on indicators and countries
    df1 = df[(df['Country Name'].isin(countries_to_filter)) & (df['Indicator Code']
                           .isin(indicators_to_filter))]

    # Selecting the required columns
    columns_to_keep = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1995', 
                       '2000', '2005', '2010', '2015', '2020']

    df1 = df1[columns_to_keep]
    df = df1
    return df



def visualize_clusters(df, columns_for_clustering):
    """
    Visualize clustering results in separate plots for each pair of columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data and cluster information.
    - columns_for_clustering (list): Columns used for clustering.

    Returns:
    - None
    """
    n_columns = len(columns_for_clustering)
    fig, axes = plt.subplots(n_columns, n_columns, figsize=(15, 15))

    for i in range(n_columns):
        for j in range(n_columns):
            if i == j:
                for cluster_label in df['Cluster'].unique():
                    sns.histplot(df[df['Cluster'] == cluster_label][columns_for_clustering[i]], 
                    kde=True, ax=axes[i, j], label=f'Cluster {cluster_label}', color=f'C{cluster_label}', 
                    element='step', stat='density')
                axes[i, j].legend()
            else:
                sns.scatterplot(x=columns_for_clustering[j], y=columns_for_clustering[i], data=df, 
                    hue='Cluster', ax=axes[i, j], palette='viridis')

    plt.suptitle("Clustered Data for all Indicators Globally", y=1.02)
    plt.tight_layout()
    plt.show()
# Used with curve_fit to fit linear model of data
def linear_func(x, a, b):
    return a * x + b
#Fitting a linear model to the normalized data for a specific country 
def fit_and_plot(df, country_code, indicator_code, columns_to_normalize):
    """
    Fit a linear model, make predictions, and plot the results.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - country_code (str): Code of the country to analyze.
    - indicator_code (str): Code of the indicator to analyze.
    - columns_to_normalize (list): Columns to normalize for fitting.

    Returns:
    - None
    """
    x_values = np.array([1995, 2000, 2005, 2010, 2015, 2020])
    
    #To extract the values for a specific country and indicator from the DataFrame df. 
    y_values = df[(df['Country Code'] == country_code) & (df['Indicator Code'] == 
                            indicator_code)][columns_to_normalize].values.flatten()

    # Fitting the model
    params, covariance = curve_fit(linear_func, x_values, y_values)

    # Predictions for future years
    future_years = np.array([2030, 2040])
    predicted_values = linear_func(future_years, *params)

    # Display the fitted parameters
    print(f'Fitted parameters for {country_code} - {indicator_code}: a = {params[0]}, b = {params[1]}')

    # Display the predicted values
    for year, value in zip(future_years, predicted_values):
        print(f'Predicted value for {year}: {value}')

    # Plotting the original data and the fitted curve
    plt.scatter(x_values, y_values, label='Actual Data')
    plt.plot(x_values, linear_func(x_values, *params), label='Fitted Curve', color='red')
    plt.xlabel('Year')
    plt.ylabel('Normalized Values')
    plt.legend()
    plt.title(f'Fitting: {indicator_code} for {country_code}')
    plt.show()

# Step 1: Reading the CSV file
data_file_path = 'world_bank.csv'
df = read_data(data_file_path)

# Step 2: Normalization
columns_to_normalize = ['1995', '2000', '2005', '2010', '2015', '2020']
scaler = StandardScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Step 3: Clustering with Silhouette Score
columns_for_clustering = ['1995', '2000', '2005', '2010', '2015', '2020']
silhouette_scores = []

for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[columns_for_clustering])
    silhouette_scores.append(silhouette_score(df[columns_for_clustering], df['Cluster']))
#
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[columns_for_clustering])

# Step 4: Back-scaling of Cluster Centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Step 5: Visualizing the Clustering Results

visualize_clusters(df, columns_for_clustering)

# Step 6: Fitting, Predictions, and Plotting for a specific country and indicator
# Looping through all country codes and indicator codes
for country_code in df['Country Code'].unique():
    for indicator_code in df['Indicator Code'].unique():
        # To check if the current combination of country and indicator is present in the data
        if df[(df['Country Code'] == country_code) & (df['Indicator Code'] == 
                                                      indicator_code)].shape[0] > 0:
            # Fit and plot for the current combination
            fit_and_plot(df, country_code, indicator_code, columns_to_normalize)

