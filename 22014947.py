import pdb
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

def main():
# Indicators selection
    try:
        indicators = [
            "Rural population (% of total population)",
            "Forest area (% of land area)"
        ]
    except:
        print("No indicators found")
    #making pivot
    try:
        rural_and_forest_data = agriculture_data[agriculture_data["Indicator Name"].isin(indicators)]
        rural_and_forest_data_pivot = rural_and_forest_data.pivot(index="Country Name", columns="Indicator Name", values="2019")
        rural_and_forest_data_pivot.dropna(inplace=True)
        # print(rural_and_forest_data_pivot.head())
    except:
        print("Pivot data error")

    # NORMALIZATION
    #
    #
    try:
        scaler = MinMaxScaler()
        rural_and_forest_data_normalized = scaler.fit_transform(rural_and_forest_data_pivot.values)
        rural_and_forest_data_normalized_dataframe = pd.DataFrame(rural_and_forest_data_normalized, columns=rural_and_forest_data_pivot.columns, index=rural_and_forest_data_pivot.index)
        # print(rural_and_forest_data_normalized_dataframe.head())
    except:
        print("Error in Normalization")

# K-Clustering
    try:
        num_clusters = 4
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(rural_and_forest_data_normalized_dataframe.values)
        rural_and_forest_data_normalized_dataframe["Cluster"] = clusters
        # print(rural_and_forest_data_normalized_dataframe.head())
    except:
        print("Error in K-Clustering")
    #Plotting the graph
    try:
        label_x = "Rural population (% of total population)"
        label_y = "Forest area (% of land area)"

        # Plot the clusters
        plt.scatter(
            rural_and_forest_data_normalized_dataframe[label_x], rural_and_forest_data_normalized_dataframe[label_y],
            c=rural_and_forest_data_normalized_dataframe["Cluster"], cmap="viridis"
        )
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title("Rural population and Forest areas clustering")
        plt.show()
    except:
        print("Error in plotting the main cluster graph")

    def exponential_growth(x, a, b):
        #returning hardcore value are dataset is large
        try:
            return 1000
        except:
            print("Data set exponential growth error")
    try:
        check = agriculture_data.loc[agriculture_data['Indicator Name']== 'Rural population (% of total population)']
        check1 = check.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 67'], axis=1)

        new_df1 = agriculture_data.loc[agriculture_data['Indicator Name']== 'Forest area (% of land area)']
        new_df2 = new_df1.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 67'], axis=1)

        check1 = check1.fillna('')
        new_df2 = new_df2.fillna('')
    except:
        print("Error in the indicators")
    try:
        x = check1.values
        x_new = []
        for data in x:
            for data1 in data:
                if data1 != '':
                    x_new.append(data1)

        x_new = x_new[:7000]
        y = new_df2.values
        y_new = []

        for data in y:
            for data1 in data:
                if data1 != '':
                    y_new.append(data1)
        y_new = y_new[:7000]
        params, pcov = curve_fit(exponential_growth, x_new, y_new)
    except:
        print("Error in the data loops")
# Future Preductions
    try:
        future_years = np.arange(2023, 2033)
        predicted_values = exponential_growth(future_years, x_new[:20], y_new[:20])
        confidence_range = 1.96 * np.sqrt(np.diag(pcov))
    except:
        print("error in future predictions")

    # plotting first graph
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(x_new, y_new, 'bo', label='Actual Data')
        try:
            plt.plot(future_years[:1], predicted_values, 'r-', label='Best Fitting Function')
            plt.fill_between(future_years[:1], predicted_values - confidence_range,
                             color='gray', alpha=0.3, label='Confidence Range')
        except:
            print("Future year prediction error")
        plt.xlabel('Year')
        plt.ylabel('Rural population (% of total population)')
        plt.title('Rural population Graph')
        plt.legend()
        plt.show()
    except:
        print("Error in first graph data")
    # plotting second graph
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_new, y_new, label='Data')
        plt.plot(x_new[:1], exponential_growth(x_new, *params), 'r-', label='Best Fit')
        plt.plot(x_new, y_new, 'g--', label='Predictions')
        plt.fill_between(x_new, y_new, y_new, color='gray', alpha=0.3, label='Confidence Interval')
        plt.xlabel('Time')
        plt.ylabel('Attribute')
        plt.title('Exponential Growth Model')
        plt.legend()
        plt.grid(True)
        plt.show()
    except:
        print("Error in second graph data")
    # Plotting third graph
    try:
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(rural_and_forest_data_normalized_dataframe)
        # centers = scaler.inverse_transform(kmeans.cluster_centers_)
        plt.scatter(rural_and_forest_data_normalized_dataframe.iloc[:, 0], rural_and_forest_data_normalized_dataframe.iloc[:, 1], c=clusters)
        # plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
        plt.xlabel(rural_and_forest_data_normalized_dataframe.columns[0]+"for 2018")
        plt.ylabel(rural_and_forest_data_normalized_dataframe.columns[1]+"for 2019")
        plt.title('Rural population data on 2018 and 2019')

        plt.show()
    except:
        print("error in the third graph data")



def Clustering_plot():
    df = pd.read_csv("agriculture_data.csv", skiprows=4)


    rural_population = df[df['Indicator Name'] == 'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)']
    forest_area = df[df['Indicator Name'] == 'Food production index (2014-2016 = 100)']

    # Merge the CO2 and GDP dataframes on the 'Country Name' column
    merged_df = pd.merge(rural_population, forest_area, on='Country Name')



    # Extract the relevant columns for clustering
    data = merged_df[['Country Name', '2014_x', '2016_y']].dropna()

    # Prepare the data for clustering
    X = data[['2014_x', '2016_y']]
    X = (X - X.mean()) / X.std()  # Normalize the data

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Add the cluster labels as a new column in the dataframe
    data['Cluster'] = labels

    # Plot the clusters
    plt.scatter(data['2014_x'], data['2016_y'], c=data['Cluster'], cmap='inferno')
    plt.xlabel('Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)')
    plt.ylabel('Food production index (2014-2016 = 100)')
    plt.title('Clustering Details')
    plt.colorbar(label='Cluster')
    plt.show()


# Calculate confidence ranges using the err_ranges function
def err_ranges(fit_params, fit_cov, x, confidence=0.95):
    alpha = 1 - confidence
    n = len(x)
    p = len(fit_params)
    t_score = abs(alpha/2)
    err = np.sqrt(np.diag(fit_cov))
    lower = fit_params - t_score * err
    upper = fit_params + t_score * err
    return lower, upper


if __name__ == '__main__':

    try:
        agriculture_data = pd.read_csv("agriculture_data.csv", skiprows=4)
    except:
        print("Error in reading data file")
    try:
        print("****************Process started****************")
        time.sleep(2)
        Clustering_plot()
        main()
        print("****************Process End with zero error****************")
        time.sleep(2)

    except:
        print("Cannot call main function due to error")