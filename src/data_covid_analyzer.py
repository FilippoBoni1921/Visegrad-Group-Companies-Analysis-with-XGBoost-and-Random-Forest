import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self,df_ls, figsize=(15, 10), fontsize=25):
        """
        Initialize the DataAnalyzer object.

        Parameters:
        - df_ls (list of pandas.DataFrame): List of DataFrames containing the data.
        - figsize (tuple): Figure size for plots. Default is (15, 10).
        - fontsize (int): Font size for labels and titles in plots. Default is 25.
        """
        self.figsize = figsize # Initializing figure size for plots
        self.fontsize = fontsize # Initializing font size for labels and titles
        self.df_ls = df_ls # Initializing the list of DataFrames

    def calc_variation(self, median_pre, median_post):
        """
        Calculate the percentage variation between two mean values.

        Parameters:
        - mean_pre (float): Mean value before a certain event.
        - mean_post (float): Mean value after a certain event.

        Returns:
        - variation (float): Percentage variation between mean_pre and mean_post.
        """
        n = median_post - median_pre # Calculating the difference between
        if median_pre == 0.: # Checking if mean_pre is equal to 0
            return np.abs(n) * 100 # Calculating and returning absolute variation
        else:
            return np.abs(n / median_pre) * 100  # Calculating and returning relative variation

    def plot_variations(self, variations, title='', custom_y_tick=False, color='skyblue'):
        """
        Plot the variations.

        Parameters:
        - variations (list): List of variation values for each feature.
        - title (str): Title for the plot. Default is an empty string.
        - custom_y_tick (bool): Whether to use custom y-axis ticks. Default is False.
        - color (str): Color for the bars in the plot. Default is 'skyblue'.
        """
        x_labels = np.arange(len(variations)) + 2
        x_positions = np.arange(len(variations))

        plt.figure(figsize=self.figsize)

        plt.bar(x_positions, variations, width=0.6, color=color, edgecolor='black')

        plt.xlabel('Feature', fontsize=self.fontsize)
        plt.ylabel('% Variation', fontsize=self.fontsize)

        if title == '':
            plt.title(f'% Variation before and after Covid (pre <= Q1 2020, after >= Q2 2020)', fontsize=self.fontsize + 2, pad=20)
        else :
            plt.title(f'% Variation {title} before and after Covid (pre <= Q1 2020, after >= Q2 2020)', fontsize=self.fontsize + 2, pad=20)


        plt.xticks(x_positions, x_labels, fontsize=10)
        if custom_y_tick:
            plt.yticks(np.arange(0, max(variations) + 25, 25), fontsize=16)

        plt.xticks(rotation=45)
        plt.tight_layout()
        if title == '':
            name = f'%_Variation_before_and_after_Covid'
        else:
            name = f'%_Variation_{title}_before_and_after_Covid'

        plt.savefig("src/plots/" + name, bbox_inches='tight') # Saving the plot

    def analyze_data(self):
        """
        Analyze the data.

        Returns:
        - variations_all (list of lists): List containing variations for all features across all sectors.
        """
        Q_ls = [np.unique(df['Quarter'].values) for df in self.df_ls] # Extracting unique quarters from each DataFrame
        idx = Q_ls.index('2020 Q2') # Finding the index of '2020 Q2'

        df_ls_pre = self.df_ls[:idx] # Slicing DataFrame list to get pre-COVID data
        df_ls_post = self.df_ls[idx:] # Slicing DataFrame list to get post-COVID data

        concatenated_df_pre = pd.concat(df_ls_pre) # Concatenating pre-COVID DataFrames
        concatenated_df_post = pd.concat(df_ls_post) # Concatenating post-COVID DataFrames

        variations_all = [] # Initializing a list to store variations for all features

        variations = []   # Initializing a list to store variations for features across all sectors
        for col in self.df_ls[0].columns[2:-1]: # Iterating over columns excluding first two and last one
            vals_pre = concatenated_df_pre[col].values
            vals_post = concatenated_df_post[col].values
            median_pre = np.median(vals_pre) # Calculating median of pre-COVID values
            median_post = np.median(vals_post) # Calculating median of post-COVID values
            variation = self.calc_variation(median_pre, median_post)
            variations.append(variation)

        variations_all.append(variations)
        self.plot_variations(variations, custom_y_tick=True) # Plotting variations

        for sector in range(1, 7): #Iteratin over sectors and repeat the process
            variations = []
            concatenated_df_pre_masked = concatenated_df_pre[concatenated_df_pre[83] == sector]
            concatenated_df_post_masked = concatenated_df_post[concatenated_df_post[83] == sector]

            for col in self.df_ls[0].columns[2:-1]:
                vals_pre = concatenated_df_pre_masked[col].values
                vals_post = concatenated_df_post_masked[col].values
                median_pre = np.median(vals_pre)
                median_post = np.median(vals_post)
                variation = self.calc_variation(median_pre, median_post)
                variations.append(variation)

            variations_all.append(variations)
            self.plot_variations(variations, custom_y_tick=True, title="(sector" + str(sector) + ")") # Plotting variations
        
        return variations_all