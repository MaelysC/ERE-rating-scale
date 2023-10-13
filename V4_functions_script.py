# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:46:41 2023

@author: Maelys.Courtat2
"""

# Imports
import pandas as pd
import numpy as np
import os

# function to get save directory


def get_chart_dir():
    cwd = os.getcwd()
    default_chart_dir = os.path.join(cwd, 'Courtat_charts')

    answer = input("Figures will be saved in {}. Press enter to use this path, or provide another path".format(
        default_chart_dir))

    if answer != "":
        assert os.path.isdir(answer), "Please enter a valid path"
        chart_dir = answer
    else:
        if not os.path.isdir(default_chart_dir):
            os.mkdir(default_chart_dir)
        assert os.path.isdir(
            default_chart_dir), "Something went wrong, make sure you can write to the folder you're in"
        chart_dir = default_chart_dir

    return chart_dir


# Defining five performance classes labels
labels = ['A', 'B', 'C', 'D', 'E']

# Importing dataset, cleaning and calculating Log single score values


def import_AGB31():
    ''' Imports AGB31 dataset (254 duplicates removed) and prepares it for analysis (removing unwanted columns and row). Returns AGB31 DataFrame'''
    AGB31_supermarket = pd.read_csv(
        'AGB 3.1 cradle to shelf unique products.csv', skiprows=1)
    BSF = AGB31_supermarket[(AGB31_supermarket['Food item in AGB 3.1 (short name)']
                             == 'Dried deffated black soldier fly protein')].index
    AGB31_supermarket = AGB31_supermarket.drop(BSF)
    Cols_tokeep = ['Food item in AGB 3.1 (short name)', 'Serving size (g)', 'Climate change per kg', 'Single score per kg',
                   'Food group', 'Food sub group', 'Single score per serving', 'Climate change per serving']
    AGB31 = AGB31_supermarket[Cols_tokeep]
    AGB31['Log (single score per serving)'] = np.log(
        AGB31['Single score per serving'])
    AGB31['Log (single score per kg)'] = np.log(AGB31['Single score per kg'])
    return AGB31

# Assigning ratings based on single scores for each functional unit and linear/log-transformed data

    # Width-based scenario (S1)


def kg_width_ratings(df):
    ''' Assigns product ratings to dataframe df based on width-based scenario (S1) applied to single score per kg data, returns a list of the bin thresholds and the df '''
    df['Score_NC_kg_width'], cut_bin_NC_kg_width = pd.cut(
        x=df['Single score per kg'], bins=5, labels=labels, include_lowest=True, retbins=True)
    return cut_bin_NC_kg_width, df


def kg_width_ratings_ln(df):
    ''' Assigns product ratings to dataframe df based on width-based scenario (S1) applied to single score per kg (log-transformed) data, returns a list of the bin thresholds and the df '''
    df['Score_NC_kg_width_ln'], cut_bin_NC_kg_width_ln = pd.cut(
        x=df['Log (single score per kg)'], bins=5, labels=labels, include_lowest=True, retbins=True)
    return np.exp(cut_bin_NC_kg_width_ln), df


def serv_width_ratings(df):
    ''' Assigns product ratings to dataframe df based on width-based scenario (S1) applied to single score per serving data, returns a list of the bin thresholds and the df '''
    df['Score_NC_serv_width'], cut_bin_NC_serv_width = pd.cut(
        x=df['Single score per serving'], bins=5, labels=labels, include_lowest=True, retbins=True)
    return cut_bin_NC_serv_width, df


def serv_width_ratings_ln(df):
    ''' Assigns product ratings to dataframe df based on width-based scenario (S1) applied to single score per serving (log-transformed) data, returns a list of the bin thresholds and the df '''
    df['Score_NC_serv_width_ln'], cut_bin_NC_serv_width_ln = pd.cut(
        x=df['Log (single score per serving)'], bins=5, labels=labels, include_lowest=True, retbins=True)
    return np.exp(cut_bin_NC_serv_width_ln), df

    # Quantiles-based scenario (S2)


def kg_quant_ratings(df):
    ''' Assigns product ratings to dataframe df based on quantiles-based thresholding scenario (S2) applied to single score per kg data, returns a list of the bin thresholds and the df '''
    df['Score_NC_kg_quant'], cut_bin_NC_kg_quant = pd.qcut(
        df['Single score per kg'], q=5, labels=labels, retbins=True)
    return cut_bin_NC_kg_quant, df


def kg_quant_ratings_ln(df):
    ''' Assigns product ratings to dataframe df based on quantiles-based thresholding scenario (S2) applied to single score per kg (log-transformed) data, returns a list of the bin thresholds and the df '''
    df['Score_NC_kg_quant_ln'], cut_bin_NC_kg_quant_ln = pd.qcut(
        df['Log (single score per kg)'], q=5, labels=labels, retbins=True)
    return np.exp(cut_bin_NC_kg_quant_ln), df


def serv_quant_ratings(df):
    ''' Assigns product ratings to dataframe df based on quantiles-based thresholding scenario (S2) applied to single score per serving data, returns a list of the bin thresholds and the df  '''
    df['Score_NC_serv_quant'], cut_bin_NC_serv_quant = pd.qcut(
        df['Single score per serving'], q=5, labels=labels, retbins=True)
    return cut_bin_NC_serv_quant, df


def serv_quant_ratings_ln(df):
    ''' Assigns product ratings to dataframe df based on quantiles-based thresholding scenario (S2) applied to single score per serving (log-transformed) data, returns a list of the bin thresholds and the df  '''
    df['Score_NC_serv_quant_ln'], cut_bin_NC_serv_quant_ln = pd.qcut(
        df['Log (single score per serving)'], q=5, labels=labels, retbins=True)
    return np.exp(cut_bin_NC_serv_quant_ln), df

    # Hybrid scenario, 10% default cut-off value (S3)


def kg_hybrid_ratings(df, col_name, cutoff):
    ''' Assigns product ratings based on dataframe df['Single score per kg'] values applying hybrid scenario (S3) (percentage of values to be assigned to A and E defined via cutoff), returns a list of the bin thresholds and the df with new column as col_name'''
    SS_kg_class_hybrid = (df['Single score per kg'].quantile(
        1 - cutoff) - df['Single score per kg'].quantile(cutoff))/3  # Calculating internal bin width
    bins_NC_kg_hybrid = [df['Single score per kg'].min(), df['Single score per kg'].quantile(cutoff),
                         df['Single score per kg'].quantile(
                             cutoff) + SS_kg_class_hybrid,
                         df['Single score per kg'].quantile(
                             cutoff) + (SS_kg_class_hybrid*2),
                         df['Single score per kg'].quantile(1-cutoff),
                         df['Single score per kg'].max()]  # Setting up bins
    df[col_name], cut_bin_NC_kg_hybrid = pd.cut(df['Single score per kg'],
                                                bins=bins_NC_kg_hybrid,
                                                labels=labels, include_lowest=True,
                                                retbins=True)
    return cut_bin_NC_kg_hybrid, df


def kg_hybrid_ratings_ln(df, col_name_ln, cutoff):
    ''' Assigns product ratings based on df['Log (single score per kg)'] values applying hybrid scenario (S3) (percentage of values to be assigned to A and E defined via cutoff), returns a list of the bin boundaries and the df with new column as col_name_ln'''
    df['Log (single score per kg)'] = np.log(df['Single score per kg'])
    SS_kg_class_hybrid_ln = (df['Log (single score per kg)'].quantile(
        1 - cutoff) - df['Log (single score per kg)'].quantile(cutoff))/3
    bins_NC_kg_hybrid_ln = [df['Log (single score per kg)'].min(), df['Log (single score per kg)'].quantile(cutoff),
                            df['Log (single score per kg)'].quantile(
                                cutoff) + SS_kg_class_hybrid_ln,
                            df['Log (single score per kg)'].quantile(
                                cutoff) + (SS_kg_class_hybrid_ln*2),
                            df['Log (single score per kg)'].quantile(1-cutoff),
                            df['Log (single score per kg)'].max()]
    df[col_name_ln], cut_bin_NC_kg_hybrid_ln = pd.cut(df['Log (single score per kg)'],
                                                      bins=bins_NC_kg_hybrid_ln,
                                                      labels=labels, include_lowest=True,
                                                      retbins=True)
    return np.exp(cut_bin_NC_kg_hybrid_ln), df


def serv_hybrid_ratings(df, col_name, cutoff):
    ''' Assigns product ratings based on dataframe df['Single score per serving'] values applying hybrid scenario (S3) (percentage of values to be assigned to A and E defined via cutoff), returns a list of the bin thresholds and the df with new column as col_name'''
    SS_serv_class_hybrid = (df['Single score per serving'].quantile(
        1 - cutoff) - df['Single score per serving'].quantile(cutoff))/3  # Calculating internal bin width
    bins_NC_serv_hybrid = [df['Single score per serving'].min(), df['Single score per serving'].quantile(cutoff),
                           df['Single score per serving'].quantile(
                               cutoff) + SS_serv_class_hybrid,
                           df['Single score per serving'].quantile(
                               cutoff) + (SS_serv_class_hybrid*2),
                           df['Single score per serving'].quantile(1-cutoff),
                           df['Single score per serving'].max()]  # Setting up bins
    df[col_name], cut_bin_NC_serv_hybrid = pd.cut(df['Single score per serving'],
                                                  bins=bins_NC_serv_hybrid,
                                                  labels=labels, include_lowest=True,
                                                  retbins=True)
    return cut_bin_NC_serv_hybrid, df


def serv_hybrid_ratings_ln(df, col_name_ln, cutoff):
    ''' Assigns product ratings based on df['Log (single score per serving)'] values applying hybrid scenario (S3) (percentage of values to be assigned to A and E defined via cutoff), returns a list of the bin boundaries and the df with new column as col_name_ln'''
    df['Log (single score per serving)'] = np.log(
        df['Single score per serving'])
    SS_serv_class_hybrid_ln = (df['Log (single score per serving)'].quantile(
        1 - cutoff) - df['Log (single score per serving)'].quantile(cutoff))/3
    bins_NC_serv_hybrid_ln = [df['Log (single score per serving)'].min(), df['Log (single score per serving)'].quantile(cutoff),
                              df['Log (single score per serving)'].quantile(
                                  cutoff) + SS_serv_class_hybrid_ln,
                              df['Log (single score per serving)'].quantile(
                                  cutoff) + (SS_serv_class_hybrid_ln*2),
                              df['Log (single score per serving)'].quantile(
                                  1-cutoff),
                              df['Log (single score per serving)'].max()]
    df[col_name_ln], cut_bin_NC_serv_hybrid_ln = pd.cut(df['Log (single score per serving)'],
                                                        bins=bins_NC_serv_hybrid_ln,
                                                        labels=labels, include_lowest=True,
                                                        retbins=True)
    return np.exp(cut_bin_NC_serv_hybrid_ln), df

    # Nested into functions based on cut-off value for hybrid scenario

# Default scenarios applied to linear and log data for both functional units


def assign_kg_ratings(df, col_name, col_name_ln, cutoff):
    cut_bin_NC_kg_width, df = kg_width_ratings(df)
    cut_bin_NC_kg_quant, df = kg_quant_ratings(df)
    cut_bin_NC_kg_hybrid, df = kg_hybrid_ratings(df, col_name, cutoff)
    cut_bin_NC_kg_width_ln, df = kg_width_ratings_ln(df)
    cut_bin_NC_kg_quant_ln, df = kg_quant_ratings_ln(df)
    cut_bin_NC_kg_hybrid_ln, df = kg_hybrid_ratings_ln(df, col_name_ln, cutoff)
    return df, cut_bin_NC_kg_width, cut_bin_NC_kg_quant, cut_bin_NC_kg_hybrid, cut_bin_NC_kg_width_ln, cut_bin_NC_kg_quant_ln, cut_bin_NC_kg_hybrid_ln


def assign_serv_ratings(df, col_name, col_name_ln, cutoff):
    cut_bin_NC_serv_width, df = serv_width_ratings(df)
    cut_bin_NC_serv_quant, df = serv_quant_ratings(df)
    cut_bin_NC_serv_hybrid, df = serv_hybrid_ratings(df, col_name, cutoff)
    cut_bin_NC_serv_width_ln, df = serv_width_ratings_ln(df)
    cut_bin_NC_serv_quant_ln, df = serv_quant_ratings_ln(df)
    cut_bin_NC_serv_hybrid_ln, df = serv_hybrid_ratings_ln(
        df, col_name_ln, cutoff)
    return df, cut_bin_NC_serv_width, cut_bin_NC_serv_quant, cut_bin_NC_serv_hybrid, cut_bin_NC_serv_width_ln, cut_bin_NC_serv_quant_ln, cut_bin_NC_serv_hybrid_ln

# Varying the cut-off value in hybrid scenarios


def assign_hybrid_ratings(df, col_name1, col_name2, col_name_ln1, col_name_ln2, cutoff):
    cut_bin_NC_kg_hybrid, df = kg_hybrid_ratings(df, col_name1, cutoff)
    cut_bin_NC_serv_hybrid, df = serv_hybrid_ratings(df, col_name2, cutoff)
    cut_bin_NC_kg_hybrid_ln, df = kg_hybrid_ratings_ln(
        df, col_name_ln1, cutoff)
    cut_bin_NC_serv_hybrid_ln, df = serv_hybrid_ratings_ln(
        df, col_name_ln2, cutoff)
    return df, cut_bin_NC_kg_hybrid, cut_bin_NC_serv_hybrid, cut_bin_NC_kg_hybrid_ln, cut_bin_NC_serv_hybrid_ln


# Calculating median class values and upper thresholds to calculate change in single score required to access the next rating class up

def get_median_values_serv(rating, df):
    """
    This function takes in a rating and a DataFrame and returns a list of median values for this rating class, for all variations of the hybrid scenario (5%, 10% and 15%).
    """
    return [df[df[f'Score_NC_serv_hybrid{num}'] == rating]['Single score per serving'].median() for num in [5, 10, 15]]


def get_median_values_serv_ln(rating, df):
    """
    This function takes in a rating and a DataFrame and returns a list of median values for this rating class, for all variations of the hybrid scenario (5%, 10% and 15%).
    """
    return [df[df[f'Score_NC_serv_hybrid{num}_ln'] == rating]['Single score per serving'].median() for num in [5, 10, 15]]


def get_median_values_kg(rating, df):
    """
    This function takes in a rating and a DataFrame and returns a list of median values for this rating class, for all variations of the hybrid scenario (5%, 10% and 15%).
    """
    return [df[df[f'Score_NC_kg_hybrid{num}'] == rating]['Single score per kg'].median() for num in [5, 10, 15]]


def get_median_values_kg_ln(rating, df):
    """
    This function takes in a rating and a DataFrame and returns a list of median values for this rating class, for all variations of the hybrid scenario (5%, 10% and 15%).
    """
    return [df[df[f'Score_NC_kg_hybrid{num}_ln'] == rating]['Single score per kg'].median() for num in [5, 10, 15]]


def get_thresholds_values(index, hybrid5_thresholds, hybrid10_thresholds, hybrid15_thresholds):
    """
    This function returns a list of values representing the upper thresholds of a rating class, for all variations of the hybrid scenario (5%, 10% and 15%). The rating class is dictacted by the index value: A (index=1), B (index=2), C (index=3) and D (index=4).
    """
    bins = [hybrid5_thresholds, hybrid10_thresholds, hybrid15_thresholds]
    return [bin[index] for bin in bins]

# Calculating % products obtaining a different rating when all duplicates are removed from the dataset


def sensitivity_duplicates(df):
    ''' Removes all duplicate values (within a similar food group) from dataset df and recalculates performance classes thresholds for all scenarios. The % of products being awarded a different rating is calculated for each scenario and returned in a dataframe (one for each functional unit)'''
    # Remove duplicate values within a similar food group in df
    AGB31_0dup = df.drop_duplicates(
        ['Climate change per kg', 'Single score per kg', 'Serving size (g)', 'Food group'], keep='first')
    AGB31_0dup.info()

    # Assign ratings for each scenario (S1, S2, S3) to 0dup dataset (n=1329) - with 10% A and E cutoff
    AGB31_0dup, bins_kg_width_0dup, bins_kg_quant_0dup, bins_kg_hybrid10_0dup, bins_kg_width_ln_0dup, bins_kg_quant_ln_0dup, bins_kg_hybrid10_ln_0dup = assign_kg_ratings(
        AGB31_0dup, 'Score_NC_kg_hybrid10', 'Score_NC_kg_hybrid10_ln', 0.1)
    AGB31_0dup, bins_serv_width_0dup, bins_serv_quant_0dup, bins_serv_hybrid10_0dup, bins_serv_width_ln_0dup, bins_serv_quant_ln_0dup, bins_serv_hybrid10_ln_0dup = assign_serv_ratings(
        AGB31_0dup, 'Score_NC_serv_hybrid10', 'Score_NC_serv_hybrid10_ln', 0.1)

    # Assign ratings under hybrid scenario (S3) varying the cut-off value (5% and 15%)
    AGB31_0dup, bins_kg_hybrid5_0dup, bins_serv_hybrid5_0dup, bins_kg_hybrid5_ln_0dup, bins_serv_hybrid5_ln_0dup = assign_hybrid_ratings(
        AGB31_0dup, 'Score_NC_kg_hybrid5', 'Score_NC_serv_hybrid5', 'Score_NC_kg_hybrid5_ln', 'Score_NC_serv_hybrid5_ln', 0.05)
    AGB31_0dup, bins_kg_hybrid15_0dup, bins_serv_hybrid15_0dup, bins_kg_hybrid15_ln_0dup, bins_serv_hybrid15_ln_0dup = assign_hybrid_ratings(
        AGB31_0dup, 'Score_NC_kg_hybrid15', 'Score_NC_serv_hybrid15', 'Score_NC_kg_hybrid15_ln', 'Score_NC_serv_hybrid15_ln', 0.15)

    # Merge dataframes to compare ratings
    merged_dup = AGB31_0dup.merge(
        df, on='Food item in AGB 3.1 (short name)', suffixes=('_dup', ''))

    # Identify  products with a different rating
    diff_list_serv = []
    diff_width_dup = merged_dup.loc[merged_dup['Score_NC_serv_width']
                                    != merged_dup['Score_NC_serv_width_dup']]
    diff_quant_dup = merged_dup.loc[merged_dup['Score_NC_serv_quant']
                                    != merged_dup['Score_NC_serv_quant_dup']]
    diff_width_ln_dup = merged_dup.loc[merged_dup['Score_NC_serv_width_ln']
                                       != merged_dup['Score_NC_serv_width_ln_dup']]
    diff_quant_ln_dup = merged_dup.loc[merged_dup['Score_NC_serv_quant_ln']
                                       != merged_dup['Score_NC_serv_quant_ln_dup']]
    diff_hybrid5_dup = merged_dup.loc[merged_dup['Score_NC_serv_hybrid5']
                                      != merged_dup['Score_NC_serv_hybrid5_dup']]
    diff_hybrid10_dup = merged_dup.loc[merged_dup['Score_NC_serv_hybrid10']
                                       != merged_dup['Score_NC_serv_hybrid10_dup']]
    diff_hybrid15_dup = merged_dup.loc[merged_dup['Score_NC_serv_hybrid15']
                                       != merged_dup['Score_NC_serv_hybrid15_dup']]
    diff_hybrid5_ln_dup = merged_dup.loc[merged_dup['Score_NC_serv_hybrid5_ln']
                                         != merged_dup['Score_NC_serv_hybrid5_ln_dup']]
    diff_hybrid10_ln_dup = merged_dup.loc[merged_dup['Score_NC_serv_hybrid10_ln']
                                          != merged_dup['Score_NC_serv_hybrid10_ln_dup']]
    diff_hybrid15_ln_dup = merged_dup.loc[merged_dup['Score_NC_serv_hybrid15_ln']
                                          != merged_dup['Score_NC_serv_hybrid15_ln_dup']]

    diff_list_kg = []
    diff_width_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_width']
                                       != merged_dup['Score_NC_kg_width_dup']]
    diff_quant_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_quant']
                                       != merged_dup['Score_NC_kg_quant_dup']]
    diff_width_ln_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_width_ln']
                                          != merged_dup['Score_NC_kg_width_ln_dup']]
    diff_quant_ln_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_quant_ln']
                                          != merged_dup['Score_NC_kg_quant_ln_dup']]
    diff_hybrid5_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_hybrid5']
                                         != merged_dup['Score_NC_kg_hybrid5_dup']]
    diff_hybrid10_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_hybrid10']
                                          != merged_dup['Score_NC_kg_hybrid10_dup']]
    diff_hybrid15_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_hybrid15']
                                          != merged_dup['Score_NC_kg_hybrid15_dup']]
    diff_hybrid5_ln_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_hybrid5_ln']
                                            != merged_dup['Score_NC_kg_hybrid5_ln_dup']]
    diff_hybrid10_ln_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_hybrid10_ln']
                                             != merged_dup['Score_NC_kg_hybrid10_ln_dup']]
    diff_hybrid15_ln_dup_kg = merged_dup.loc[merged_dup['Score_NC_kg_hybrid15_ln']
                                             != merged_dup['Score_NC_kg_hybrid15_ln_dup']]

    # Organise results into a long format dataframe results_dup_long
    diff_list_serv.append([len(diff_width_dup), len(diff_quant_dup), len(diff_width_ln_dup), len(diff_quant_ln_dup), len(diff_hybrid5_dup), len(
        diff_hybrid10_dup), len(diff_hybrid15_dup), len(diff_hybrid5_ln_dup), len(diff_hybrid10_ln_dup), len(diff_hybrid15_ln_dup)])
    diff_list_kg.append([len(diff_width_dup_kg), len(diff_quant_dup_kg), len(diff_width_ln_dup_kg), len(diff_quant_ln_dup_kg), len(diff_hybrid5_dup_kg), len(
        diff_hybrid10_dup_kg), len(diff_hybrid15_dup_kg), len(diff_hybrid5_ln_dup_kg), len(diff_hybrid10_ln_dup_kg), len(diff_hybrid15_ln_dup_kg)])

    cols = ['Width-based', 'Quantiles-based', 'Width-based_ln', 'Quantiles-based_ln',
            'Hybrid_5%', 'Hybrid_10%', 'Hybrid_15%', 'Hybrid_5%_ln', 'Hybrid_10%_ln', 'Hybrid_15%_ln']
    results_dup = pd.DataFrame(columns=cols, data=diff_list_serv)
    results_dup_kg = pd.DataFrame(columns=cols, data=diff_list_kg)

    for col in cols:
        results_dup[col+'_norm'] = (results_dup[col]/len(AGB31_0dup))*100
        results_dup_kg[col+'_norm'] = (results_dup_kg[col]/len(AGB31_0dup))*100

    results_dup_long = pd.melt(results_dup[['Width-based_norm', 'Quantiles-based_norm', 'Width-based_ln_norm', 'Quantiles-based_ln_norm', 'Hybrid_5%_norm', 'Hybrid_10%_norm', 'Hybrid_15%_norm', 'Hybrid_5%_ln_norm', 'Hybrid_10%_ln_norm', 'Hybrid_15%_ln_norm']], value_vars=[
                               'Width-based_norm', 'Quantiles-based_norm', 'Width-based_ln_norm', 'Quantiles-based_ln_norm', 'Hybrid_5%_norm', 'Hybrid_10%_norm', 'Hybrid_15%_norm', 'Hybrid_5%_ln_norm', 'Hybrid_10%_ln_norm', 'Hybrid_15%_ln_norm'], var_name='Scenario', value_name='%_products')
    results_dup_long_kg = pd.melt(results_dup_kg[['Width-based_norm', 'Quantiles-based_norm', 'Width-based_ln_norm', 'Quantiles-based_ln_norm', 'Hybrid_5%_norm', 'Hybrid_10%_norm', 'Hybrid_15%_norm', 'Hybrid_5%_ln_norm', 'Hybrid_10%_ln_norm', 'Hybrid_15%_ln_norm']], value_vars=[
                                  'Width-based_norm', 'Quantiles-based_norm', 'Width-based_ln_norm', 'Quantiles-based_ln_norm', 'Hybrid_5%_norm', 'Hybrid_10%_norm', 'Hybrid_15%_norm', 'Hybrid_5%_ln_norm', 'Hybrid_10%_ln_norm', 'Hybrid_15%_ln_norm'], var_name='Scenario', value_name='%_products')

    return merged_dup, results_dup_long_kg, results_dup_long

# Calculate % products obtaining a different rating when they are established on subsets rather than the full dataset

    # Required functions


def subsetting(to_drop, df):
    '''Subsetting df dataframe by dropping a fraction (to_drop) of the rows, returns the subset and prints the number of rows in the subset'''
    df_subset = df.sample(frac=1 - to_drop)
    return df_subset


def norm_diff(diff_df, sub_df):
    ''' normalises the average number of products having a different rating to a percentage based on the number of products in sub_df'''
    diff_df['S1'] = (diff_df['Width-based']/len(sub_df))*100
    diff_df['S2'] = (diff_df['Quantiles-based']/len(sub_df))*100
    diff_df['S3_10'] = (diff_df['Hybrid_10%']/len(sub_df))*100
    diff_df['S1_ln'] = (diff_df['Width-based_ln']/len(sub_df))*100
    diff_df['S2_ln'] = (diff_df['Quantiles-based_ln']/len(sub_df))*100
    diff_df['S3_10_ln'] = (diff_df['Hybrid_10%_ln']/len(sub_df))*100
    diff_df['S3_5'] = (diff_df['Hybrid_5%']/len(sub_df))*100
    diff_df['S3_5_ln'] = (diff_df['Hybrid_5%_ln']/len(sub_df))*100
    diff_df['S3_15'] = (diff_df['Hybrid_15%']/len(sub_df))*100
    diff_df['S3_15_ln'] = (diff_df['Hybrid_15%_ln']/len(sub_df))*100
    return diff_df

    # Resulting function


def sensitivity_subset_serv(nb_runs, df):
    '''Generate three subsets fractions (10%, 25%, 50%) of dataset df randomly nb_runs times. Rating classes boundaries are calculated for each subset, and ratings awarded. Differences in ratings compared to the full dataset ae calculated for each scenario. Returns a dataframe with the results'''

    cols = ['Width-based', 'Quantiles-based', 'Hybrid_10%', 'Width-based_ln', 'Quantiles-based_ln',
            'Hybrid_10%_ln', 'Hybrid_5%', 'Hybrid_5%_ln', 'Hybrid_15%', 'Hybrid_15%_ln']

    # Initialise empty dataframes and setting fractions to be removed to generate subsets
    results_diff_10 = pd.DataFrame(columns=cols, index=range(nb_runs))
    results_diff_25 = pd.DataFrame(columns=cols, index=range(nb_runs))
    results_diff_50 = pd.DataFrame(columns=cols, index=range(nb_runs))
    subsets = [0.1, 0.25, 0.5]  # fractions of df to be removed

    # Loop and append results for each subset to corresponding df
    for i in range(nb_runs):
        np.random.seed(i)
        dct = {}

        # Generating the three subsets and assigning ratings for each scenario
        for sub in subsets:
            subset = subsetting(sub, df)
            subset, bins_serv_width_sub, bins_serv_quant_sub, bins_serv_hybrid10_sub, bins_serv_width_ln_sub, bins_serv_quant_ln_sub, bins_serv_hybrid10_ln_sub = assign_serv_ratings(
                subset, 'Score_NC_serv_hybrid10', 'Score_NC_serv_hybrid10_ln', 0.1)
            bins_serv_hybrid5_sub, subset = serv_hybrid_ratings(
                subset, 'Score_NC_serv_hybrid5', 0.05)
            bins_serv_hybrid15_sub, subset = serv_hybrid_ratings(
                subset, 'Score_NC_serv_hybrid15', 0.15)
            bins_serv_hybrid5_ln_sub, subset = serv_hybrid_ratings_ln(
                subset, 'Score_NC_serv_hybrid5_ln', 0.05)
            bins_serv_hybrid15_ln_sub, subset = serv_hybrid_ratings_ln(
                subset, 'Score_NC_serv_hybrid15_ln', 0.15)
            bins_dict = {'serv_width_sub': bins_serv_width_sub, 'serv_quant_sub': bins_serv_quant_sub, 'serv_hybrid10_sub': bins_serv_hybrid10_sub, 'serv_width_ln_sub': bins_serv_width_ln_sub, 'serv_quant_ln_sub': bins_serv_quant_ln_sub,
                         'serv_hybrid10_ln_sub': bins_serv_hybrid10_ln_sub, 'serv_hybrid5_sub': bins_serv_hybrid5_sub, 'serv_hybrid15_sub': bins_serv_hybrid15_sub, 'serv_hybrid5_ln_sub': bins_serv_hybrid5_ln_sub, 'serv_hybrid15_ln_sub': bins_serv_hybrid15_ln_sub}
            dct['diff_%s' % sub] = []

            for key, value in bins_dict.items():
                df['Score_NC_'+str(key)] = pd.cut(x=df['Single score per serving'],
                                                  bins=value, labels=labels, include_lowest=True)

            # Identifying products with different ratings in df and in the different subsets for each scenario
            diff_width_sub = df.loc[df['Score_NC_serv_width']
                                    != df['Score_NC_serv_width_sub']]
            diff_quant_sub = df.loc[df['Score_NC_serv_quant']
                                    != df['Score_NC_serv_quant_sub']]
            diff_hybrid10_sub = df.loc[df['Score_NC_serv_hybrid10']
                                       != df['Score_NC_serv_hybrid10_sub']]
            diff_width_ln_sub = df.loc[df['Score_NC_serv_width_ln']
                                       != df['Score_NC_serv_width_ln_sub']]
            diff_quant_ln_sub = df.loc[df['Score_NC_serv_quant_ln']
                                       != df['Score_NC_serv_quant_ln_sub']]
            diff_hybrid10_ln_sub = df.loc[df['Score_NC_serv_hybrid10_ln']
                                          != df['Score_NC_serv_hybrid10_ln_sub']]
            diff_hybrid5_sub = df.loc[df['Score_NC_serv_hybrid5']
                                      != df['Score_NC_serv_hybrid5_sub']]
            diff_hybrid5_ln_sub = df.loc[df['Score_NC_serv_hybrid5_ln']
                                         != df['Score_NC_serv_hybrid5_ln_sub']]
            diff_hybrid15_sub = df.loc[df['Score_NC_serv_hybrid15']
                                       != df['Score_NC_serv_hybrid15_sub']]
            diff_hybrid15_ln_sub = df.loc[df['Score_NC_serv_hybrid15_ln']
                                          != df['Score_NC_serv_hybrid15_ln_sub']]

            # Saving results in lists and adding list to created dataframes
            dct['diff_%s' % sub].append([len(diff_width_sub), len(diff_quant_sub), len(diff_hybrid10_sub), len(diff_width_ln_sub), len(diff_quant_ln_sub), len(
                diff_hybrid10_ln_sub), len(diff_hybrid5_sub), len(diff_hybrid5_ln_sub), len(diff_hybrid15_sub), len(diff_hybrid15_ln_sub)])

        results_diff_10.loc[i] = dct['diff_0.1'][0]
        results_diff_25.loc[i] = dct['diff_0.25'][0]
        results_diff_50.loc[i] = dct['diff_0.5'][0]

    # Calculating normalised results to obtain a % of products with a different rating (Number of products obtaining a different rating for a subset / total number of products in the subset)
    results_diff_10 = norm_diff(results_diff_10, df)
    results_diff_25 = norm_diff(results_diff_25, df)
    results_diff_50 = norm_diff(results_diff_50, df)

    # Transforming data wide to long and concatenating to obtain a single dataframe
    results_diff_10_long = pd.melt(results_diff_10[['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln']], value_vars=[
                                   'S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln'], var_name='Scenario', value_name='%_products')
    results_diff_25_long = pd.melt(results_diff_25[['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln']], value_vars=[
                                   'S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln'], var_name='Scenario', value_name='%_products')
    results_diff_50_long = pd.melt(results_diff_50[['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln']], value_vars=[
                                   'S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln'], var_name='Scenario', value_name='%_products')

    concat_results = pd.concat([results_diff_10_long, results_diff_25_long, results_diff_50_long], keys=[
                               '90% subset', '75% subset', '50% subset']).reset_index()
    return concat_results, results_diff_10.mean(), results_diff_25.mean(), results_diff_50.mean()


def sensitivity_subset_kg(nb_runs, df):
    '''
    Generate three subsets fractions (10%, 25%, 50%) of dataset df randomly nb_runs times. Rating classes boundaries are calculated for each subset, and ratings awarded. Differences in ratings compared to the full dataset ae calculated for each scenario. Returns a dataframe with the results
    '''

    cols = ['Width-based', 'Quantiles-based', 'Hybrid_10%', 'Width-based_ln', 'Quantiles-based_ln',
            'Hybrid_10%_ln', 'Hybrid_5%', 'Hybrid_5%_ln', 'Hybrid_15%', 'Hybrid_15%_ln']

    # Initialise empty dataframes and setting fractions to be removed to generate subsets
    results_diff_10 = pd.DataFrame(columns=cols, index=range(nb_runs))
    results_diff_25 = pd.DataFrame(columns=cols, index=range(nb_runs))
    results_diff_50 = pd.DataFrame(columns=cols, index=range(nb_runs))
    subsets = [0.1, 0.25, 0.5]  # fractions of df to be removed

    # Loop and append results for each subset to corresponding df
    for i in range(nb_runs):
        np.random.seed(i)
        dct = {}

        # Generating the three subsets and assigning ratings for each scenario
        for sub in subsets:
            subset = subsetting(sub, df)
            subset, bins_kg_width_sub, bins_kg_quant_sub, bins_kg_hybrid10_sub, bins_kg_width_ln_sub, bins_kg_quant_ln_sub, bins_kg_hybrid10_ln_sub = assign_kg_ratings(
                subset, 'Score_NC_kg_hybrid10', 'Score_NC_kg_hybrid10_ln', 0.1)
            bins_kg_hybrid5_sub, subset = kg_hybrid_ratings(
                subset, 'Score_NC_kg_hybrid5', 0.05)
            bins_kg_hybrid15_sub, subset = kg_hybrid_ratings(
                subset, 'Score_NC_kg_hybrid15', 0.15)
            bins_kg_hybrid5_ln_sub, subset = kg_hybrid_ratings_ln(
                subset, 'Score_NC_kg_hybrid5_ln', 0.05)
            bins_kg_hybrid15_ln_sub, subset = kg_hybrid_ratings_ln(
                subset, 'Score_NC_kg_hybrid15_ln', 0.15)
            bins_dict = {'kg_width_sub': bins_kg_width_sub, 'kg_quant_sub': bins_kg_quant_sub, 'kg_hybrid10_sub': bins_kg_hybrid10_sub, 'kg_width_ln_sub': bins_kg_width_ln_sub, 'kg_quant_ln_sub': bins_kg_quant_ln_sub,
                         'kg_hybrid10_ln_sub': bins_kg_hybrid10_ln_sub, 'kg_hybrid5_sub': bins_kg_hybrid5_sub, 'kg_hybrid15_sub': bins_kg_hybrid15_sub, 'kg_hybrid5_ln_sub': bins_kg_hybrid5_ln_sub, 'kg_hybrid15_ln_sub': bins_kg_hybrid15_ln_sub}
            dct['diff_%s' % sub] = []

            for key, value in bins_dict.items():
                df['Score_NC_'+str(key)] = pd.cut(x=df['Single score per kg'],
                                                  bins=value, labels=labels, include_lowest=True)

            # Identifying products with different ratings in df and in the different subsets for each scenario
            diff_width_sub = df.loc[df['Score_NC_kg_width']
                                    != df['Score_NC_kg_width_sub']]
            diff_quant_sub = df.loc[df['Score_NC_kg_quant']
                                    != df['Score_NC_kg_quant_sub']]
            diff_hybrid10_sub = df.loc[df['Score_NC_kg_hybrid10']
                                       != df['Score_NC_kg_hybrid10_sub']]
            diff_width_ln_sub = df.loc[df['Score_NC_kg_width_ln']
                                       != df['Score_NC_kg_width_ln_sub']]
            diff_quant_ln_sub = df.loc[df['Score_NC_kg_quant_ln']
                                       != df['Score_NC_kg_quant_ln_sub']]
            diff_hybrid10_ln_sub = df.loc[df['Score_NC_kg_hybrid10_ln']
                                          != df['Score_NC_kg_hybrid10_ln_sub']]
            diff_hybrid5_sub = df.loc[df['Score_NC_kg_hybrid5']
                                      != df['Score_NC_kg_hybrid5_sub']]
            diff_hybrid5_ln_sub = df.loc[df['Score_NC_kg_hybrid5_ln']
                                         != df['Score_NC_kg_hybrid5_ln_sub']]
            diff_hybrid15_sub = df.loc[df['Score_NC_kg_hybrid15']
                                       != df['Score_NC_kg_hybrid15_sub']]
            diff_hybrid15_ln_sub = df.loc[df['Score_NC_kg_hybrid15_ln']
                                          != df['Score_NC_kg_hybrid15_ln_sub']]

            # Saving results in lists and adding list to created dataframes
            dct['diff_%s' % sub].append([len(diff_width_sub), len(diff_quant_sub), len(diff_hybrid10_sub), len(diff_width_ln_sub), len(diff_quant_ln_sub), len(
                diff_hybrid10_ln_sub), len(diff_hybrid5_sub), len(diff_hybrid5_ln_sub), len(diff_hybrid15_sub), len(diff_hybrid15_ln_sub)])

        results_diff_10.loc[i] = dct['diff_0.1'][0]
        results_diff_25.loc[i] = dct['diff_0.25'][0]
        results_diff_50.loc[i] = dct['diff_0.5'][0]

    # Calculating normalised results to obtain a % of products with a different rating (Number of products obtaining a different rating for a subset / total number of products in the subset)
    results_diff_10 = norm_diff(results_diff_10, df)
    results_diff_25 = norm_diff(results_diff_25, df)
    results_diff_50 = norm_diff(results_diff_50, df)
    print(results_diff_10.mean())
    print(results_diff_25.mean())
    print(results_diff_50.mean())

    # Transforming data wide to long and concatenating to obtain a single dataframe
    results_diff_10_long = pd.melt(results_diff_10[['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln']], value_vars=[
                                   'S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln'], var_name='Scenario', value_name='%_products')
    results_diff_25_long = pd.melt(results_diff_25[['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln']], value_vars=[
                                   'S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln'], var_name='Scenario', value_name='%_products')
    results_diff_50_long = pd.melt(results_diff_50[['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln']], value_vars=[
                                   'S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln', 'S3_5', 'S3_5_ln', 'S3_15', 'S3_15_ln'], var_name='Scenario', value_name='%_products')

    concat_results = pd.concat([results_diff_10_long, results_diff_25_long, results_diff_50_long], keys=[
                               '90% subset', '75% subset', '50% subset']).reset_index()
    return concat_results, results_diff_10.mean(), results_diff_25.mean(), results_diff_50.mean()
