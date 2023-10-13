# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:16:22 2023

@author: Maelys.Courtat2
"""
# Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from V4_functions_script import *
import os
import warnings

warnings.filterwarnings('ignore') # don't show SettingWithCopyWarnings or FutureWarnings

# Pick where to save the figures

chart_dir = get_chart_dir()


def save(f, fn):
    ''' Saves figure f in the location specified in chart_dir under the name 'fn' '''
    f.savefig(os.path.join(chart_dir, fn))

# Data preparation and ratings calculation


def import_calcs():
    ''' imports the Agribalyse 3.1 dataset and assigns ratings to products based on the thresholding scenario selected'''
    # Import dataset as AGB31
    AGB31 = import_AGB31()

    # Assigning ratings
    # Assigning ratings under the three thresholding scenarios (S1, S2, S3 10%) to full dataset (n=2253) - per kg data
    AGB31, bins_kg_width, bins_kg_quant, bins_kg_hybrid10, bins_kg_width_ln, bins_kg_quant_ln, bins_kg_hybrid10_ln = assign_kg_ratings(
        AGB31, 'Score_NC_kg_hybrid10', 'Score_NC_kg_hybrid10_ln', 0.1)

    # Assigning ratings under the three thresholding scenarios (S1, S2, S3 10%) to full dataset (n=2253) - per serving data
    AGB31, bins_serv_width, bins_serv_quant, bins_serv_hybrid10, bins_serv_width_ln, bins_serv_quant_ln, bins_serv_hybrid10_ln = assign_serv_ratings(
        AGB31, 'Score_NC_serv_hybrid10', 'Score_NC_serv_hybrid10_ln', 0.1)

    # Assigning ratings under hybrid scenario varying the cut-off value (5% and 15%)
    AGB31, bins_kg_hybrid5, bins_serv_hybrid5, bins_kg_hybrid5_ln, bins_serv_hybrid5_ln = assign_hybrid_ratings(
        AGB31, 'Score_NC_kg_hybrid5', 'Score_NC_serv_hybrid5', 'Score_NC_kg_hybrid5_ln', 'Score_NC_serv_hybrid5_ln', 0.05)
    AGB31, bins_kg_hybrid15, bins_serv_hybrid15, bins_kg_hybrid15_ln, bins_serv_hybrid15_ln = assign_hybrid_ratings(
        AGB31, 'Score_NC_kg_hybrid15', 'Score_NC_serv_hybrid15', 'Score_NC_kg_hybrid15_ln', 'Score_NC_serv_hybrid15_ln', 0.15)

    # Writing resulting AGB31 dataframe to Excel
    AGB31.to_excel(os.path.join(chart_dir, "V4_output_ratings.xlsx"))

    return AGB31, bins_kg_width, bins_kg_quant, bins_kg_hybrid10, bins_kg_width_ln, bins_kg_quant_ln, bins_kg_hybrid10_ln, bins_serv_width, bins_serv_quant, bins_serv_hybrid10, bins_serv_width_ln, bins_serv_quant_ln, bins_serv_hybrid10_ln, bins_kg_hybrid5, bins_serv_hybrid5, bins_kg_hybrid5_ln, bins_serv_hybrid5_ln, bins_kg_hybrid15, bins_serv_hybrid15, bins_kg_hybrid15_ln, bins_serv_hybrid15_ln


# Visualisations - Main text
colors = ['#00B050', '#92D050', '#FFC000', '#F4B183', '#F53D76']

# Figure 3 - Data distribution under the various functional unit and data linearity scenarios


def figure3():
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=15)
    sns.set_palette('Spectral')
    f, axs = plt.subplots(4, 2, figsize=(18, 13), gridspec_kw={
        "height_ratios": (.15, .85, .15, .85)})
    sns.boxplot(data=AGB31, x='Single score per kg', ax=axs[0, 0])
    sns.histplot(data=AGB31, x='Single score per kg', alpha=0.5, ax=axs[1, 0])
    sns.boxplot(data=AGB31, x='Log (single score per kg)', ax=axs[0, 1])
    sns.histplot(data=AGB31, x='Log (single score per kg)',
                 alpha=0.5, ax=axs[1, 1])
    sns.boxplot(data=AGB31, x='Single score per serving', ax=axs[2, 0])
    sns.histplot(data=AGB31, x='Single score per serving',
                 alpha=0.5, ax=axs[3, 0])
    sns.boxplot(data=AGB31, x='Log (single score per serving)', ax=axs[2, 1])
    sns.histplot(data=AGB31, x='Log (single score per serving)',
                 alpha=0.5, ax=axs[3, 1])
    axs[0, 0].set_title('a. FU: per kilogram, linear data',
                        fontsize=19, fontweight='semibold')
    axs[0, 1].set_title('b. FU: per kilogram, log-transformed data',
                        fontsize=19, fontweight='semibold')
    axs[2, 0].set_title('c. FU: per serving, linear data',
                        fontsize=19, fontweight='semibold')
    axs[2, 1].set_title('d. FU: per serving, log-transformed data',
                        fontsize=19, fontweight='semibold')
    sns.despine(ax=axs[1, 0])
    sns.despine(ax=axs[1, 1])
    sns.despine(ax=axs[3, 0])
    sns.despine(ax=axs[3, 1])
    sns.despine(ax=axs[0, 0], left=True)
    sns.despine(ax=axs[0, 1], left=True)
    sns.despine(ax=axs[2, 0], left=True)
    sns.despine(ax=axs[2, 1], left=True)
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xlabel('')
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xlabel('')
    axs[2, 0].set_yticks([])
    axs[2, 0].set_xlabel('')
    axs[2, 1].set_yticks([])
    axs[2, 1].set_xlabel('')
    axs[0, 0].sharex(axs[1, 0])
    axs[0, 1].sharex(axs[1, 1])
    axs[2, 0].sharex(axs[3, 0])
    axs[2, 1].sharex(axs[3, 1])
    axs[1, 1].sharey(axs[1, 0])
    axs[3, 1].sharey(axs[3, 0])
    f.tight_layout()

    save(f, 'figure3_main.png')

    # Figure 4 - comparing how ratings are distributed across the three scenarios - both functional units


def figure4():
    sns.set_context('talk')
    f, axs = plt.subplots(2, 3, figsize=(18, 13), sharey=True)
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_width,
                 hue='Score_NC_kg_width', palette=colors, alpha=0.5, legend=False, ax=axs[0, 0])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_quant,
                 hue='Score_NC_kg_quant', palette=colors, alpha=0.5, legend=False, ax=axs[0, 1])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid10,
                 hue='Score_NC_kg_hybrid10', palette=colors, alpha=0.5, legend=False, ax=axs[0, 2])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_width,
                 hue='Score_NC_serv_width', palette=colors, alpha=0.5, legend=False, ax=axs[1, 0])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_quant,
                 hue='Score_NC_serv_quant', palette=colors, alpha=0.5, legend=False, ax=axs[1, 1])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid10,
                 hue='Score_NC_serv_hybrid10', palette=colors, alpha=0.5, legend=True, ax=axs[1, 2])
    axs[0, 0].set_title('a. Width-based (S1), per kg data',
                        fontsize=17, fontweight='semibold')
    axs[0, 1].set_title('b. Quantiles-based (S2), per kg data',
                        fontsize=17, fontweight='semibold')
    axs[0, 2].set_title('c. Hybrid 10% A/E (S3), per kg data',
                        fontsize=17, fontweight='semibold')
    axs[1, 0].set_title('d. Width-based (S1), per serving data',
                        fontsize=17, fontweight='semibold')
    axs[1, 1].set_title('e. Quantiles-based (S2), per serving data',
                        fontsize=17, fontweight='semibold')
    axs[1, 2].set_title('f. Hybrid 10% A/E (S3), per serving data',
                        fontsize=17, fontweight='semibold')
    for ax in axs[0, :].flatten():
        ax.vlines(x=[AGB31['Single score per kg'].mean(), AGB31['Single score per kg'].median(
        )], ymin=0, ymax=2300, colors=['red', 'blue'], lw=1, label=['mean', 'median'])
    for ax in axs[1, :].flatten():
        ax.vlines(x=[AGB31['Single score per serving'].mean(), AGB31['Single score per serving'].median(
        )], ymin=0, ymax=2300, colors=['red', 'blue'], lw=1, label=['mean', 'median'])
    f.tight_layout()

    save(f, 'figure4_main')

    # Generating pie charts
    cols = ['Score_NC_kg_width', 'Score_NC_kg_quant', 'Score_NC_kg_hybrid10',
            'Score_NC_serv_width', 'Score_NC_serv_quant', 'Score_NC_serv_hybrid10']

    for col in cols:
        plt.clf()
        data = AGB31[col].value_counts().sort_index()
        plt.pie(data, labels=labels, colors=colors, wedgeprops=dict(alpha=0.5, linewidth=2,
                edgecolor='grey'), textprops={'size': 17}, autopct='%1.1f%%', pctdistance=1.3, labeldistance=.7)
        plt.savefig(os.path.join(chart_dir, 'figure4_pie_{}.png'.format(col)))
        # plt.show()

    # Figure 5 - comparing how thresholds and ratings are influenced by log transformation of the data


def figure5():
    sns.set_context('talk')
    f, axs = plt.subplots(2, 3, figsize=(
        18, 13), sharey=True, layout='constrained')
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_width_ln,
                 hue='Score_NC_kg_width_ln', palette=colors, alpha=0.5, legend=False, ax=axs[0, 0])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_quant_ln,
                 hue='Score_NC_kg_quant_ln', palette=colors, alpha=0.5, legend=False, ax=axs[0, 1])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid10_ln,
                 hue='Score_NC_kg_hybrid10_ln', palette=colors, alpha=0.5, legend=False, ax=axs[0, 2])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_width_ln,
                 hue='Score_NC_serv_width_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 0])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_quant_ln,
                 hue='Score_NC_serv_quant_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 1])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid10_ln,
                 hue='Score_NC_serv_hybrid10_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 2])
    axs[0, 0].set_title('a. Width-based (S1), per kg data',
                        fontsize=17, fontweight='semibold')
    axs[0, 1].set_title('b. Quantiles-based (S2), per kg data',
                        fontsize=17, fontweight='semibold')
    axs[0, 2].set_title('c. Hybrid 10% A/E (S3), per kg data',
                        fontsize=17, fontweight='semibold')
    axs[1, 0].set_title('d. Width-based (S1), per serving data',
                        fontsize=17, fontweight='semibold')
    axs[1, 1].set_title('e. Quantiles-based (S2), per serving data',
                        fontsize=17, fontweight='semibold')
    axs[1, 2].set_title('f. Hybrid 10% A/E (S3), per serving data',
                        fontsize=17, fontweight='semibold')
    for ax in axs[0, :].flatten():
        ax.vlines(x=[AGB31['Single score per kg'].mean(), AGB31['Single score per kg'].median(
        )], ymin=0, ymax=2300, colors=['red', 'blue'], lw=1, label=['mean', 'median'])
    for ax in axs[1, :].flatten():
        ax.vlines(x=[AGB31['Single score per serving'].mean(), AGB31['Single score per serving'].median(
        )], ymin=0, ymax=2300, colors=['red', 'blue'], lw=1, label=['mean', 'median'])
    f.tight_layout()

    save(f, 'figure5_main')

    # Generating pie charts
    cols_ln = ['Score_NC_kg_width_ln', 'Score_NC_kg_quant_ln', 'Score_NC_kg_hybrid10_ln',
               'Score_NC_serv_width_ln', 'Score_NC_serv_quant_ln', 'Score_NC_serv_hybrid10_ln']
    for col in cols_ln:
        plt.clf()
        data = AGB31[col].value_counts().sort_index()
        plt.pie(data, labels=labels, colors=colors, wedgeprops=dict(alpha=0.5, linewidth=2,
                edgecolor='grey'), textprops={'size': 17}, autopct='%1.1f%%', pctdistance=1.3, labeldistance=.7)
        plt.savefig(os.path.join(chart_dir, 'figure5_pie_{}.png'.format(col)))

    # Figure 6 - comparing how thresholds and ratings are influenced by variation of cut-off value in the hybrid scenario (per serving data)


def figure6():
    sns.set_context('talk')
    f, axs = plt.subplots(2, 3, figsize=(
        18, 13), sharey=True, layout='constrained')
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid5,
                 hue='Score_NC_serv_hybrid5', palette=colors, alpha=0.5, legend=False, ax=axs[0, 0])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid10,
                 hue='Score_NC_serv_hybrid10', palette=colors, alpha=0.5, legend=False, ax=axs[0, 1])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid15,
                 hue='Score_NC_serv_hybrid15', palette=colors, alpha=0.5, legend=False, ax=axs[0, 2])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid5_ln,
                 hue='Score_NC_serv_hybrid5_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 0])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid10_ln,
                 hue='Score_NC_serv_hybrid10_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 1])
    sns.histplot(data=AGB31, x='Single score per serving', bins=bins_serv_hybrid15_ln,
                 hue='Score_NC_serv_hybrid15_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 2])
    axs[0, 0].set_title('a. Hybrid 5% A/E, linear data',
                        fontsize=17, fontweight='semibold')
    axs[0, 1].set_title('b. Hybrid 10% A/E, linear data (S3)',
                        fontsize=17, fontweight='semibold')
    axs[0, 2].set_title('c. Hybrid 15% A/E, linear data',
                        fontsize=17, fontweight='semibold')
    axs[1, 0].set_title('d. Hybrid 5% A/E, log-transformed data',
                        fontsize=17, fontweight='semibold')
    axs[1, 1].set_title('e. Hybrid 10% A/E, log-transformed data',
                        fontsize=17, fontweight='semibold')
    axs[1, 2].set_title('f. Hybrid 15% A/E, log-transformed data',
                        fontsize=17, fontweight='semibold')
    for ax in axs[:, :].flatten():
        ax.vlines(x=[AGB31['Single score per serving'].mean(), AGB31['Single score per serving'].median(
        )], ymin=0, ymax=2300, colors=['red', 'blue'], lw=1, label=['mean', 'median'])
    f.tight_layout()

    save(f, 'figure6_main')

    # Generating pie charts
    cols_cutoff = ['Score_NC_serv_hybrid5', 'Score_NC_serv_hybrid10', 'Score_NC_serv_hybrid15',
                   'Score_NC_serv_hybrid5_ln', 'Score_NC_serv_hybrid10_ln', 'Score_NC_serv_hybrid15_ln']
    for col in cols_cutoff:
        plt.clf()
        data = AGB31[col].value_counts().sort_index()
        plt.pie(data, labels=labels, colors=colors, wedgeprops=dict(alpha=0.5, linewidth=2,
                edgecolor='grey'), textprops={'size': 17}, autopct='%1.1f%%', pctdistance=1.3, labeldistance=.7)
        plt.savefig(os.path.join(chart_dir, 'figure6_pie_{}.png'.format(col)))

    # Figure 7 - Calculating change in single score required to access next rating class up (per serving data)


def figure7():
    # Creating a dictionnary containing the median values and top thresholds of B, C, D and E rating classes for each variation of the hybrid scenario
    dict_median_top_serv = {
        'Scenario': ['Hybrid 5% A/E', 'Hybrid 10% A/E', 'Hybrid 15% A/E'],
        'Median B value': get_median_values_serv('B', AGB31),
        'Median C value': get_median_values_serv('C', AGB31),
        'Median D value': get_median_values_serv('D', AGB31),
        'Median E value': get_median_values_serv('E', AGB31),
        'B/A threshold': get_thresholds_values(1, bins_serv_hybrid5, bins_serv_hybrid10, bins_serv_hybrid15),
        'C/B threshold': get_thresholds_values(2, bins_serv_hybrid5, bins_serv_hybrid10, bins_serv_hybrid15),
        'D/C threshold': get_thresholds_values(3, bins_serv_hybrid5, bins_serv_hybrid10, bins_serv_hybrid15),
        'E/D threshold': get_thresholds_values(4, bins_serv_hybrid5, bins_serv_hybrid10, bins_serv_hybrid15)
    }

    # Calculating the difference between the two values to obtain the change in single score required to access the next performance class (as absolute and relative values)
    Rating_change = pd.DataFrame(dict_median_top_serv)
    Rating_change['B to A (mPts)'] = Rating_change['Median B value'] - \
        Rating_change['B/A threshold']
    Rating_change['C to B (mPts)'] = Rating_change['Median C value'] - \
        Rating_change['C/B threshold']
    Rating_change['D to C (mPts)'] = Rating_change['Median D value'] - \
        Rating_change['D/C threshold']
    Rating_change['E to D (mPts)'] = Rating_change['Median E value'] - \
        Rating_change['E/D threshold']
    Rating_change['B to A (%)'] = Rating_change['B to A (mPts)'] / \
        Rating_change['Median B value'] * 100
    Rating_change['C to B (%)'] = Rating_change['C to B (mPts)'] / \
        Rating_change['Median C value'] * 100
    Rating_change['D to C (%)'] = Rating_change['D to C (mPts)'] / \
        Rating_change['Median D value'] * 100
    Rating_change['E to D (%)'] = Rating_change['E to D (mPts)'] / \
        Rating_change['Median E value'] * 100

    print(Rating_change)

    # Preparing the data for plotting
    data_to_plot = Rating_change[['Scenario', 'B to A (mPts)', 'C to B (mPts)', 'D to C (mPts)',
                                  'E to D (mPts)', 'B to A (%)', 'C to B (%)', 'D to C (%)', 'E to D (%)']]
    data_to_plot = pd.melt(data_to_plot, id_vars='Scenario',
                           var_name='Rating_change', value_name='Change_required')

    # Create a bar plot for mPts using seaborn
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change', data=data_to_plot[data_to_plot['Rating_change'].str.contains(
        'mPts')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax1)

    # Create a bar plot for % using seaborn
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change', data=data_to_plot[data_to_plot['Rating_change'].str.contains(
        '%')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax2)

    # Add labels and title
    plt.xlabel('Scenario', fontsize=20)
    ax1.set_ylabel('Change required (mPts)', fontsize=18)
    ax2.set_ylabel('Change required (%)', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax1.set_xlabel('')

    # Show a single legend in the top left corner
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['B to A', 'C to B', 'D to C', 'E to D']
    ax1.legend(handles[:4], labels, title='Rating change',
               loc='upper left', title_fontsize=16, fontsize=16)
    ax2.get_legend().remove()
    for i in ax1.containers:
        ax1.bar_label(i, fmt='%.3f', fontsize=14)
    for i in ax2.containers:
        ax2.bar_label(i, fmt='%.1f', fontsize=14)

    # Show the plot
    plt.gcf().set_size_inches(16, 12)
    save(f, 'figure7_main')

    # For scenarios applied to log-transformed data


def figure7_ln():
    dict_median_top_serv_ln = {
        'Scenario': ['Hybrid 5% A/E', 'Hybrid 10% A/E', 'Hybrid 15% A/E'],
        'Median B value': get_median_values_serv_ln('B', AGB31),
        'Median C value': get_median_values_serv_ln('C', AGB31),
        'Median D value': get_median_values_serv_ln('D', AGB31),
        'Median E value': get_median_values_serv_ln('E', AGB31),
        'B/A threshold': get_thresholds_values(1, bins_serv_hybrid5_ln, bins_serv_hybrid10_ln, bins_serv_hybrid15_ln),
        'C/B threshold': get_thresholds_values(2, bins_serv_hybrid5_ln, bins_serv_hybrid10_ln, bins_serv_hybrid15_ln),
        'D/C threshold': get_thresholds_values(3, bins_serv_hybrid5_ln, bins_serv_hybrid10_ln, bins_serv_hybrid15_ln),
        'E/D threshold': get_thresholds_values(4, bins_serv_hybrid5_ln, bins_serv_hybrid10_ln, bins_serv_hybrid15_ln)
    }

    Rating_change_ln = pd.DataFrame(dict_median_top_serv_ln)
    Rating_change_ln['B to A (mPts)'] = Rating_change_ln['Median B value'] - \
        Rating_change_ln['B/A threshold']
    Rating_change_ln['C to B (mPts)'] = Rating_change_ln['Median C value'] - \
        Rating_change_ln['C/B threshold']
    Rating_change_ln['D to C (mPts)'] = Rating_change_ln['Median D value'] - \
        Rating_change_ln['D/C threshold']
    Rating_change_ln['E to D (mPts)'] = Rating_change_ln['Median E value'] - \
        Rating_change_ln['E/D threshold']
    Rating_change_ln['B to A (%)'] = Rating_change_ln['B to A (mPts)'] / \
        Rating_change_ln['Median B value'] * 100
    Rating_change_ln['C to B (%)'] = Rating_change_ln['C to B (mPts)'] / \
        Rating_change_ln['Median C value'] * 100
    Rating_change_ln['D to C (%)'] = Rating_change_ln['D to C (mPts)'] / \
        Rating_change_ln['Median D value'] * 100
    Rating_change_ln['E to D (%)'] = Rating_change_ln['E to D (mPts)'] / \
        Rating_change_ln['Median E value'] * 100

    print(Rating_change_ln)

    # Preparing the data for plotting
    data_to_plot_ln = Rating_change_ln[[
        'Scenario', 'B to A (mPts)', 'C to B (mPts)', 'D to C (mPts)', 'E to D (mPts)', 'B to A (%)', 'C to B (%)', 'D to C (%)', 'E to D (%)']]
    data_to_plot_ln = pd.melt(data_to_plot_ln, id_vars='Scenario',
                              var_name='Rating_change', value_name='Change_required')

    # Create a bar plot for mPts using seaborn
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change', data=data_to_plot_ln[data_to_plot_ln['Rating_change'].str.contains(
        'mPts')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax1)

    # Create a bar plot for % using seaborn
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change', data=data_to_plot_ln[data_to_plot_ln['Rating_change'].str.contains(
        '%')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax2)

    # Add labels and title
    plt.xlabel('Scenario (log-transformed data)', fontsize=20)
    ax1.set_ylabel('Change required (mPts)', fontsize=18)
    ax2.set_ylabel('Change required (%)', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax1.set_xlabel('')

    # Show a single legend in the top left corner
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['B to A', 'C to B', 'D to C', 'E to D']
    ax1.legend(handles[:4], labels, title='Rating change',
               loc='upper left', title_fontsize=16, fontsize=16)
    ax2.get_legend().remove()
    for i in ax1.containers:
        ax1.bar_label(i, fmt='%.3f', fontsize=14)
    for i in ax2.containers:
        ax2.bar_label(i, fmt='%.1f', fontsize=14)

    # Show the plot
    plt.gcf().set_size_inches(16, 10)
    plt.show()

    # Table 2 - Calculating % of products with different ratings before and after removal of duplicates (n=924)


def duplicates_removal():
    merged_dup_df, sensitivity_dup_kg_norm, sensitivity_dup_serv_norm = sensitivity_duplicates(
        AGB31)
    print('% products with different ratings (all scenarios, per serving data):',
          sensitivity_dup_serv_norm)

    # Analysing the trends in rating changes per scenario (number of products and normalised results)
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_width'] != merged_dup_df['Score_NC_serv_width_dup']][[
        'Score_NC_serv_width', 'Score_NC_serv_width_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_quant'] != merged_dup_df['Score_NC_serv_quant_dup']][[
        'Score_NC_serv_quant', 'Score_NC_serv_quant_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_width_ln'] != merged_dup_df['Score_NC_serv_width_ln_dup']][[
        'Score_NC_serv_width_ln', 'Score_NC_serv_width_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_quant_ln'] != merged_dup_df['Score_NC_serv_quant_ln_dup']][[
        'Score_NC_serv_quant_ln', 'Score_NC_serv_quant_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_hybrid5'] != merged_dup_df['Score_NC_serv_hybrid5_dup']][[
        'Score_NC_serv_hybrid5', 'Score_NC_serv_hybrid5_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_hybrid10'] != merged_dup_df['Score_NC_serv_hybrid10_dup']][[
        'Score_NC_serv_hybrid10', 'Score_NC_serv_hybrid10_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_hybrid15'] != merged_dup_df['Score_NC_serv_hybrid15_dup']][[
        'Score_NC_serv_hybrid15', 'Score_NC_serv_hybrid15_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_hybrid5_ln'] != merged_dup_df['Score_NC_serv_hybrid5_ln_dup']][[
        'Score_NC_serv_hybrid5_ln', 'Score_NC_serv_hybrid5_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_hybrid10_ln'] != merged_dup_df['Score_NC_serv_hybrid10_ln_dup']][[
        'Score_NC_serv_hybrid10_ln', 'Score_NC_serv_hybrid10_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_serv_hybrid15_ln'] != merged_dup_df['Score_NC_serv_hybrid15_ln_dup']][[
        'Score_NC_serv_hybrid15_ln', 'Score_NC_serv_hybrid15_ln_dup']].value_counts())

    return merged_dup_df, sensitivity_dup_kg_norm, sensitivity_dup_serv_norm

    # Figure 8 - Calculating % of products with different ratings before and after random removal of datapoints (average of 500 runs)


def figure8():
    sensitivity_additions_serv, mean_90, mean_75, mean_50 = sensitivity_subset_serv(
        500, AGB31)
    print('Average % of products with different ratings - per scenario and subset size (per serving data)', sensitivity_additions_serv.groupby(
        ['Scenario', 'level_0'])['%_products'].mean())

    print('Average % of products with different ratings - per scenario (across all subsets, per serving data)',
          sensitivity_additions_serv.groupby(['Scenario'])['%_products'].mean())

    print('Number of runs where ratings under the width-based (S1) scenario do not get altered (per serving data)', sensitivity_additions_serv[(sensitivity_additions_serv['Scenario'] == 'S1') & (
        sensitivity_additions_serv['%_products'] == 0)].count())

    print(' Median % of products with different ratings - Width-based scenario, all runs (per serving data)',
          sensitivity_additions_serv[sensitivity_additions_serv['Scenario'] == 'S1']['%_products'].median())

    print('Average % of products with different ratings - Width-based scenario, all runs (per serving data)',
          sensitivity_additions_serv[sensitivity_additions_serv['Scenario'] == 'S1']['%_products'].mean())

    print('Average % of products with different ratings - Width-based scenario, only runs where ratings have been altered (per serving data)', sensitivity_additions_serv[(sensitivity_additions_serv['Scenario'] == 'S1') & (
        sensitivity_additions_serv['%_products'] != 0)]['%_products'].mean())

    print('Location of outlier values for hybrid scenario (S3) (per serving data)', sensitivity_additions_serv[(sensitivity_additions_serv['Scenario'] == 'S3_10') & (
        sensitivity_additions_serv['%_products'] > 5.3)]['level_0'].value_counts())

    print('Location of outlier values for quantiles-based scenario (S2) (per serving data)', sensitivity_additions_serv[(sensitivity_additions_serv['Scenario'] == 'S2') & (
        sensitivity_additions_serv['%_products'] > 4.8)]['level_0'].value_counts())

    # Plot with main three scenarios (linear, per serving data)
    main_scenarios = sensitivity_additions_serv[sensitivity_additions_serv['Scenario'].isin([
                                                                                            'S1', 'S2', 'S3_10'])]
    f, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.pointplot(data=main_scenarios, x='level_0', y='%_products',
                  hue='Scenario', capsize=0.1, palette='rocket_r', ax=axs[0])
    sns.boxplot(data=main_scenarios, y='%_products',
                x='Scenario', palette='rocket_r', ax=axs[1])
    axs[0].set_title('a.', fontsize=17, fontweight='semibold')
    axs[0].set_xlabel('Subset')
    axs[0].set_ylabel('Products awarded a different rating (%)')
    axs[1].set_title('b.', fontsize=17, fontweight='semibold')
    axs[1].set_ylabel('Products awarded a different rating (%)')
    axs[0].set_ylim([0, 3.5])
    axs[1].set_ylim([0, 18.5])
    handles, labels = axs[0].get_legend_handles_labels()
    labels = ['Width-based (S1)', 'Quantiles-based (S2)',
              'Hybrid 10% A/E (S3)']
    axs[0].legend(handles[:3], labels, title='Scenario',
                  loc='upper left', title_fontsize=16, fontsize=16)
    axs[1].set_xticklabels(labels)
    f.tight_layout()
    save(f, 'figure8_main')
    return sensitivity_additions_serv

    # Code for including log-transformed data


def figure8_withln():
    main_scenarios_ln = sensitivity_additions_serv[sensitivity_additions_serv['Scenario'].isin(
        ['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln'])]

    f, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.pointplot(data=main_scenarios_ln, x='level_0', y='%_products',
                  hue='Scenario', capsize=0.1, palette='rocket_r', ax=axs[0])
    sns.boxplot(data=main_scenarios_ln, y='%_products',
                x='Scenario', palette='rocket_r', ax=axs[1])
    axs[0].set_title('a.', fontsize=17, fontweight='semibold')
    axs[0].set_xlabel('Subsetting')
    axs[0].set_ylabel('Products awarded a different rating (%)')
    axs[1].set_title('b.', fontsize=17, fontweight='semibold')
    axs[1].set_ylabel('Products awarded a different rating (%)')
    axs[0].set_ylim([0, 3.5])
    axs[1].set_ylim([0, 18.5])
    f.tight_layout()

    # Code for hybrid scenarios


def figure8_cutoff():
    hybrid_scenarios = sensitivity_additions_serv[sensitivity_additions_serv['Scenario'].isin(
        ['S3_5', 'S3_10', 'S3_15', 'S3_5_ln', 'S3_10_ln', 'S3_15_ln'])]

    f, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.pointplot(data=hybrid_scenarios, x='level_0', y='%_products',
                  hue='Scenario', capsize=0.1, palette='rocket_r', ax=axs[0])
    sns.boxplot(data=hybrid_scenarios, y='%_products', x='Scenario', order=[
                'S3_5', 'S3_10', 'S3_15', 'S3_5_ln', 'S3_10_ln', 'S3_15_ln'], palette='rocket_r', ax=axs[1])
    axs[0].set_title('a.', fontsize=17, fontweight='semibold')
    axs[0].set_xlabel('Subsetting')
    axs[0].set_ylabel('Products awarded a different rating (%)')
    axs[1].set_title('b.', fontsize=17, fontweight='semibold')
    axs[1].set_ylabel('Products awarded a different rating (%)')
    axs[0].set_ylim([0, 3.5])
    axs[1].set_ylim([0, 18.5])
    f.tight_layout()


# Visualisations - Supplementary

    # Figure S1 - comparing how thresholds and ratings are influenced by variation of cut-off value in the hybrid scenario (per kg data)

def figure_s1():
    sns.set_context('talk')
    f, axs = plt.subplots(2, 3, figsize=(
        18, 13), sharey=True, layout='constrained')
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid5,
                 hue='Score_NC_kg_hybrid5', palette=colors, alpha=0.5, legend=False, ax=axs[0, 0])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid10,
                 hue='Score_NC_kg_hybrid10', palette=colors, alpha=0.5, legend=False, ax=axs[0, 1])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid15,
                 hue='Score_NC_kg_hybrid15', palette=colors, alpha=0.5, legend=False, ax=axs[0, 2])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid5_ln,
                 hue='Score_NC_kg_hybrid5_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 0])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid10_ln,
                 hue='Score_NC_kg_hybrid10_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 1])
    sns.histplot(data=AGB31, x='Single score per kg', bins=bins_kg_hybrid15_ln,
                 hue='Score_NC_kg_hybrid15_ln', palette=colors, alpha=0.5, legend=False, ax=axs[1, 2])
    axs[0, 0].set_title('a. Hybrid 5% A/E, linear data',
                        fontsize=17, fontweight='semibold')
    axs[0, 1].set_title('b. Hybrid 10% A/E, linear data (S3)',
                        fontsize=17, fontweight='semibold')
    axs[0, 2].set_title('c. Hybrid 15% A/E, linear data',
                        fontsize=17, fontweight='semibold')
    axs[1, 0].set_title('d. Hybrid 5% A/E, log-transformed data',
                        fontsize=17, fontweight='semibold')
    axs[1, 1].set_title('e. Hybrid 10% A/E, log-transformed data',
                        fontsize=17, fontweight='semibold')
    axs[1, 2].set_title('f. Hybrid 15% A/E, log-transformed data',
                        fontsize=17, fontweight='semibold')
    for ax in axs[:, :].flatten():
        ax.vlines(x=[AGB31['Single score per kg'].mean(), AGB31['Single score per kg'].median(
        )], ymin=0, ymax=2300, colors=['red', 'blue'], lw=1, label=['mean', 'median'])
    f.tight_layout()
    save(f, 'figureS1_main')

    # Generating pie charts
    cols_cutoff = ['Score_NC_kg_hybrid5', 'Score_NC_kg_hybrid10', 'Score_NC_kg_hybrid15',
                   'Score_NC_kg_hybrid5_ln', 'Score_NC_kg_hybrid10_ln', 'Score_NC_kg_hybrid15_ln']
    labels = ['A', 'B', 'C', 'D', 'E']
    for col in cols_cutoff:
        plt.clf()
        data = AGB31[col].value_counts().sort_index()
        plt.pie(data, labels=labels, colors=colors, wedgeprops=dict(alpha=0.5, linewidth=2,
                edgecolor='grey'), textprops={'size': 17}, autopct='%1.1f%%', pctdistance=1.3, labeldistance=.7)
        plt.savefig(os.path.join(chart_dir, 'figureS1_pie_{}.png'.format(col)))

    # Figure S2 - Calculating change in single score required to access next rating class up (per kg data)


def figure_s2():
    # Creating a dictionnary containing the median values and top thresholds of B, C, D and E rating classes for each variation of the hybrid scenario
    dict_median_top_kg = {
        'Scenario': ['Hybrid 5% A/E', 'Hybrid 10% A/E', 'Hybrid 15% A/E'],
        'Median B value': get_median_values_kg('B', AGB31),
        'Median C value': get_median_values_kg('C', AGB31),
        'Median D value': get_median_values_kg('D', AGB31),
        'Median E value': get_median_values_kg('E', AGB31),
        'B/A threshold': get_thresholds_values(1, bins_kg_hybrid5, bins_kg_hybrid10, bins_kg_hybrid15),
        'C/B threshold': get_thresholds_values(2, bins_kg_hybrid5, bins_kg_hybrid10, bins_kg_hybrid15),
        'D/C threshold': get_thresholds_values(3, bins_kg_hybrid5, bins_kg_hybrid10, bins_kg_hybrid15),
        'E/D threshold': get_thresholds_values(4, bins_kg_hybrid5, bins_kg_hybrid10, bins_kg_hybrid15)
    }

    # Calculating the difference between the two values to obtain the change in single score required to access the next performance class (as absolute and relative values)
    Rating_change_kg = pd.DataFrame(dict_median_top_kg)
    Rating_change_kg['B to A (mPts)'] = Rating_change_kg['Median B value'] - \
        Rating_change_kg['B/A threshold']
    Rating_change_kg['C to B (mPts)'] = Rating_change_kg['Median C value'] - \
        Rating_change_kg['C/B threshold']
    Rating_change_kg['D to C (mPts)'] = Rating_change_kg['Median D value'] - \
        Rating_change_kg['D/C threshold']
    Rating_change_kg['E to D (mPts)'] = Rating_change_kg['Median E value'] - \
        Rating_change_kg['E/D threshold']
    Rating_change_kg['B to A (%)'] = Rating_change_kg['B to A (mPts)'] / \
        Rating_change_kg['Median B value'] * 100
    Rating_change_kg['C to B (%)'] = Rating_change_kg['C to B (mPts)'] / \
        Rating_change_kg['Median C value'] * 100
    Rating_change_kg['D to C (%)'] = Rating_change_kg['D to C (mPts)'] / \
        Rating_change_kg['Median D value'] * 100
    Rating_change_kg['E to D (%)'] = Rating_change_kg['E to D (mPts)'] / \
        Rating_change_kg['Median E value'] * 100

    print(Rating_change_kg)

    # Preparing data for plotting - slicing and melting
    data_to_plot = Rating_change_kg[['Scenario', 'B to A (mPts)', 'C to B (mPts)', 'D to C (mPts)',
                                     'E to D (mPts)', 'B to A (%)', 'C to B (%)', 'D to C (%)', 'E to D (%)']]
    data_to_plot = pd.melt(data_to_plot, id_vars='Scenario',
                           var_name='Rating_change_kg', value_name='Change_required')

    # Create a bar plot for mPts using seaborn
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change_kg', data=data_to_plot[data_to_plot['Rating_change_kg'].str.contains(
        'mPts')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax1)

    # Create a bar plot for % using seaborn
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change_kg', data=data_to_plot[data_to_plot['Rating_change_kg'].str.contains(
        '%')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax2)

    # Add labels and title
    plt.xlabel('Scenario', fontsize=20)
    ax1.set_ylabel('Change required (mPts)', fontsize=18)
    ax2.set_ylabel('Change required (%)', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax1.set_xlabel('')

    # Show a single legend in the top left corner
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['B to A', 'C to B', 'D to C', 'E to D']
    ax1.legend(handles[:4], labels, title='Rating change',
               loc='upper left', title_fontsize=16, fontsize=16)
    ax2.get_legend().remove()
    for i in ax1.containers:
        ax1.bar_label(i, fmt='%.3f', fontsize=14)
    for i in ax2.containers:
        ax2.bar_label(i, fmt='%.1f', fontsize=14)

    # Show the plot
    plt.gcf().set_size_inches(16, 12)
    save(f, 'figureS2_main')

    # For ln scenarios


def figure_s2_ln():
    dict_median_top_kg_ln = {
        'Scenario': ['Hybrid 5% A/E', 'Hybrid 10% A/E', 'Hybrid 15% A/E'],
        'Median B value': get_median_values_kg_ln('B', AGB31),
        'Median C value': get_median_values_kg_ln('C', AGB31),
        'Median D value': get_median_values_kg_ln('D', AGB31),
        'Median E value': get_median_values_kg_ln('E', AGB31),
        'B/A threshold': get_thresholds_values(1, bins_kg_hybrid5_ln, bins_kg_hybrid10_ln, bins_kg_hybrid15_ln),
        'C/B threshold': get_thresholds_values(2, bins_kg_hybrid5_ln, bins_kg_hybrid10_ln, bins_kg_hybrid15_ln),
        'D/C threshold': get_thresholds_values(3, bins_kg_hybrid5_ln, bins_kg_hybrid10_ln, bins_kg_hybrid15_ln),
        'E/D threshold': get_thresholds_values(4, bins_kg_hybrid5_ln, bins_kg_hybrid10_ln, bins_kg_hybrid15_ln)
    }

    Rating_change_kg_ln = pd.DataFrame(dict_median_top_kg_ln)
    Rating_change_kg_ln['B to A (mPts)'] = Rating_change_kg_ln['Median B value'] - \
        Rating_change_kg_ln['B/A threshold']
    Rating_change_kg_ln['C to B (mPts)'] = Rating_change_kg_ln['Median C value'] - \
        Rating_change_kg_ln['C/B threshold']
    Rating_change_kg_ln['D to C (mPts)'] = Rating_change_kg_ln['Median D value'] - \
        Rating_change_kg_ln['D/C threshold']
    Rating_change_kg_ln['E to D (mPts)'] = Rating_change_kg_ln['Median E value'] - \
        Rating_change_kg_ln['E/D threshold']
    Rating_change_kg_ln['B to A (%)'] = Rating_change_kg_ln['B to A (mPts)'] / \
        Rating_change_kg_ln['Median B value'] * 100
    Rating_change_kg_ln['C to B (%)'] = Rating_change_kg_ln['C to B (mPts)'] / \
        Rating_change_kg_ln['Median C value'] * 100
    Rating_change_kg_ln['D to C (%)'] = Rating_change_kg_ln['D to C (mPts)'] / \
        Rating_change_kg_ln['Median D value'] * 100
    Rating_change_kg_ln['E to D (%)'] = Rating_change_kg_ln['E to D (mPts)'] / \
        Rating_change_kg_ln['Median E value'] * 100

    print(Rating_change_kg_ln)

    # Preparing data for plotting - slicing and melting
    data_to_plot = Rating_change_kg_ln[[
        'Scenario', 'B to A (mPts)', 'C to B (mPts)', 'D to C (mPts)', 'E to D (mPts)', 'B to A (%)', 'C to B (%)', 'D to C (%)', 'E to D (%)']]
    data_to_plot = pd.melt(data_to_plot, id_vars='Scenario',
                           var_name='Rating_change_kg_ln', value_name='Change_required')

    # Create a bar plot for mPts using seaborn
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change_kg_ln', data=data_to_plot[data_to_plot['Rating_change_kg_ln'].str.contains(
        'mPts')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax1)

    # Create a bar plot for % using seaborn
    sns.barplot(x='Scenario', y='Change_required', hue='Rating_change_kg_ln',
                data=data_to_plot[data_to_plot['Rating_change_kg_ln'].str.contains('%')], palette=colors[1:5], alpha=0.5, edgecolor='black', ax=ax2)

    # Add labels and title
    plt.xlabel('Scenario', fontsize=20)
    ax1.set_ylabel('Change required (mPts)', fontsize=18)
    ax2.set_ylabel('Change required (%)', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax1.set_xlabel('')

    # Show a single legend in the top left corner
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['B to A', 'C to B', 'D to C', 'E to D']
    ax1.legend(handles[:4], labels, title='Rating change',
               loc='upper left', title_fontsize=16, fontsize=16)
    ax2.get_legend().remove()
    for i in ax1.containers:
        ax1.bar_label(i, fmt='%.3f', fontsize=14)
    for i in ax2.containers:
        ax2.bar_label(i, fmt='%.1f', fontsize=14)

    # Show the plot
    plt.gcf().set_size_inches(16, 12)
    plt.show()

    # Table S3 - Calculating % of products with different ratings before and after removal of duplicates (n=924) (per kg data)


def duplicates_removal_kg():
    print('% products with different ratings (all scenarios, per kg data):',
          sensitivity_dup_kg_norm)
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_width'] != merged_dup_df['Score_NC_kg_width_dup']][[
          'Score_NC_kg_width', 'Score_NC_kg_width_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_quant'] != merged_dup_df['Score_NC_kg_quant_dup']][[
          'Score_NC_kg_quant', 'Score_NC_kg_quant_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_width_ln'] != merged_dup_df['Score_NC_kg_width_ln_dup']][[
          'Score_NC_kg_width_ln', 'Score_NC_kg_width_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_quant_ln'] != merged_dup_df['Score_NC_kg_quant_ln_dup']][[
          'Score_NC_kg_quant_ln', 'Score_NC_kg_quant_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_hybrid5'] != merged_dup_df['Score_NC_kg_hybrid5_dup']][[
          'Score_NC_kg_hybrid5', 'Score_NC_kg_hybrid5_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_hybrid10'] != merged_dup_df['Score_NC_kg_hybrid10_dup']][[
          'Score_NC_kg_hybrid10', 'Score_NC_kg_hybrid10_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_hybrid15'] != merged_dup_df['Score_NC_kg_hybrid15_dup']][[
          'Score_NC_kg_hybrid15', 'Score_NC_kg_hybrid15_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_hybrid5_ln'] != merged_dup_df['Score_NC_kg_hybrid5_ln_dup']][[
          'Score_NC_kg_hybrid5_ln', 'Score_NC_kg_hybrid5_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_hybrid10_ln'] != merged_dup_df['Score_NC_kg_hybrid10_ln_dup']][[
          'Score_NC_kg_hybrid10_ln', 'Score_NC_kg_hybrid10_ln_dup']].value_counts())
    print(merged_dup_df.loc[merged_dup_df['Score_NC_kg_hybrid15_ln'] != merged_dup_df['Score_NC_kg_hybrid15_ln_dup']][[
          'Score_NC_kg_hybrid15_ln', 'Score_NC_kg_hybrid15_ln_dup']].value_counts())

    # Figure S3 - Calculating % of products with different ratings before and after random removal of datapoints (average of 500 runs) (per kg data)


def figure_s3():
    sensitivity_additions_kg, mean_90_kg, mean_75_kg, mean_50_kg = sensitivity_subset_kg(
        500, AGB31)

    main_scenarios_kg = sensitivity_additions_kg[sensitivity_additions_kg['Scenario'].isin([
                                                                                           'S1', 'S2', 'S3_10'])]
    print('Average % of products with different ratings - per scenario and subset size (per kg data)):', sensitivity_additions_kg.groupby(
        ['Scenario', 'level_0'])['%_products'].mean())
    print('Average % of products with different ratings - per scenario (across all subsets, per kg data):',
          sensitivity_additions_kg.groupby(['Scenario'])['%_products'].mean())

    print('Number of runs where ratings under the width-based (S1) scenario do not get altered (per kg data):', sensitivity_additions_kg[(sensitivity_additions_kg['Scenario'] == 'S1') & (
        sensitivity_additions_kg['%_products'] == 0)].count())

    print(' Median % of products with different ratings - Width-based scenario, all runs (per kg data):', sensitivity_additions_kg[sensitivity_additions_kg['Scenario']
          == 'S1']['%_products'].median())

    print('Average % of products with different ratings - Width-based scenario, all runs (per kg data):',
          sensitivity_additions_kg[sensitivity_additions_kg['Scenario'] == 'S1']['%_products'].mean())

    print('Average % of products with different ratings - Width-based scenario, only runs where ratings have been altered (per kg data):', sensitivity_additions_kg[(sensitivity_additions_kg['Scenario'] == 'S1') & (
        sensitivity_additions_kg['%_products'] != 0)]['%_products'].mean())

    print('Location of outlier values for hybrid scenario (S3) (per kg data):', sensitivity_additions_kg[(sensitivity_additions_kg['Scenario'] == 'S3_10') & (
        sensitivity_additions_kg['%_products'] > 5.3)]['level_0'].value_counts())

    print('Location of outlier values for quantiles-based scenario (S2) (per kg data):', sensitivity_additions_kg[(sensitivity_additions_kg['Scenario'] == 'S2') & (
        sensitivity_additions_kg['%_products'] > 4.8)]['level_0'].value_counts())

    f, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.pointplot(data=main_scenarios_kg, x='level_0', y='%_products',
                  hue='Scenario', capsize=0.1, palette='rocket_r', ax=axs[0])
    sns.boxplot(data=main_scenarios_kg, y='%_products',
                x='Scenario', palette='rocket_r', ax=axs[1])
    axs[0].set_title('a.', fontsize=17, fontweight='semibold')
    axs[0].set_xlabel('Subset')
    axs[0].set_ylabel('Products awarded a different rating (%)')
    axs[1].set_title('b.', fontsize=17, fontweight='semibold')
    axs[1].set_ylabel('Products awarded a different rating (%)')
    axs[0].set_ylim([0, 5])
    axs[1].set_ylim([0, 20])
    handles, labels = axs[0].get_legend_handles_labels()
    labels = ['Width-based (S1)', 'Quantiles-based (S2)',
              'Hybrid 10% A/E (S3)']
    axs[0].legend(handles[:3], labels, title='Scenario',
                  loc='upper left', title_fontsize=16, fontsize=16)
    axs[1].set_xticklabels(labels)
    f.tight_layout()
    save(f, 'figureS3_main')
    return sensitivity_additions_kg

    # Code for including log-transformed data


def figure_s3_withln():
    main_scenarios_ln_kg = sensitivity_additions_kg[sensitivity_additions_kg['Scenario'].isin(
        ['S1', 'S2', 'S3_10', 'S1_ln', 'S2_ln', 'S3_10_ln'])]

    f, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.pointplot(data=main_scenarios_ln_kg, x='level_0', y='%_products',
                  hue='Scenario', capsize=0.1, palette='rocket_r', ax=axs[0])
    sns.boxplot(data=main_scenarios_ln_kg, y='%_products',
                x='Scenario', palette='rocket_r', ax=axs[1])
    axs[0].set_title('a.', fontsize=17, fontweight='semibold')
    axs[0].set_xlabel('Subsetting')
    axs[0].set_ylabel('Products awarded a different rating (%)')
    axs[1].set_title('b.', fontsize=17, fontweight='semibold')
    axs[1].set_ylabel('Products awarded a different rating (%)')
    f.tight_layout()

    # Code for hybrid scenarios


def figure_s3_cutoff():
    hybrid_scenarios = sensitivity_additions_kg[sensitivity_additions_kg['Scenario'].isin(
        ['S3_5', 'S3_10', 'S3_15', 'S3_5_ln', 'S3_10_ln', 'S3_15_ln'])]

    f, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.pointplot(data=hybrid_scenarios, x='level_0', y='%_products',
                  hue='Scenario', capsize=0.1, palette='rocket_r', ax=axs[0])
    sns.boxplot(data=hybrid_scenarios, y='%_products', x='Scenario', order=[
                'S3_5', 'S3_10', 'S3_15', 'S3_5_ln', 'S3_10_ln', 'S3_15_ln'], palette='rocket_r', ax=axs[1])
    axs[0].set_title('a.', fontsize=17, fontweight='semibold')
    axs[0].set_xlabel('Subsetting')
    axs[0].set_ylabel('Products awarded a different rating (%)')
    axs[1].set_title('b.', fontsize=17, fontweight='semibold')
    axs[1].set_ylabel('Products awarded a different rating (%)')
    f.tight_layout()

    # Figure S4 - Visualisation of clusters of identical single score values in the full dataset (n=2253)


def figure_s4():
    values = AGB31['Single score per serving'].sort_values().unique()
    count_dup = AGB31.groupby('Single score per serving')[
        'Single score per serving'].count()
    condition = count_dup > 5

    sns.relplot(x=values, y=count_dup, kind='scatter', height=10, aspect=1.5,
                hue=condition, style=condition, palette='colorblind', s=100, legend=False)
    plt.ylabel('Count of occurences', fontsize=18)
    plt.xlabel('Single score per serving value', fontsize=18)
    save(plt, 'figureS4_main')

    # Figure S5 - Calculation of results for alternative number of runs (per serving data)


def figure_s5_data():
    sensitivity_additions_serv_1, mean_90_1, mean_75_1, mean_50_1 = sensitivity_subset_serv(
        1, AGB31)
    sensitivity_additions_serv_10, mean_90_10, mean_75_10, mean_50_10 = sensitivity_subset_serv(
        10, AGB31)
    sensitivity_additions_serv_50, mean_90_50, mean_75_50, mean_50_50 = sensitivity_subset_serv(
        50, AGB31)
    sensitivity_additions_serv_100, mean_90_100, mean_75_100, mean_50_100 = sensitivity_subset_serv(
        100, AGB31)
    sensitivity_additions_serv_250, mean_90_250, mean_75_250, mean_50_250 = sensitivity_subset_serv(
        250, AGB31)
    sensitivity_additions_serv_1000, mean_90_1000, mean_75_1000, mean_50_1000 = sensitivity_subset_serv(
        1000, AGB31)
    
    # write to Excel file
    with pd.ExcelWriter(os.path.join(chart_dir, "V4_FigureS5_data.xlsx")) as writer:
        sensitivity_additions_serv_1.to_excel(writer, sheet_name='1 run')
        sensitivity_additions_serv_10.to_excel(writer, sheet_name='10 runs')
        sensitivity_additions_serv_50.to_excel(writer, sheet_name='50 runs')
        sensitivity_additions_serv_100.to_excel(writer, sheet_name='100 runs')
        sensitivity_additions_serv_250.to_excel(writer, sheet_name='250 runs')
        sensitivity_additions_serv.to_excel(writer, sheet_name='500 runs')
        sensitivity_additions_serv_1000.to_excel(
            writer, sheet_name='1000 runs')
        
    return sensitivity_additions_serv_1, sensitivity_additions_serv_10, sensitivity_additions_serv_50, sensitivity_additions_serv_100, sensitivity_additions_serv_250, sensitivity_additions_serv_1000

    
<<<<<<< HEAD
#if name == "__main__": # FIXME
=======
if __name__ == "__main__": # FIXME

    sup = input("Also generate supplementary figures? (y/n)")
    
>>>>>>> 6ca17a04578c409ae28a16ee0164d5a039ada4b5
    # Run elements from main text
    AGB31, bins_kg_width, bins_kg_quant, bins_kg_hybrid10, bins_kg_width_ln, bins_kg_quant_ln, bins_kg_hybrid10_ln, bins_serv_width, bins_serv_quant, bins_serv_hybrid10, bins_serv_width_ln, bins_serv_quant_ln, bins_serv_hybrid10_ln, bins_kg_hybrid5, bins_serv_hybrid5, bins_kg_hybrid5_ln, bins_serv_hybrid5_ln, bins_kg_hybrid15, bins_serv_hybrid15, bins_kg_hybrid15_ln, bins_serv_hybrid15_ln = import_calcs()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    merged_dup_df, sensitivity_dup_kg_norm, sensitivity_dup_serv_norm = duplicates_removal()
    sensitivity_additions_serv = figure8()
    
<<<<<<< HEAD
    # Supplementary to be run separately? Otherwise running time is very long (because of figure s5)
    figure_s1()
    figure_s2()
    sensitivity_additions_kg = figure_s3()
    figure_s4()
    sensitivity_additions_serv_1, sensitivity_additions_serv_10, sensitivity_additions_serv_50, sensitivity_additions_serv_100, sensitivity_additions_serv_250, sensitivity_additions_serv_1000 = figure_s5_data()
=======
    if sup.lower() in ['y', 'yes']:
        # Supplementary to be run separately? Otherwise running time is very long (because of figure s5)
        figure_s1()
        figure_s2()
        sensitivity_additions_kg = figure_s3()
        figure_s4()
        sensitivity_additions_serv_1, sensitivity_additions_serv_10, sensitivity_additions_serv_50, sensitivity_additions_serv_100, sensitivity_additions_serv_250, sensitivity_additions_serv_1000 = figure_s5_data()
>>>>>>> 6ca17a04578c409ae28a16ee0164d5a039ada4b5
