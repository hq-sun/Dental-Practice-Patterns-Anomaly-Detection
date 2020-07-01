#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:55:39 2020

@author: Heqing Sun
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get current working directory
os.getcwd()

# Read csv file
df = pd.read_csv("./data/Dental_Anomaly_Detection_Case_Study.csv")
df.head()
df_orig = df.copy()

# Define some functions for EDA
def print_dataframe_description(df, col):
    print('Column Name:', col)
    print('Number of Rows:', len(df.index))
    print('Number of Missing Values:', df[col].isnull().sum())
    print('Percent Missing:', df[col].isnull().sum()/len(df.index)*100, '%')
    print('Number of Unique Values:', len(df[col].unique()))
    print('\n')

# For continuous variables    
def print_descriptive_stats(df, col):
    print('Column Name:', col)
    print('Mean:', np.mean(df[col]))
    print('Median:', np.nanmedian(df[col]))
    print('Standard Deviation:', np.std(df[col]))
    print('Minimum:', np.min(df[col]))
    print('Maximum:', np.max(df[col]))
    print('\n')

# Plotting countplots   
def plot_counts(df, col):
    sns.set(style='darkgrid')
    ax = sns.countplot(x=col, data=df)
    plt.xticks(rotation=90)
    plt.title('Count Plot')
    plt.show()
    
# Plotting distribution plots 
def plot_distribution(df, col):
    sns.set(style='darkgrid')
    ax = sns.distplot(df[col].dropna())
    plt.xticks(rotation=90)
    plt.title('Distribution Plot')
    plt.show()
    
#############################################################################
########################### EDA and Data Cleaning ###########################
#############################################################################
for col in df.columns:
    print_dataframe_description(df, col)
# Column Name: Per_Visit_Payment
# Number of Rows: 36888
# Number of Missing Values: 613
# Percent Missing: 1.6617870310127953 %
# Number of Unique Values: 33914
## Per_Visit_Payment is the only variable has missing values in the original dataset

for col in df.columns:
    print_descriptive_stats(df, col)
    
# Take care of the missing value in Per_Visit_Payment variable
# Add indicator variable for missing value
df['Per_Visit_Payment_missing'] = df['Per_Visit_Payment'].isnull()
df['Per_Visit_Payment_missing'] = df['Per_Visit_Payment_missing'].astype(int)

# Impute missing value with median
df['Per_Visit_Payment'].fillna(df['Per_Visit_Payment'].median(), inplace=True)
df['Per_Visit_Payment'].isnull().sum()

#############################################################################
########################### Feature Engineering #############################
#############################################################################
# Create Crowns to Filling Ratio varibale
df['Crowns_to_Filling_Ratio'] = df['Crown_Count']/df['Filling_Count']
df['Crowns_to_Filling_Ratio'].fillna(0, inplace=True) # fill in missing value with 0 because all NANs come from the 'Crown_Count' and 'Filling_Count' are both 0
df['Crowns_to_Filling_Ratio'].isnull().sum()

# Create Root Canals to Crown Ratio varibale
df['Root_Canals_to_Crown_Ratio'] = df['Root_Canal_Count']/df['Crown_Count'] #nan and inf both
df['Root_Canals_to_Crown_Ratio'].fillna(0, inplace=True) # fill in NAN with 0, leave inf as it is
df['Root_Canals_to_Crown_Ratio'].isnull().sum()

# Create Extract to Crown Ratio varibale
df['Extract_to_Crown_Ratio'] = df['Extract_Count']/df['Crown_Count'] #nan and inf both
df['Extract_to_Crown_Ratio'].fillna(0, inplace=True) # fill in NAN with 0, leave inf as it is
df['Extract_to_Crown_Ratio'].isnull().sum()

# Replace infinity with an extreme number 9999
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(9999, inplace=True) 

# Create top 1% indicator variable for these four variables
df['Crowns_to_Filling_Ratio_top1_pct'] = 0
df.loc[df['Crowns_to_Filling_Ratio'] >= np.percentile(df['Crowns_to_Filling_Ratio'], 99), 'Crowns_to_Filling_Ratio_top1_pct'] = 1

df['Root_Canals_to_Crown_Ratio_top1_pct'] = 0
df.loc[df['Root_Canals_to_Crown_Ratio'] >= np.percentile(df['Root_Canals_to_Crown_Ratio'], 99), 'Root_Canals_to_Crown_Ratio_top1_pct'] = 1

df['Extract_to_Crown_Ratio_top1_pct'] = 0
df.loc[df['Extract_to_Crown_Ratio'] >= np.percentile(df['Extract_to_Crown_Ratio'], 99), 'Extract_to_Crown_Ratio_top1_pct'] = 1

df['Per_Visit_Payment_top1_pct'] = 0
df.loc[df['Per_Visit_Payment'] >= np.percentile(df['Per_Visit_Payment'], 99), 'Per_Visit_Payment_top1_pct'] = 1

# Indicator variable for percentile
df['Crowns_to_Filling_Ratio_pct'] = df['Crowns_to_Filling_Ratio'].rank(pct=True)
df['Root_Canals_to_Crown_Ratio_pct'] = df['Root_Canals_to_Crown_Ratio'].rank(pct=True)
df['Extract_to_Crown_Ratio_pct'] = df['Extract_to_Crown_Ratio'].rank(pct=True)
df['Per_Visit_Payment_pct'] = df['Per_Visit_Payment'].rank(pct=True)

# Write cleaned dataset to csv and pickle file
df.to_csv('./data/clean/df_newfeatures.csv')
df.to_pickle('./data/clean/df_newfeatures.pkl')

#######################################################################
########################### Visualization #############################
#######################################################################
sns.distplot(df['Crowns_to_Filling_Ratio'], color='blue')
plt.title('Crowns to Filling Ratio Distribution Plot')
plt.xlabel('Crowns to Filling Ratio')
plt.show() ## not good

sns.distplot(df[df['Root_Canals_to_Crown_Ratio']<3]['Root_Canals_to_Crown_Ratio'], color='blue')
plt.title('Root Canals to Crown Ratio Distribution Plot')
plt.xlabel('Root Canals to Crown Ratio')
plt.show() ## good

sns.distplot(df[df['Extract_to_Crown_Ratio']<600]['Extract_to_Crown_Ratio'], color='blue')
plt.title('Extract to Crown Ratio Distribution Plot')
plt.xlabel('Extract to Crown Ratio')
plt.show() ## not good

sns.distplot(df['Per_Visit_Payment'], color='blue')
plt.title('Per Visit Payment Distribution Plot')
plt.xlabel('Per Visit Payment')
plt.show() ## good

#############################################################################################
########################### Identify Potential Outliers - k-means ###########################
#############################################################################################
df = pd.read_pickle('./data/clean/df_newfeatures.pkl')

from sklearn.cluster import KMeans
df_kmeans = df.copy()
df_kmeans.set_index('Dentist_Id', inplace=True)

# Find the approporiate cluster number - Elbow Plot
wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_kmeans)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 30), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
## Looks like k=3 is the elbow

# Fitting k-means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_kmeans)
# Beginning of the cluster numbering with 1 instead of 0
y_kmeans = y_kmeans + 1

# Adding cluster column to the dataset
df_kmeans['Cluster'] = y_kmeans
df_kmeans['Cluster'].value_counts()
# Cluster# Counts
# 1    34773
# 2     2063
# 3       52

# Save potential outliers to a list
potentialList_kmeans = df_kmeans[df_kmeans['Cluster'] == 3].index.tolist()

# Mean of clusters - Description Matrix
kmeans_mean_cluster = pd.DataFrame(round(df_kmeans.groupby('Cluster').mean(), 1))
kmeans_mean_cluster

# Try to visualize k-means result
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df_pca = df.copy()
df_pca.set_index('Dentist_Id', inplace=True)

# Create a PCA instance: pca
pca = PCA(n_components=15)
principalComponents = pca.fit_transform(df_pca)

# Create a dataframe contains Principal Components and Cluster labels
pC = pd.DataFrame(principalComponents)
pC['Cluster'] = y_kmeans
pC['Cluster'].value_counts(dropna=False)

# Visualize k-means plot in 2D
plt.figure(figsize=(10,8))
plt.scatter(pC[0], pC[1], c=y_kmeans, cmap='prism')
plt.legend(handles=scatter.legend_elements()[0], labels=y_kmeans)
plt.show()

# Visualize k-means plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
handles = []
Cluster1 = ax.scatter3D(pC[pC['Cluster'] == 1][0], pC[pC['Cluster'] == 1][1], pC[pC['Cluster'] == 1][2], c='tan', label='Cluster 1')
Cluster2 = ax.scatter3D(pC[pC['Cluster'] == 2][0], pC[pC['Cluster'] == 2][1], pC[pC['Cluster'] == 2][2], c='hotpink', label='Cluster 2')
Cluster3 = ax.scatter3D(pC[pC['Cluster'] == 3][0], pC[pC['Cluster'] == 3][1], pC[pC['Cluster'] == 3][2], c='dodgerblue', label='Cluster 3')
ax.set_xlabel('PC 1', fontsize=8)
ax.set_ylabel('PC 2', fontsize=8)
ax.set_zlabel('PC 3', fontsize=8)
plt.legend(numpoints=1, ncol=3, bbox_to_anchor=(1,1))
plt.show()

###########################################################################
########################### Statistical Testing ###########################
###########################################################################
from scipy import stats
df_t = df.copy()

# Subset dataset by the top1% indicator variable - Crowns_to_Filling_Ratio_top1_pct
C_to_F_top1 = df_t[(df_t['Crowns_to_Filling_Ratio_top1_pct'] == 1)]
C_to_F_top1.set_index('Dentist_Id', inplace=True)

C_to_F_non_top1 = df_t[(df_t['Crowns_to_Filling_Ratio_top1_pct'] == 0)]
C_to_F_non_top1.set_index('Dentist_Id', inplace=True)

# Check assumption for Two Sample t-test
# Homogeneity of variances
stats.levene(C_to_F_top1['Crowns_to_Filling_Ratio'], C_to_F_non_top1['Crowns_to_Filling_Ratio'])
# LeveneResult(statistic=2075.3042703329666, pvalue=0.0)
## The test is significant meaning，there is no homogeneity of variance, so need to conduct a Welch’s t-test

# Test the assumption of normality - Shapiro-Wilk test
stats.shapiro(C_to_F_top1['Crowns_to_Filling_Ratio'])
# (0.23350894451141357, 3.05601590410039e-36)
stats.shapiro(C_to_F_non_top1['Crowns_to_Filling_Ratio'])
# (0.7396008968353271, 0.0)
## Both of the variables of interest violates the assumption of normality so we cannot use Welch’s t-test
## Here, decided to use Mann-Whitney Rank Test because it does not require normal distribution

########################################################################################
########################### Bootstrap Mann-Whitney Rank Test ###########################
########################################################################################
import random

def real_u_test(small, big, size):
    sm = random.sample(small, k=size)
    bg = random.sample(big, k=size)
    u_stat = stats.mannwhitneyu(sm, bg)[0]
    return u_stat 

def bootstrap_u_test(small, big, size, nboot):
    all_stats = []
    for i in range(nboot):
        # Random select same size samples from two original lists
        sm = random.sample(small, k=size)
        bg = random.sample(big, k=size)
        # Combined two same size samples and shuffle it, select again, assume one is the small group and the other is big group
        cmbd = sm + bg
        m = random.sample(cmbd, k=size)
        n = list(set(cmbd)-set(m))
        u_stat = stats.mannwhitneyu(m, n)[0]
        all_stats.append(u_stat)
    return all_stats    

def plot_u_stats(all_stats, u_stat, colname):
    lower_bound = np.percentile(all_stats, 2.5)
    upper_bound = np.percentile(all_stats, 97.5)
    sns.distplot(all_stats, bins=100, kde=False, color='blue')
    plt.axvline(u_stat, color='red', label='True Statistics')
    plt.axvline(lower_bound, color='gray', linestyle='dashed', label='Confidence Interval')
    plt.axvline(upper_bound, color='gray', linestyle='dashed')
    plt.title('Bootstrap M-W Test for ' + colname)
    plt.xlabel('Mann-Whitney Statistics')
    plt.ylabel('Counts')
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1))
    plt.show()

# Get ratio columns
a = C_to_F_top1['Crowns_to_Filling_Ratio'].tolist()
b = C_to_F_non_top1['Crowns_to_Filling_Ratio'].tolist()

all_stats = bootstrap_u_test(a, b, 300, 10000)
u_stat = real_u_test(a, b, 300)
plot_u_stats(all_stats, u_stat, "Crowns to Filling Ratio")
## Crowns_to_Filling_Ratio is statistically significant


# Subset dataset by the top1% indicator variable - Root_Canals_to_Crown_Ratio_top1_pct
R_to_C_top1 = df_t[(df_t['Root_Canals_to_Crown_Ratio_top1_pct'] == 1)]
R_to_C_top1.set_index('Dentist_Id', inplace=True)
R_to_C_non_top1 = df_t[(df_t['Root_Canals_to_Crown_Ratio_top1_pct'] == 0)]
R_to_C_non_top1.set_index('Dentist_Id', inplace=True)

a = R_to_C_top1['Root_Canals_to_Crown_Ratio'].tolist()
b = R_to_C_non_top1['Root_Canals_to_Crown_Ratio'].tolist()

all_stats = bootstrap_u_test(a, b, 675, 10000)
u_stat = real_u_test(a, b, 675)
plot_u_stats(all_stats, u_stat, "Root Canals to Crown Ratio")
## Root_Canals_to_Crown_Ratio is statistically significant


# Subset dataset by the top1% indicator variable - Extract_to_Crown_Ratio_top1_pct
E_to_C_top1 = df_t[(df_t['Extract_to_Crown_Ratio_top1_pct'] == 1)]
E_to_C_top1.set_index('Dentist_Id', inplace=True)
E_to_C_non_top1 = df_t[(df_t['Extract_to_Crown_Ratio_top1_pct'] == 0)]
E_to_C_non_top1.set_index('Dentist_Id', inplace=True)

a = E_to_C_top1['Extract_to_Crown_Ratio'].tolist()
b = E_to_C_non_top1['Extract_to_Crown_Ratio'].tolist()

all_stats = bootstrap_u_test(a, b, 1650, 10000)
u_stat = real_u_test(a, b, 1650)
plot_u_stats(all_stats, u_stat, "Extract to Crown Ratio")
## Extract_to_Crown_Ratio is statistically significant


# Subset dataset by the top1% indicator variable - Per_Visit_Payment_top1_pct
Per_Visit_Payment_top1 = df_t[(df_t['Per_Visit_Payment_top1_pct'] == 1)]
Per_Visit_Payment_top1.set_index('Dentist_Id', inplace=True)
Per_Visit_Payment_non_top1 = df_t[(df_t['Per_Visit_Payment_top1_pct'] == 0)]
Per_Visit_Payment_non_top1.set_index('Dentist_Id', inplace=True)

a = Per_Visit_Payment_top1['Per_Visit_Payment'].tolist()
b = Per_Visit_Payment_non_top1['Per_Visit_Payment'].tolist()

all_stats = bootstrap_u_test(a, b, 300, 10000)
u_stat = real_u_test(a, b, 300)
plot_u_stats(all_stats, u_stat, "Per Visit Payment")
## Per_Visit_Payment is statistically significant


# Create a new column to sum all these four top1% flag variables
df_t['Four_top1_pct_sum'] = df_t['Crowns_to_Filling_Ratio_top1_pct'] + df_t['Root_Canals_to_Crown_Ratio_top1_pct'] + df_t['Extract_to_Crown_Ratio_top1_pct'] + df_t['Per_Visit_Payment_top1_pct']
df_t['Four_top1_pct_sum'].value_counts()
# Freq. Counts
# 0    33315
# 1     3500
# 2       73
## These 73 dentists might needed further investigation

# Save potential outliers to a list
potentialList_t = df_t[df_t['Four_top1_pct_sum'] == 2].index.tolist()

# Compare two candidate lists: potentialList_kmeans, potentialList_t to see if there are any overlap
potentialList_combined = potentialList_kmeans + potentialList_t
potentialList_combined_unique = list(set(potentialList_combined))
## There are no overlap between two candidate lists: potentialList_kmeans, potentialList_t

# Write providers that should be investigated to csv file
providersList = pd.DataFrame(potentialList_combined_unique, columns=['Dentist ID'])
providersList.to_csv('./data/clean/providersList.csv')
