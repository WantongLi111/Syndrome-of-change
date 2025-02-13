#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import numpy as np
import sys
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import os
import time
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.font_manager
import warnings
warnings.filterwarnings('ignore')
import matplotlib.font_manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.gridspec as gridspec
from shapely.geometry import mapping
import ppca
from ppca import PPCA
import pymannkendall as mk
import cca_zoo
from cca_zoo.linear import MCCA
import scipy.stats as stats

start_time = time.time()

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
plt.rcParams.update({'font.size': 4})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 0.3
plt.rcParams['lines.markersize'] = 0.001


def data_path(filename):
    file_path = "{path}/{filename}".format(
        path="/Users/wantongli/Documents/GitHub/Syndrome-of-change",
        filename=filename
    )
    return file_path


def read_data(path):
    data = np.load(path)
    return data

###### set variable names
bio_var1 = ['albedo', 'albedo_times_ssrd',
           # 'ET_gleam', 'ET_era5',
           'sm', 'tws', 'ndwi',
           'LST_divide_t2m',
           'ndvi', 'nirv', 'sif_gosif',
           'VOD_day','VOD_night','VOD_ratio']
atm_var1 = ['PET', 'ssrd', 'tp', 'rh', 't2m', 't2m_min_min', 't2m_max_max', 't2m_min_mean', 't2m_max_mean', 'cloud', 'PEI_90']

bio_var = []
atm_var = []
column_name = ['','_weightedbyPOP']
for multi_var in range(len(column_name)):
    bio_var.append([s + column_name[multi_var] for s in bio_var1])
    atm_var.append([s + column_name[multi_var] for s in atm_var1])
bio_var = [item for sublist in bio_var for item in sublist]
atm_var = [item for sublist in atm_var for item in sublist]
atm_var = [item for item in atm_var if item not in ['t2m_min_min_weightedbyPOP', 't2m_max_max_weightedbyPOP']]

sio_gapfilled = pd.read_csv(data_path('data/WDI_2003_2022_gapfilled_mask0.7.csv'))
sio_var = sio_gapfilled.drop(columns=['Country Code','year']).columns.values.tolist()


### read data
df_all = pd.read_csv(data_path('data/df_all_annual_normal_MinusSpatialMean_MinusMKTrend_global.csv'))[bio_var+atm_var+sio_var+['year','Country Code']]
df_all = df_all.loc[:, ~df_all.columns.str.contains('^Unnamed')]
print(len(np.unique(df_all['Country Code'])))

## standardised
data_standard = StandardScaler().fit_transform(df_all[bio_var+sio_var+atm_var])
df_standard = pd.DataFrame(data=data_standard, columns=(bio_var+sio_var+atm_var))
df_standard['Country Code'] = df_all['Country Code']
df_standard['year'] = df_all['year']
print(df_standard.shape)

###################################### CCA robustness
domain = [bio_var,atm_var,sio_var]
X_mc = df_standard[domain[0]]
Y_mc = df_standard[domain[1]]
Z_mc = df_standard[domain[2]]
dim=10
mcca = MCCA(latent_dimensions=dim).fit((X_mc.values, Y_mc.values, Z_mc.values))
X_components_ref, Y_components_ref, Z_components_ref = mcca.transform((X_mc.values, Y_mc.values, Z_mc.values))
average_pairwise_correlations = mcca.average_pairwise_correlations((X_mc.values, Y_mc.values, Z_mc.values)) # average correlation
print('r2:',average_pairwise_correlations)

pvalue_sum = np.zeros(dim)
pvalue_num = np.zeros(dim)
n = 10000 # you can adjust it to 100 to get a quick run
cor = np.zeros((n, 3, dim)) * np.nan
sum = 0
for permu in range(n):
    if sum < n:
        bio_var1 = ['albedo', 'albedo_times_ssrd',
                    # 'ET_gleam', 'ET_era5',
                    'sm', 'tws', 'ndwi',
                    'LST_divide_t2m',
                    'ndvi', 'nirv', 'sif_gosif',
                    'VOD_day', 'VOD_night', 'VOD_ratio']
        atm_var1 = ['PET', 'ssrd', 'tp', 'rh', 't2m', 't2m_min_min', 't2m_max_max', 't2m_min_mean', 't2m_max_mean', 'cloud', 'PEI_90']

        bio_var = []
        atm_var = []
        column_name = ['', '_weightedbyPOP']
        for multi_var in range(len(column_name)):
            bio_var.append([s + column_name[multi_var] for s in bio_var1])
            atm_var.append([s + column_name[multi_var] for s in atm_var1])
        bio_var = [item for sublist in bio_var for item in sublist]
        atm_var = [item for sublist in atm_var for item in sublist]
        atm_var = [item for item in atm_var if item not in ['t2m_min_min_weightedbyPOP', 't2m_max_max_weightedbyPOP']]

        sio_gapfilled = pd.read_csv(data_path('data/WDI_2003_2022_gapfilled_mask0.7.csv'))
        sio_var = sio_gapfilled.drop(columns=['Country Code','year']).columns.values.tolist()

        remove_1times3 = np.random.choice(bio_var + atm_var + sio_var, replace=False,
                                          size=int(len(bio_var + atm_var + sio_var) * 1 / 5))
        bio_var = [ele for ele in bio_var if ele not in remove_1times3]
        atm_var = [ele for ele in atm_var if ele not in remove_1times3]
        sio_var = [ele for ele in sio_var if ele not in remove_1times3]

        domain = [bio_var, atm_var, sio_var]
        X_mc = df_standard[domain[0]]
        Y_mc = df_standard[domain[1]]
        Z_mc = df_standard[domain[2]]
        if len(bio_var) > 0 and len(atm_var) > 0 and len(sio_var) > 0:
            if min(len(bio_var), len(atm_var), len(sio_var)) > dim:
                dim_min = dim
            else:
                dim_min = min(len(bio_var), len(atm_var), len(sio_var))
            mcca = MCCA(latent_dimensions=dim_min).fit( (X_mc.values, Y_mc.values, Z_mc.values))
            X_components_test, Y_components_test, Z_components_test = mcca.transform(
                (X_mc.values, Y_mc.values, Z_mc.values))

            for comp in range(dim_min):
                r1, p = stats.pearsonr(X_components_ref[:, comp], X_components_test[:, comp])
                r2, p = stats.pearsonr(Y_components_ref[:, comp], Y_components_test[:, comp])
                r3, p = stats.pearsonr(Z_components_ref[:, comp], Z_components_test[:, comp])
                cor[sum, 0, comp] = np.abs(r1)
                cor[sum, 1, comp] = np.abs(r2)
                cor[sum, 2, comp] = np.abs(r3)

            sum = sum + 1

print(sum)


fig = plt.figure(figsize=(4, 6), dpi=300, tight_layout=True)
for domain in range(3):
    for comp in range(10): ### plot dim 1-10
        ax = fig.add_subplot(dim, 3, 1 + comp * 3 + domain)
        print(cor[:, domain, comp])
        ax.hist(cor[:, domain, comp], 50, color='black')
        ax.axvline(x=np.nanpercentile(cor[:, domain, comp], 50), color='red', ls='--')
        ax.text(np.nanpercentile(cor[:, domain, comp], 50) - 0.1, -400,
                s='r=' + str(np.round(np.nanpercentile(cor[:, domain, comp], 50), 2)), color='r',fontsize=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 3000)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 1500, 3000])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_ylabel('Variate '+str(comp+1),fontsize=5)

        if comp == 0:
            # ax.set_title(['CCX', 'CCY', 'CCZ'][domain] + str(comp) + ', N=' + str(np.sum(~np.isnan(cor[:, domain, comp]))))
            ax.set_title(['CCX', 'CCY', 'CCZ'][domain],fontsize=5)

        if domain != 0:
            plt.gca().axes.get_yaxis().set_visible(False)

        if comp!=9:
            plt.gca().axes.get_xaxis().set_visible(False)

        plt.setp(ax.spines.values(), lw=0.5)

plt.show()
plt.savefig(data_path('figure/sfig_similarity_dim10_remove20%_10000.jpg'), bbox_inches='tight')



