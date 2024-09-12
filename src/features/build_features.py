from enum import unique
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import cluster
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
predictor_columns = list(df.columns[:6]) # acc_(x,y,z) and gyr_(x,y,z)
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
   df[col] = df[col].interpolate()   # interpolation estimate the value of a function at an unknown point based on its known values at other points. 


df.info()   

# --------------------------------------------------------------

# Calculating set duration
# --------------------------------------------------------------
# Finding the set duration and average set duration so that we can filtter out the noise data  (uncessary movement during the rep) 

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration=df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0] 
duration.seconds

for s in df["set"].unique():
   
   start=df[df["set"] == s].index[0] 
   stop=df[df["set"] == s].index[-1] 
   
   duration= stop - start
   df.loc[(df["set"] == s) , "duration"] = duration.seconds
   
duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] /5  #Average
duration_df.iloc[1] /10

#This give us some info to apply Butterworth lowpass filter

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
 # removes high frequency noise from dataset 
df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200.
cutt_off =1.3 # low = smooth , high= raw  , cutoff frequency

df_lowpass =LowPass.low_pass_filter(df_lowpass, "acc_y", fs , cutt_off, order=5)

subset = df_lowpass [df_lowpass ["set"] == 45]
print(subset["label"] [0])


fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset ["acc_y"].reset_index(drop=True), label="raw data")

ax [1].plot(subset ["acc_y_lowpass"].reset_index(drop=True), label="butterworthfilter")

ax [0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

ax [1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
   df_lowpass =LowPass.low_pass_filter(df_lowpass, col , fs , cutt_off, order=5)
   df_lowpass[col]= df_lowpass[col + "_lowpass"]
   del df_lowpass[col +"_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca [df_pca ["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()



# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared ["acc_y"] ** 2 + df_squared ["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared ["gyr_y"] **2 + df_squared ["gyr_z"] ** 2

df_squared ["acc_r"] = np.sqrt(acc_r)
df_squared ["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared ["set"] == 14]

subset[["acc_r", "gyr_r"]].plot(subplots=True)
df_squared

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

#Here we counter the daata spill of the mix of different set into the next  data set

df_temporal = df_squared.copy()
df_temporal.info()

NumsAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 /200)  #Window size finding  


for col in predictor_columns:
   df_temporal = NumsAbs.abstract_numerical(df_temporal, [col], ws, "mean")
   df_temporal = NumsAbs.abstract_numerical(df_temporal, [col], ws, "std") #standard deviation


df_temporal.info()


df_temporal_list = []
for s in df_temporal["set"].unique():
   subset = df_temporal[df_temporal["set"] == s].copy()
   for col in predictor_columns:       
      subset = NumsAbs.abstract_numerical(subset,[col], ws, "mean")
      subset = NumsAbs.abstract_numerical(subset,[col], ws, "std")
   df_temporal_list.append(subset)
# when we take the mean/std of first value of any set it would take the values  from previous set -> addition of noise value   

df_temporal = pd.concat(df_temporal_list) #stored in original df so that it can overwrite it 
df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq= df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int ( 1000 / 200)
ws = int ( 2000/ 200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)


df_freq_list = []
for s in df_freq["set"].unique():
   print(f"Appying Fourier transformation to set {s}")
   subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
   subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
   df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
# All the extra col are based on rolling window the value in all the col between the rdifferent rows are highly co-related which should be avoided while building model because it cause overfitting to tackal that we allow for certain % of overlap and remove the remaining data
 
df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]  #Skipping the every other row

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_colums = ["acc_x", "acc_y", "acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
   subset = df_cluster[cluster_colums]
   kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
   cluster_labels = kmeans.fit_predict(subset) # Subset as input 
   inertias.append(kmeans.inertia_)
 
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
subset = df_cluster[cluster_colums]
df_cluster["cluster"] = kmeans.fit_predict(subset) # Subset as input 

fig= plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
   subset = df_cluster[df_cluster["cluster"] == c] 
   ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

fig= plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
   subset = df_cluster[df_cluster["label"] == l] 
   ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
