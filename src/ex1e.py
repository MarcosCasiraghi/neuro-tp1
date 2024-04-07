from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.ex1b import fire_rate

import pickle
mypath = "../data/Ej1/"

with open(mypath+'DataTP1.pkl', 'rb') as fp:
    data = pickle.load(fp)

tiempos_velocidades = data['tiempos_velocidades']
velocidades = data['velocidades']

# Conseguir el promedio de las velocidades por cada bin
df = pd.DataFrame({'Velocidad': velocidades}, index=tiempos_velocidades)
df.index = pd.to_datetime(df.index, unit='s')
averaged_velocities = df.resample('1S').mean().interpolate()
averaged_velocities = averaged_velocities['Velocidad'].values[0:-1]

tasa_disparo = fire_rate().transpose()

# std_scaler = StandardScaler()
# scaled_tasa_disparo = std_scaler.fit_transform(tasa_disparo)

pca = PCA(n_components=len(fire_rate()[0]))
pca_result = pca.fit_transform(tasa_disparo)

pc1 = pca_result[:, 0]
pc2 = pca_result[:, 1]

# PC1 vs PC2
plt.figure(figsize=(8, 6))
plt.scatter(pc1, pc2)
plt.title('PC1 vs PC2')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()


# PC1 vs Tiempo
time_bins = np.arange(tasa_disparo.shape[0])
plt.figure(figsize=(8, 6))
plt.scatter(time_bins, pc1)
plt.title('PC1 vs Time')
plt.xlabel('Tiempo (Bins 1s)')
plt.ylabel('Componente Principal 1')
plt.grid(True)
plt.show()


# PC1 vs Velocidad
plt.figure(figsize=(8,6))
plt.scatter(averaged_velocities, pc1)
plt.title('PC1 vs Velocidad')
plt.xlabel('Velocidad Promedio m/s (Bins 1s)')
plt.ylabel('Componente Principal 1')
plt.grid(True)
plt.show()
