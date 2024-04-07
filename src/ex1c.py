import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from src.ex1b import fire_rate
import pickle
mypath = "../data/Ej1/"

with open(mypath+'DataTP1.pkl', 'rb') as fp:
    data = pickle.load(fp)

tiempos_disparos = data['tiempos_disparos']
tiempos_velocidades = data['tiempos_velocidades']
velocidades = data['velocidades']

# Conseguir el promedio de las velocidades por cada bin
df = pd.DataFrame({'Velocidad': velocidades}, index=tiempos_velocidades)
df.index = pd.to_datetime(df.index, unit='s')
averaged_velocities = df.resample('1S').mean().interpolate()
averaged_velocities = averaged_velocities['Velocidad'].values[0:-1]


indices_de_neuronas = [44, 207, 331, 643, 656, 660, 699, 779]

tasa_disparo = fire_rate()

disparo_neuronas = tasa_disparo[indices_de_neuronas]

# Plotting
plt.figure(figsize=(10, 6))

for i, neuron_id in enumerate(indices_de_neuronas):
    neuron_spikes = disparo_neuronas[i]
    plt.scatter(averaged_velocities, neuron_spikes, label=f'Neurona {neuron_id}', alpha=0.6)

plt.xlabel('Velocidades promedio')
plt.ylabel('Disparos de neurona')
plt.title('Disparos de neurona vs Velocidades promedio')
plt.legend()
plt.grid(True)
plt.show()

for i, neuron_id in enumerate(indices_de_neuronas):
    neuron_spikes = disparo_neuronas[i]
    plt.scatter(averaged_velocities, neuron_spikes, label=f'Neuron {neuron_id}', alpha=0.6)
    plt.xlabel('Velocidades promedio')
    plt.ylabel('Disparo de neurona ' + str(neuron_id))
    plt.title('Disparos de Neurona ' + str(neuron_id) + ' vs Velocidades promedio')
    plt.legend()
    plt.grid(True)
    plt.show()
