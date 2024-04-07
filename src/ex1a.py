import numpy as np
import matplotlib.pylab as plt
import pickle
mypath = "../data/Ej1/"

with open(mypath+'DataTP1.pkl', 'rb') as fp:
    data = pickle.load(fp)

tiempos_disparos = data['tiempos_disparos']
tiempos_velocidades = data['tiempos_velocidades']
velocidades = data['velocidades']

numNeurons = len(tiempos_disparos)

# raster plot
for neuron_idx, spikes in enumerate(tiempos_disparos):
    plt.eventplot(spikes, lineoffsets=neuron_idx, colors='blue')

plt.ylim(-1, numNeurons)
plt.gca().invert_yaxis()
plt.grid(True)
plt.ylabel("Disparos de Nueronas")
plt.xlabel("Tiempo (s)")
plt.title("Disparos de neuronas en funcion del tiempo")
plt.show()

# Velocidad vs Tiempo
plt.plot(tiempos_velocidades, velocidades)
plt.ylabel("Velocidad (m/s)")
plt.xlabel("Tiempo (s)")
plt.title("Velocidad vs Tiempo")
plt.show()




