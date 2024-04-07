import numpy as np
import matplotlib.pylab as plt
import pickle


def fire_rate():
    mypath = "../data/Ej1/"

    with open(mypath+'DataTP1.pkl', 'rb') as fp:
        data = pickle.load(fp)

    tiempos_disparos = data['tiempos_disparos']
    tiempos_velocidades = data['tiempos_velocidades']
    velocidades = data['velocidades']

    dt = np.mean(np.diff(tiempos_velocidades))
    N = len(tiempos_disparos)

    ancho_bin = 1

    bin_tiempos_velocidades = tiempos_velocidades[::int(ancho_bin/dt)]
    bin_velocidades = [np.mean(velocidades[i*int(ancho_bin/dt):(i+1)*int(ancho_bin/dt)]) for i in range(len(bin_tiempos_velocidades)-1)] #velocidad media por bin

    tasa_disparo = np.zeros((N, len(bin_tiempos_velocidades)-1))
    for n in range(tasa_disparo.shape[0]):
      tasa_disparo[n, :], _ = np.histogram(tiempos_disparos[n], bins=bin_tiempos_velocidades)

    return tasa_disparo



