import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv
import glob

from src.ex2Utils import explained_variance_plot, pc1_concentrations_plot, pc1_v_pc2_with_concentrations_plot, \
    pc_3D_plot, pc_interactable_3D_plot, t_sne_plots

csv_file = '../data/Ej2/archive/data/*'

files = glob.glob(csv_file)

data_matrix = []
concentrations = []

# Para conseguir una matriz con todos los valores
for data_file in files:
    with open(data_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            row_data = [float(value) for value in row[1:-1]]  # ecluyendo el primero y el ultimo
            classification = float(row[-1])  # ultimo es la concentracion

            data_matrix.append(row_data)
            concentrations.append(classification)

data_matrix = np.array(data_matrix)
concentrations = np.array(concentrations)

scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(data_matrix)

explained_variance_plot(scaled_matrix)


pca = PCA(len(scaled_matrix[0]))
result = pca.fit_transform(scaled_matrix)

pc1 = result[:, 0]
pc2 = result[:, 1]
pc3 = result[:, 2]

pc1_concentrations_plot(pc1, concentrations)

pc1_v_pc2_with_concentrations_plot(pc1, pc2, concentrations)

pc_3D_plot(pc1, pc2, pc3, concentrations)

pc_interactable_3D_plot(pc1, pc2, pc3, concentrations)

t_sne_plots(scaled_matrix, concentrations, 10, 131, 40)








