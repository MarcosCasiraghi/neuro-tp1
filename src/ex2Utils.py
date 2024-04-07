import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from sklearn.manifold import TSNE


def explained_variance_plot(scaled_matrix):
    nums = np.arange(len(scaled_matrix[0]) + 1)
    var_ratio = []

    for num in nums:
        pca = PCA(num)
        result = pca.fit_transform(scaled_matrix)
        var_ratio.append(np.sum(pca.explained_variance_ratio_))

    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(nums, var_ratio, marker='o')
    plt.xlabel('Cantidad de componentes principales')
    plt.ylabel('Varianza cubierta acumulada')
    plt.show()


def pc1_concentrations_plot(pc1, concentrations):
    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, concentrations)
    plt.xlabel('PC1')
    plt.ylabel('Concentraciones')
    plt.grid(True)
    plt.show()


def pc1_v_pc2_with_concentrations_plot(pc1, pc2, concentrations):
    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, c=concentrations, cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Nivel de concentracion')
    plt.grid(True)
    plt.show()


def pc_3D_plot(pc1, pc2, pc3, concentrations):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pc1, pc2, pc3, c=concentrations, cmap='viridis')
    plt.colorbar(scatter, label='Nivel de concentracion')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Plot 3D de componentes principales con nivel de concentracion')
    plt.show()


def pc_interactable_3D_plot(pc1, pc2, pc3, concentrations):
    trace = go.Scatter3d(
        x=pc1,
        y=pc2,
        z=pc3,
        mode='markers',
        marker=dict(
            size=12,
            color=concentrations,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Nivel de concentracion'),
            symbol='circle'
        ),
        name='Nivel de concentracion'
    )

    # Define layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='PC1'),
            yaxis=dict(title='PC2'),
            zaxis=dict(title='PC3')
        ),
        title='Plot 3D de componentes principales para concentracion',
        legend=dict(
            title='Nivel de concentracion',
            font=dict(size=12)
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


def t_sne_plots(scaled_matrix, concentrations, perplexity_start, perplexity_end, jump):
    perplexity_values = np.arange(perplexity_start, perplexity_end, jump)

    for perplexity in perplexity_values:
        tsne = TSNE(n_components=2, perplexity=perplexity)
        tsne_result = tsne.fit_transform(scaled_matrix)

        tsne_pc1 = tsne_result[:, 0]
        tsne_pc2 = tsne_result[:, 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_pc1, tsne_pc2, c=concentrations, cmap='viridis')
        plt.colorbar(label='Nivel de concentracion')
        plt.title('Analisis T-SNE para niveles de concentracion con perplejidad: ' + str(perplexity))
        plt.show()

