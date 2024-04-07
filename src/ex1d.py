from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from src.ex1b import fire_rate

tasa_diparo = fire_rate()

tasa_diparo = tasa_diparo.transpose()

# std_scaler = StandardScaler()
# scaled_tasa_disparo = std_scaler.fit_transform(tasa_diparo)

nums = np.arange(300)
var_ratio = []
for num in nums:
    pca = PCA(n_components=num)
    pca.fit_transform(tasa_diparo)
    var_ratio.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(nums, var_ratio, marker='o')
plt.xlabel('Cantitad de componentes principales')
plt.ylabel('Varianza explicada cumulativa')
plt.title('Cantidad de componentes principales vs Varianza explicada cumulativa')
plt.show()
