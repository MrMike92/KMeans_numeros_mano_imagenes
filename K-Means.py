"""MIT License
Copyright (c) 2023 MrMike92
Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y de los archivos de documentación asociados (el "Software"), para tratar el Software sin restricciones, incluyendo, sin limitación, los derechos de uso, copia, modificación, fusión, publicación, distribución, sublicencia y/o venta de copias del Software, y para permitir a las personas a las que se les proporcione el Software a hacerlo, sujeto a las siguientes condiciones:
El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o porciones sustanciales del Software.
EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A LAS GARANTÍAS DE COMERCIABILIDAD, ADECUACIÓN PARA UN PROPÓSITO PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DE LOS DERECHOS DE AUTOR SERÁN RESPONSABLES POR CUALQUIER RECLAMO, DAÑO U OTRA RESPONSABILIDAD, YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O DE OTRO MODO, DERIVADA DE, FUERA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTROS TRATOS EN EL SOFTWARE."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class KMeans:
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations
        self.centroides = None
        self.labels = None

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def kmeans_plusplus_init(self, x):
        self.centroides = [x[np.random.choice(len(x))]]
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(c - x_i) ** 2 for c in self.centroides]) for x_i in x])
            probabilities = distances / distances.sum()
            cum_probabilities = probabilities.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cum_probabilities):
                if r < p:
                    self.centroides = np.append(self.centroides, [x[j]], axis=0)
                    break

    def fitting(self, x):
        self.kmeans_plusplus_init(x)
        for _ in range(self.max_iterations):
            distances = np.array([[self.euclidean_distance(xi, cj) for cj in self.centroides] for xi in x])
            self.labels = np.argmin(distances, axis=1)
            self.centroides = np.array([x[self.labels == idx].mean(axis=0) for idx in range(self.k)])
        return self.labels, self.centroides

def calculate_inertia(data, labels, centroides):
    distances = np.array([np.sum((data[i] - centroides[labels[i]])**2) for i in range(len(data))])
    return np.sum(distances)

def iterations_analysis(data, k, max_iterations_range):
    inertia_values = []
    for max_iterations in max_iterations_range:
        kmeans = KMeans(k=k, max_iterations=max_iterations)
        labels, centroides = kmeans.fitting(data)
        inertia = calculate_inertia(data, labels, centroides)
        inertia_values.append(inertia)
    return inertia_values

def visualize_iterations_analysis(max_iterations_range, inertia_values):
    plt.plot(max_iterations_range, inertia_values, marker='o')
    plt.title('Análisis de Iteraciones')
    plt.xlabel('Número Máximo de Iteraciones')
    plt.ylabel('Inercia')
    plt.show()

def visualize_clusters(data, labels, centroides, max_iterations):
    u_labels = np.unique(labels)
    for i in u_labels:
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1} ({len(cluster_points)} puntos)', alpha=0.4)
    plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', label='centroides')
    plt.title(f'Clusters (K-Means con {max_iterations} iteraciones)')
    plt.legend()
    plt.show()

def save_cluster_images_as_jpeg(data, labels, centroides, output_folder):
    u_labels = np.unique(labels)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    for i in u_labels:
        cluster_points = data[labels == i]
        cluster_folder = os.path.join(output_folder, f'Cluster{i + 1}')
        os.makedirs(cluster_folder, exist_ok=True)
        for idx, point in enumerate(cluster_points):
            # Escala los valores de las características a [0, 255]
            scaled_point = ((point - np.min(point)) / (np.max(point) - np.min(point))) * 255
            image_data = scaled_point.astype(np.uint8).reshape(28, 28)

            # Crea una imagen a partir de los datos
            image = Image.fromarray(image_data)

            # Guardar la imagen en formato JPG
            image_path = os.path.join(cluster_folder, f'punto_{idx+1}.jpg')
            image.save(image_path)

def main():
    try:
        data = np.loadtxt('test.csv', delimiter=',')
    except FileNotFoundError:
        print("Error: Archivo no encontrado.")
        return
    
    # Paso 1: Centrar los datos
    mean_data = np.mean(data, axis=0)
    centered_data = data - mean_data
    
    # Paso 2: Calcular la matriz de covarianza
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # Paso 3: Calcular autovectores y autovalores manualmente
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Ordenar autovectores por autovalores descendentes
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Paso 4: Seleccionar los primeros dos componentes principales manualmente
    components = eigenvectors[:, :2]
    
    # Transformar los datos
    df = np.dot(centered_data, components)
    
    # Aplicar análisis de iteraciones
    max_iterations_range = range(1, 101)
    inertia_values = iterations_analysis(df, k=10, max_iterations_range=max_iterations_range)
    
    # Visualizar el análisis de iteraciones
    visualize_iterations_analysis(max_iterations_range, inertia_values)
    
    # Elegir el valor de max_iterations con menor valor
    chosen_max_iterations = np.argmin(inertia_values) + 1
    
    # Aplicar KMeans con el número óptimo de iteraciones
    kmeans = KMeans(k=10, max_iterations=chosen_max_iterations)
    labels, centroides = kmeans.fitting(df)
    
    # Visualizar los datos y los centroides
    visualize_clusters(df, labels, centroides, chosen_max_iterations)
    
    # Guardar las imágenes en formato JPEG en carpetas separadas para cada cluster
    save_cluster_images_as_jpeg(data, labels, centroides, output_folder='cluster_images')

if __name__ == "__main__":
    main()
