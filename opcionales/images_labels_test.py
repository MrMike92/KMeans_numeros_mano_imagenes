
# Opcional / No relevante
# Programa que junta las caracteristicas de las imagenes con sus respectivas etiquetas de PRUEBA en un archivo CVS

import numpy as np
import csv

def idx3_to_features(idx3_file):
    with open(idx3_file, 'rb') as f:
        # Lee el encabezado del archivo IDX3
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Lee los datos de las imágenes
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)

        return data

def idx1_to_labels(idx1_file):
    with open(idx1_file, 'rb') as f:
        # Lee el encabezado del archivo IDX1
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Lee los datos de las etiquetas
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

def combine_data(features, labels, output_csv):
    # Combina las características con las etiquetas
    combined_data = np.column_stack((features, labels))

    # Guarda los datos combinados en un archivo CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'feature_{i + 1}' for i in range(features.shape[1])] + ['Label']
        writer.writerow(header)

        for row in combined_data:
            writer.writerow(row)

# Uso del script
features = idx3_to_features('date/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
labels = idx1_to_labels('date/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
combine_data(features, labels, 'test_labels.csv')