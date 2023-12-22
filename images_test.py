
# Necesario
# Programa que coloca las caracteristicas de las imagenes de PRUEBA en un archivo CVS

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

def combine_data(features, output_csv):
    # Guarda solo las características en un archivo CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in features:
            writer.writerow(row)

features = idx3_to_features('date/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
combine_data(features, 'test.csv')