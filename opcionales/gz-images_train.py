
# Opcional / No relevante
# Programa que convierte el conjunto de datos del archivo .gz a imagenes .jpg en una carpeta llamada "train"

import numpy as np
from PIL import Image
import os

def idx3_to_images(idx1_file, output_folder):
    with open(idx1_file, 'rb') as f:
        
        # Lee el encabezado del archivo IDX3
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Lee los datos de las imágenes
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)

        # Crea el directorio de salida si no existe
        os.makedirs(output_folder, exist_ok=True)

        # Guarda cada imagen como archivo JPEG
        for i in range(num_images):
            image_data = data[i]
            image = Image.fromarray(image_data)

            # Guarda la imagen en formato JPEG
            image_path = os.path.join(output_folder, f'train_{i + 1}.jpg')
            image.save(image_path)

idx3_to_images('date/train-images-idx3-ubyte/train-images-idx3-ubyte', 'train_images')