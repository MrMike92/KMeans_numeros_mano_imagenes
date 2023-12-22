"""MIT License
Copyright (c) 2023 MrMike92
Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y de los archivos de documentación asociados (el "Software"), para tratar el Software sin restricciones, incluyendo, sin limitación, los derechos de uso, copia, modificación, fusión, publicación, distribución, sublicencia y/o venta de copias del Software, y para permitir a las personas a las que se les proporcione el Software a hacerlo, sujeto a las siguientes condiciones:
El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o porciones sustanciales del Software.
EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A LAS GARANTÍAS DE COMERCIABILIDAD, ADECUACIÓN PARA UN PROPÓSITO PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DE LOS DERECHOS DE AUTOR SERÁN RESPONSABLES POR CUALQUIER RECLAMO, DAÑO U OTRA RESPONSABILIDAD, YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O DE OTRO MODO, DERIVADA DE, FUERA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTROS TRATOS EN EL SOFTWARE."""

# Opcional / No relevante
# Programa que convierte el conjunto de datos del archivo .gz a imagenes .jpg en una carpeta llamada "test"

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
            image_path = os.path.join(output_folder, f'test_{i + 1}.jpg')
            image.save(image_path)

idx3_to_images('date/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte', 'test_images')
