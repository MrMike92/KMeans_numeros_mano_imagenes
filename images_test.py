"""MIT License
Copyright (c) 2023 MrMike92
Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y de los archivos de documentación asociados (el "Software"), para tratar el Software sin restricciones, incluyendo, sin limitación, los derechos de uso, copia, modificación, fusión, publicación, distribución, sublicencia y/o venta de copias del Software, y para permitir a las personas a las que se les proporcione el Software a hacerlo, sujeto a las siguientes condiciones:
El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o porciones sustanciales del Software.
EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A LAS GARANTÍAS DE COMERCIABILIDAD, ADECUACIÓN PARA UN PROPÓSITO PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DE LOS DERECHOS DE AUTOR SERÁN RESPONSABLES POR CUALQUIER RECLAMO, DAÑO U OTRA RESPONSABILIDAD, YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O DE OTRO MODO, DERIVADA DE, FUERA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTROS TRATOS EN EL SOFTWARE."""

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
