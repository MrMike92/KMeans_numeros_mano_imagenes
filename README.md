# k-Means de números escrito a mano
Este es un clasificador K-Means para una base de datos de imagenes de reconocimiento óptico de un conjunto de datos de dígitos escritos a mano.

## Instrucciones de uso.

1. Clona este repositorio en tu máquina local.
2. Asegure que el respositorio se haya descargado correctamente.
3. Descomprimir el zip de la carpeta "date"
4. Ejecute images_train.py o images_test.py para obtener los arhcivos .cvs
5. Ejecute K-Means.py

> [!CAUTION]
> - En el archivo K-Means.py, cambiar la linea #96 el nombre del archivo: data = np.loadtxt('X.csv', delimiter=','), donde X es el nombre que tenga
> - En el archivo K-Means.py, ajustar las iteraciones en la linea #123: max_iterations_range = range(1, Y), dobde Y es el número de iteraciones maxima que se va a evaluar, es decir, del 1 a Y
> - Recuerda: Mientras mayor sea Y, más tardara el programa en evaluar pero tendra menor inercia.
> - Al final el archivo K-Means.py, mostrara lo siguente en este orden:
>    1. Una gráfica con el analisis de las iteraciones.
>    2. Los clusters resultantes del K-Means con la iteración que tuvo menor inercia.
>    3. Creara una carpeta llamada "cluster_images"
>    <br>![image](https://github.com/MrMike92/KMeans_numeros_mano_imagenes/assets/93272523/44d6c9be-31e5-4cf2-8508-e7e396b371ee)
>    3. Dentro de la carpeta "cluster_images" creara carpetas enumaradas segun la cantidad de clusters (en caso 10)
>    <br>![image](https://github.com/MrMike92/KMeans_numeros_mano_imagenes/assets/93272523/93649034-24fb-4136-93fb-09a447f2cb5a)
>    4. Dentro de ellas estara todos los puntos de cada cluster en imagenes con la extensión .jpg para que sea más fácil analizar la presición del agrupamiento.
>    ![image](https://github.com/MrMike92/KMeans_numeros_mano_imagenes/assets/93272523/5c4f32e6-9cfc-48f1-be39-9b4df91197e2)

Se utiliza el dataset MNIST (conjunto de imágenes de dígitos escritos a mano), donde cada imagen del dataset MNIST es de 28x28 píxeles en escala de grises (784 pixeles por imagen).
Lon archivos que contienen las imágenes a blanco y negro que tiene un conjunto de entrenamiento de 60.000 ejemplos y un conjunto de prueba de 10.000 ejemplos.
Hay cuatro archivos disponibles:
*	train-images-idx3-ubyte.gz: imágenes del conjunto de entrenamiento.
*	train-labels-idx1-ubyte.gz: etiquetas del conjunto de entrenamiento.
*	t10k-images-idx3-ubyte.gz: imágenes del conjunto de prueba.
* t10k-labels-idx1-ubyte.gz: etiquetas del conjunto de prueba.

> [!IMPORTANT]
> La base de datos pertenece a sus resprectivos creadores, HOJJAT KHODABAKHSH
> <br>Link de la base de datos: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data

Si deseas contribuir a este proyecto, puedes enviar solicitudes de extracción (pull requests) con mejoras o características adicionales y si tienes alguna pregunta o problema, puedes contactarme a través de mi perfil de GitHub MrMike92. 🐢
