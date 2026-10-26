# Predicción de supervivencia en el Titanic | Machine Learning

### Análisis de datos, clasificación y selección de modelos con Python

##  1. Descripción del proyecto

Este proyecto desarrolla un sistema de **Machine Learning supervisado** para predecir la supervivencia de los pasajeros del Titanic a partir de sus características personales y de viaje.

Se utiliza el dataset de la competición [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic), disponible en Kaggle.

El trabajo abarca las principales etapas de un proyecto de aprendizaje automático: preparación de datos, ingeniería de características, entrenamiento, evaluación, optimización de hiperparámetros, selección de modelos y generación de predicciones sobre datos no etiquetados.

**Objetivo principal:** identificar el modelo de clasificación que consiga un buen equilibrio entre precisión predictiva, capacidad de generalización, estabilidad y complejidad computacional.

##  2. Objetivos

- Preparar y transformar los datos para su utilización en modelos de Machine Learning.
- Identificar las características más relevantes para predecir la supervivencia.
- Entrenar diferentes algoritmos de clasificación supervisada.
- Evaluar los modelos mediante métricas de clasificación.
- Detectar posibles problemas de sobreajuste (*overfitting*).
- Optimizar hiperparámetros mediante validación cruzada.
- Comparar modelos tradicionales con redes neuronales.
- Seleccionar un modelo y evaluar sus predicciones en Kaggle.

##  3. Tecnologías utilizadas

| Tecnología | Aplicación |
|---|---|
| Python | Lenguaje de programación principal |
| Pandas | Manipulación y transformación de datos |
| NumPy | Operaciones numéricas |
| Matplotlib | Visualización de resultados |
| Seaborn | Análisis gráfico y comparaciones |
| SciPy | Herramientas estadísticas |
| Scikit-learn | Entrenamiento, evaluación y optimización de modelos |
| TensorFlow y Keras | Implementación de redes neuronales |
| Jupyter Notebook | Desarrollo y documentación de experimentos |
| Git y GitHub | Control de versiones |
| Kaggle | Evaluación externa de las predicciones |

## 4. Dataset y variables

El conjunto de datos corresponde a pasajeros del Titanic. La variable objetivo es:

**`Survived`**

- `0`: el pasajero no sobrevivió.
- `1`: el pasajero sobrevivió.

Entre las características empleadas o generadas durante el preprocesamiento se encuentran:

| Variable | Descripción |
|---|---|
| `Pclass` | Clase del billete |
| `Age` | Edad del pasajero |
| `Sex_male` | Codificación del sexo |
| `Fare_log` | Transformación logarítmica de la tarifa |
| `FamilySize` | Tamaño del grupo familiar |
| `IsAlone` | Indicador de si viajaba solo |
| `Has_Cabin` | Indicador de disponibilidad de información sobre cabina |
| `Embarked_Q`, `Embarked_S` | Variables del puerto de embarque |
| `Title_Mr`, `Title_Mrs`, `Title_Miss`, `Title_Rare` | Tratamientos extraídos del nombre |

Estas características permiten representar la información original de forma adecuada para los algoritmos de clasificación.

## 5. Metodología

### 5.1. Preprocesamiento

La preparación de los datos se desarrolla en `02_Preprocessing.ipynb`.

Las transformaciones aplicadas incluyen tratamiento de valores ausentes, codificación de variables categóricas, ingeniería de características y escalado de variables numéricas.

El objetivo es obtener datasets limpios y preparados para el entrenamiento de los modelos.

### 5.2. Entrenamiento de modelos

Se han implementado y comparado los siguientes algoritmos:

**Modelos tradicionales**

1. Regresión Logística.
2. Árbol de Decisión.
3. Random Forest.
4. Support Vector Machine (SVM) con kernel lineal.
5. SVM con kernel RBF.

**Modelos optimizados**

6. Random Forest optimizado.
7. SVM optimizado.

**Redes neuronales**

8. Red neuronal con `MLPClassifier` de scikit-learn.
9. Red neuronal con Keras y TensorFlow.

Las redes neuronales utilizan dos capas ocultas de 64 y 32 neuronas. La implementación con Keras incorpora `Dropout` y `EarlyStopping` para regularizar el entrenamiento.

### 5.3. Evaluación

Se emplean diferentes métricas de clasificación:

- **Accuracy:** proporción de predicciones correctas.
- **Precision:** proporción de positivos predichos que realmente son positivos.
- **Recall:** capacidad para identificar los casos positivos.
- **F1-Score:** media armónica de precision y recall.
- **ROC-AUC:** capacidad de discriminación entre clases.
- **Matriz de confusión:** distribución de aciertos y errores por clase.

También se compara la accuracy en entrenamiento y test para identificar posibles problemas de sobreajuste.

### 5.4. Validación cruzada

Los modelos se comparan mediante validación cruzada estratificada de **10 particiones** (`StratifiedKFold`).

Se utiliza principalmente la accuracy media de validación cruzada para seleccionar el modelo, evitando basar la decisión exclusivamente en un único reparto de datos.

### 5.5. Optimización de hiperparámetros

Se utiliza `GridSearchCV` para optimizar Random Forest y SVM.

En Random Forest se estudian parámetros relacionados con la complejidad del modelo, mientras que en SVM se optimizan parámetros como `C` y `gamma`.

### 5.6. Análisis de estabilidad

Se realiza un experimento adicional utilizando **30 repartos diferentes de entrenamiento y test**.

Se comparan Regresión Logística, Random Forest optimizado y SVM optimizado para analizar cuánto varía el rendimiento según el reparto de los datos.

## 6. Resultados

### Comparación de los principales modelos

| Modelo | Accuracy CV | Accuracy test |
|---|---:|---:|
| Random Forest optimizado | **83,57 %** | 79,33 % |
| SVM optimizado | 83,14 % | **83,24 %** |
| SVM lineal | 83,00 % | 82,68 % |
| Regresión Logística | 82,31 % | 81,01 % |
| Red neuronal scikit-learn | 81,45 % | 81,01 % |

Los resultados de Keras se sitúan aproximadamente entre el 82 % y el 83 % de accuracy en validación cruzada, con pequeñas variaciones entre ejecuciones.

**El Random Forest optimizado fue seleccionado** por obtener la mayor accuracy de validación cruzada en el experimento utilizado para elegir el modelo final.

### Estabilidad de los modelos

| Modelo | Accuracy media | Desviación típica |
|---|---:|---:|
| Random Forest optimizado | 83,3 % | 1,9 % |
| Regresión Logística | 82,7 % | 2,1 % |
| SVM optimizado | **83,5 %** | **1,6 %** |

Aunque Random Forest obtuvo el mejor resultado de validación cruzada, el SVM optimizado presentó una estabilidad ligeramente superior en el experimento de 30 repartos.

Las diferencias fueron reducidas, por lo que no se puede afirmar una superioridad estadística entre ambos únicamente a partir de estos resultados.

##  7. Resultado en Kaggle

El modelo final se entrenó con los datos disponibles y se utilizó para generar predicciones de supervivencia sobre los pasajeros del conjunto de test de Kaggle.

**Puntuación oficial obtenida:**

| Evaluación | Resultado |
|---|---:|
| Mejor accuracy en validación cruzada | 83,57 % |
| Accuracy del modelo seleccionado en test local | 79,33 % |
| **Kaggle Public Score** | **0.78229** |
| **Accuracy pública en Kaggle** | **78,23 %** |

La puntuación pública de Kaggle se calcula sobre una parte de los datos de evaluación ocultos.

El resultado muestra una diferencia moderada respecto al rendimiento local, lo que refleja la variabilidad del comportamiento predictivo entre conjuntos de datos.

**Competición:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

##  8. Principales conclusiones

**1. La validación cruzada es fundamental para seleccionar modelos.**

Un modelo puede obtener buenos resultados en un reparto concreto y comportarse de forma distinta en otros. Evaluarlo mediante varias particiones permite conseguir una estimación más robusta.

**2. Una mayor complejidad no garantiza mejores resultados.**

Las redes neuronales ofrecieron resultados competitivos, pero no demostraron una mejora clara frente a los modelos tradicionales optimizados.

**3. Es importante identificar el sobreajuste.**

El Árbol de Decisión y el Random Forest sin optimizar mostraron una diferencia importante entre accuracy de entrenamiento y de evaluación, indicando una menor capacidad de generalización.

**4. El rendimiento depende parcialmente del reparto de los datos.**

El experimento con 30 semillas mostró diferencias en la accuracy incluso utilizando el mismo algoritmo.

**5. Los resultados de Kaggle son una evaluación externa del modelo.**

La puntuación obtenida permite comprobar cómo se comportan las predicciones sobre datos diferentes a los utilizados durante el desarrollo.

##  9. Estructura del repositorio

- **`notebooks/02_Preprocessing.ipynb`**: limpieza de datos, transformaciones e ingeniería de características.
- **`notebooks/03_ModelSelection.ipynb`**: entrenamiento, evaluación, comparación, optimización y selección de modelos.
- **`requirements.txt`**: dependencias necesarias para ejecutar el proyecto.
- **`.gitignore`**: exclusión de datos, modelos entrenados, archivos temporales y credenciales.
- **`README.md`**: documentación del proyecto.

Los datasets y archivos de modelos no se incluyen en el repositorio.

##  10. Instalación y ejecución

### Requisitos

- Python y un entorno compatible con las dependencias del proyecto.
- Jupyter Notebook o Visual Studio Code con soporte para notebooks.
- Los datasets originales de Titanic disponibles en Kaggle.

### Instalación

**1. Clonar el repositorio**

`git clone https://github.com/jaimeamatt-prog/titanic-machine-learning.git`

**2. Acceder al proyecto**

`cd titanic-machine-learning`

**3. Instalar dependencias**

`pip install -r requirements.txt`

**4. Descargar los datos**

Descargar `train.csv` y `test.csv` desde la competición Titanic de Kaggle.

Crear la estructura de datos utilizada por los notebooks:

- `00_Data/Raw/train.csv`
- `00_Data/Raw/test.csv`
- `00_Data/Cleaned/train_clean.csv` — generado durante el preprocesamiento.
- `00_Data/Cleaned/test_clean.csv` — generado durante el preprocesamiento.

**5. Ejecutar los notebooks en orden**

Primero, ejecutar `02_Preprocessing.ipynb` para generar los datos procesados.

Después, ejecutar `03_ModelSelection.ipynb` para entrenar, comparar y seleccionar los modelos.

**Nota sobre las rutas:** el notebook de selección utiliza rutas relativas del tipo `./00_Data/...`. Según el directorio de trabajo configurado en Jupyter o VS Code, puede ser necesario adaptarlas a `../00_Data/...` si se ejecuta desde la carpeta `notebooks/`.

### Envío a Kaggle

El notebook incluye el código para generar `submission.csv`, con las columnas `PassengerId` y `Survived`.

El envío mediante API de Kaggle es opcional y requiere autenticación. No se ejecuta automáticamente al abrir el notebook.

##  11. Limitaciones y posibles mejoras

Entre las principales limitaciones del proyecto destacan:

- Tamaño relativamente reducido del dataset.
- Sensibilidad de los resultados al reparto de entrenamiento y test.
- Posible sobreajuste de algunos modelos.
- Riesgo de estimaciones optimistas al realizar transformaciones de preprocesamiento antes de la partición local.
- Necesidad de validación cruzada anidada para estimar con menos sesgo el rendimiento de modelos optimizados.

Como mejoras futuras se podrían estudiar nuevos métodos de ingeniería de características, pipelines de preprocesamiento sin fuga de información, validación cruzada anidada y técnicas de ensamblado de modelos.

##  12. Referencias

- [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Proyecto de clasificación supervisada desarrollado con Python como práctica de Machine Learning y análisis de datos.**