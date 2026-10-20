# Predicción de renuncia de clientes (Customer Churn)

Proyecto del **Desafío 4 – Redes Neuronales (Parte I)** del curso de Data Science de Desafío Latam.

**Autor:** Jose Ignacio Kuoman Jimenez

## Descripción

A partir de una base de datos de una compañía de telecomunicaciones (3.150 clientes, 13 variables regresoras y la etiqueta `Churn`), se construyen modelos para predecir qué clientes tienen más probabilidad de renunciar al servicio. Así la empresa puede tomar acciones de retención dirigidas.

## Contenido

| Archivo | Descripción |
|---|---|
| `Desafio_4-Prediccion_de_renuncia_de_clientes-Jose_Kuoman_Jimenez.ipynb` | Notebook con todo el análisis y los modelos |
| `Customer Churn.csv` | Conjunto de datos |
| `requirements.txt` | Librerías necesarias |

## Etapas del análisis

1. **Preprocesamiento:** revisión de valores nulos, distribución de variables categóricas y continuas, y detección de outliers.
2. **Análisis de `Subscription Length` por clase de `Churn`** y estandarización de las variables regresoras.
3. **Balanceo de clases con SMOTE** y división en entrenamiento y test (33 %).
4. **Random Forest** con hiperparámetros por defecto: F1-Score por clase y las 7 variables más importantes.
5. **Red neuronal feedforward** (Keras/TensorFlow) con una capa oculta y optimizador SGD, entrenada con esas 7 variables, con el objetivo de lograr un AUC mayor al 90 %. Se muestran la curva ROC y su AUC.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook
```

Luego abre el notebook. El archivo `Customer Churn.csv` tiene que estar en la misma carpeta.
