# Regresión logística: predicción de cancelación de pólizas (churn)

Ejercicio de clasificación binaria para predecir si un cliente de una aseguradora cancela su póliza, usando Regresión Logística sobre datos sintéticos. Entregable de la Semana 6 (Machine Learning Supervisado) de la especialización en Data Analytics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USUARIO/REPO/blob/main/regresion_logistica_churn.ipynb)

## Variables

| Variable | Descripción |
|---|---|
| `antiguedad_meses` | Antigüedad del cliente, en meses |
| `prima_mensual` | Valor de la prima mensual |
| `reclamos_12m` | Número de reclamos en los últimos 12 meses |
| `contactos_soporte` | Número de contactos a soporte |
| `cancela` (objetivo) | 0 = renueva, 1 = cancela |

## Datos

Los datos son sintéticos, generados con `numpy` (`random_state`/semilla 42) para que el ejercicio sea reproducible. Las cuatro variables de entrada se simulan de forma independiente y la probabilidad de cancelar se construye con una función logística conocida: más antigüedad reduce la probabilidad de cancelar; prima más alta, más reclamos y más contactos a soporte la aumentan. El valor final de `cancela` se sortea con esa probabilidad (ruido binomial), y el intercepto se ajustó para que la tasa de cancelación quede alrededor del 26 %, un desbalance moderado pero suficiente para que el Accuracy por sí solo no sea confiable.

## Contenido del notebook

- Exploración de los datos (distribución de variables, balance de clases).
- Visualización de la relación de cada variable con la cancelación.
- Partición en entrenamiento y prueba (75/25, estratificada).
- Pipeline con `StandardScaler` + `LogisticRegression`.
- Evaluación con Accuracy, matriz de confusión, precision, recall, F1 y ROC-AUC.
- Barrido de umbrales de decisión para analizar el trade-off precision/recall.
- Interpretación de coeficientes como odds ratios.
- Revisión de estabilidad de las métricas con distintas semillas de partición.

## Resultados principales

Con el umbral por defecto (0.50) el modelo obtiene un Accuracy de 0.78 y un ROC-AUC de 0.735. El Accuracy engaña por el desbalance de clases: la clase "cancela" tiene un Recall de solo 0.30 (matriz de confusión `[[105, 5], [28, 12]]`), es decir, el modelo detecta menos de un tercio de los clientes que realmente cancelan. El ROC-AUC, al no depender de un umbral fijo, muestra que el modelo sí ordena razonablemente bien a los clientes por riesgo de cancelar, aunque el umbral 0.50 no es el adecuado si el objetivo es detectar la mayor cantidad de cancelaciones posible. El barrido de umbrales del notebook permite elegir un punto de operación distinto según el costo de cada tipo de error.

## Cómo ejecutarlo

**En Colab:** usar el badge de arriba (abre el notebook directamente desde este repositorio).

**En local:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook regresion_logistica_churn.ipynb
```

## Autor

Santiago Moreno
