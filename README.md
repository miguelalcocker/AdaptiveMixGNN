# AdaptiveMixGNN: Red Neuronal de Grafos con Banco de Filtros para Grafos Heterofílicos

Implementación de un modelo de clasificación de nodos basado en principios de Procesamiento de Señales en Grafos (GSP).

## Descripción General

AdaptiveMixGNN aborda el desafío de la heterofilia en redes neuronales de grafos mediante una arquitectura de **Banco de Filtros** que procesa señales de baja frecuencia (homofílicas) y alta frecuencia (heterofílicas) por separado.

### Innovación Clave

El modelo aprende un **parámetro de mezcla α por nodo** α_i ∈ [0,1] que balancea adaptativamente:
- **Filtro paso-bajo** (S_LP): Agrega información de vecinos similares (homofilia)
- **Filtro paso-alto** (S_HP): Captura patrones de vecinos disimilares (heterofilia)

**Hipótesis**: α → 1 en grafos homofílicos, α → 0 en grafos heterofílicos

## Especificación Matemática

### Propagación de Señales

Para cada capa l, la propagación de señales se define como:

```
z_LP = S_LP · X_{l-1}
z_HP = S_HP · X_{l-1}
α_i = σ(x_i · θ + b)           (por nodo)
z_mix = α ⊙ z_LP + (1-α) ⊙ z_HP
X_l = σ(z_mix · W + bias)
```

Donde:

**Operadores de Desplazamiento de Grafo (GSOs):**

1. **GSO Paso-Bajo (S_LP)**: Adyacencia normalizada estilo GCN con auto-bucles
   ```
   S_LP = D̃^(-1/2) · Ã · D̃^(-1/2)
   donde Ã = A + I
   ```
   Referencia: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks

2. **GSO Paso-Alto (S_HP)**: Filtro de diferencia espectral (diseño de banco de filtros)
   ```
   S_HP = I - S_LP
   ```
   Referencia: Teoría de bancos de filtros para procesamiento de señales en grafos

**Parámetros Aprendibles:**

- **θ, b**: Parámetros del predictor de alpha por nodo
  - α_i = sigmoid(x_i · θ + b) para cada nodo i
  - Controla el balance homofilia/heterofilia a nivel de nodo
  - Proporciona interpretabilidad del modelo

- **W**: Matriz de pesos compartida (F_in × F_out)
  - Transforma la señal mezclada
  - Diseño eficiente con peso único

- **bias**: Vector de sesgo

**Función de Activación:**
- σ: ReLU para capas ocultas
- Última capa: Sin activación (retorna logits para CrossEntropyLoss)

### Diseño de la Arquitectura

La arquitectura procesa señales mediante:
1. **Rama de baja frecuencia**: S_LP · X (captura patrones homofílicos)
2. **Rama de alta frecuencia**: S_HP · X (captura patrones heterofílicos)
3. **Mezcla adaptativa por nodo**: α aprendido balancea las contribuciones
4. **Transformación compartida**: W proyecta la señal mezclada

## Características de Implementación

### Eficiencia
- **GSOs pre-calculados**: S_LP y S_HP calculados una vez en `__init__` como tensores COO dispersos
- **Operaciones de matrices dispersas**: Usa `torch.sparse.mm` para convolución eficiente
- Auto-bucles añadidos antes de la normalización para estabilidad numérica
- **Complejidad**: O(E) donde E es el número de aristas

### Interpretabilidad
- **Monitoreo de Alpha**: Valores α por nodo rastreados durante entrenamiento
- **Validación de hipótesis**: Logging CSV permite análisis de distribución de α
- Demuestra comportamiento adaptativo en diferentes tipos de grafos

### Conteo de Parámetros

Para una arquitectura de 2 capas con dimensión oculta H:
```
Total = (F_in + 1) + (F_in × H + H) + (H + 1) + (H × C + C)
```

Donde:
- F_in: Dimensión de entrada (características)
- H: Dimensión oculta
- C: Número de clases

## Instalación

```bash
# Crear entorno
conda create -n adaptivemix python=3.10
conda activate adaptivemix

# Instalar dependencias
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Uso

### Entrenamiento Básico

```bash
python train_ref.py --dataset Cora --epochs 200 --verbose
```

### Extracción de Distribución de Alpha

```bash
python train_ref.py --dataset Cora --save_alpha_distribution --verbose
```

Esto genera archivos CSV con los valores de α por nodo: `alpha_cora.csv`, `alpha_texas.csv`, etc.

### Estudios de Ablación

#### GCN Puro (α=1, solo paso-bajo)
```bash
python train_ref.py --dataset Cora --ablation_mode gcn
```

#### Paso-Alto Puro (α=0)
```bash
python train_ref.py --dataset Cora --ablation_mode hp
```

#### α Aprendido (por defecto)
```bash
python train_ref.py --dataset Cora
```

### Experimentos de Heterofilia

```bash
# Esperado: α → 1 (dataset homofílico)
python train_ref.py --dataset Cora --save_alpha_distribution

# Esperado: α → 1 (dataset homofílico)
python train_ref.py --dataset CiteSeer --save_alpha_distribution

# Esperado: α → 0 (dataset heterofílico)
python train_ref.py --dataset Texas --save_alpha_distribution

# Esperado: α → 0 (dataset heterofílico)
python train_ref.py --dataset Wisconsin --save_alpha_distribution
```

## Argumentos de Línea de Comandos

```
Arquitectura del Modelo:
  --hidden_dim        Dimensión de capa oculta (por defecto: 64)
  --num_layers        Número de capas GNN (por defecto: 2)
  --dropout           Tasa de dropout (por defecto: 0.5)

Entrenamiento:
  --epochs            Épocas de entrenamiento (por defecto: 200)
  --lr                Tasa de aprendizaje (por defecto: 0.01)
  --weight_decay      Regularización L2 (por defecto: 5e-4)
  --patience          Paciencia para early stopping (por defecto: 50)
  --warmup_epochs     Épocas de calentamiento antes de early stopping (por defecto: 100)

Dataset:
  --dataset           Elección de dataset: Cora, CiteSeer, Texas, Wisconsin (por defecto: Cora)

Ablación:
  --ablation_mode     Forzar α: 'gcn' (α=1), 'hp' (α=0), o None (aprendido)

Logging:
  --save_alpha_distribution  Guardar distribución de α por nodo en CSV
  --verbose                  Imprimir progreso detallado
```

## Resultados

### Precisión en Clasificación de Nodos

| Dataset | Tipo | Precisión Test | Parámetros | α Promedio |
|---------|------|----------------|------------|------------|
| Cora | Homofílico | 79.04 ± 0.91% | 93,730 | 0.897 |
| CiteSeer | Homofílico | 65.12 ± 7.92% | 241,215 | 0.842 |
| Texas | Heterofílico | 81.08 ± 4.52% | 111,150 | 0.480 |
| Wisconsin | Heterofílico | 79.61 ± 0.96% | 111,150 | 0.450 |

### Validación de Hipótesis

| Dataset | Tipo | α Esperado | α Observado | Estado |
|---------|------|------------|-------------|--------|
| Cora | Homofílico | α > 0.5 | 0.897 ± 0.027 | CONFIRMADO |
| CiteSeer | Homofílico | α > 0.5 | 0.842 ± 0.175 | CONFIRMADO |
| Texas | Heterofílico | α < 0.5 | 0.480 ± 0.029 | CONFIRMADO |
| Wisconsin | Heterofílico | α < 0.5 | 0.450 ± 0.020 | CONFIRMADO |

**Interpretación**:
- α → 1 indica preferencia por filtrado paso-bajo (homofilia)
- α → 0 indica preferencia por filtrado paso-alto (heterofilia)

## Análisis de Distribución de Alpha

### Visualización

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar valores de alpha
df_cora = pd.read_csv('alpha_cora.csv')
df_texas = pd.read_csv('alpha_texas.csv')

# Crear histogramas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.kdeplot(data=df_cora, x='alpha_value', ax=axes[0], fill=True)
axes[0].set_title('Cora (Homofílico)')
axes[0].axvline(x=0.5, color='red', linestyle='--')

sns.kdeplot(data=df_texas, x='alpha_value', ax=axes[1], fill=True)
axes[1].set_title('Texas (Heterofílico)')
axes[1].axvline(x=0.5, color='red', linestyle='--')

plt.savefig('distribucion_alpha.pdf')
```

### Observaciones Clave

1. **Adaptabilidad**: El modelo aprende automáticamente el balance apropiado para cada tipo de grafo
2. **Interpretabilidad**: α directamente interpreta la preferencia homofilia/heterofilia por nodo
3. **Eficiencia**: GSOs dispersos pre-calculados permiten entrenamiento rápido
4. **Generalización**: Una única arquitectura funciona tanto en grafos homofílicos como heterofílicos

## Estructura del Código

```
.
├── model_ref.py                # Implementación de AdaptiveMixGNN
│   ├── compute_graph_shift_operators()  # Pre-calcular S_LP, S_HP
│   ├── AdaptiveMixGNNLayer              # Capa individual con mezcla α por nodo
│   ├── AdaptiveMixGNN                   # Modelo completo
│   └── get_optimizer()                  # Optimizador con tasas diferenciadas
│
├── train_ref.py                # Script de entrenamiento
│   ├── Extracción de distribución de alpha
│   ├── Soporte de modo ablación
│   ├── Conteo de parámetros
│   └── Evaluación en splits val/test
│
├── plotting_paper.py           # Generación de figuras
│   └── Histogramas de distribución de α
│
├── benchmark_runner.py         # Ejecución de benchmarks
│   └── Genera tabla de resultados
│
└── README_ES.md                # Este archivo
```

## Referencias

### Procesamiento de Señales en Grafos
1. **Kipf & Welling (2017)**: Semi-Supervised Classification with Graph Convolutional Networks (GCN)
2. **Sandryhaila & Moura (2013)**: Discrete Signal Processing on Graphs
3. **Defferrard et al. (2016)**: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering

### Heterofilia en GNNs
- **Zhu et al. (2020)**: Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs
- **Chien et al. (2021)**: Adaptive Universal Generalized PageRank Graph Neural Network
- **Bo et al. (2021)**: Beyond Low-frequency Information in Graph Convolutional Networks

## Licencia

Este código se proporciona con fines de investigación y educación.
