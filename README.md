# AdaptiveMixGNN: Red Neuronal de Grafos con Banco de Filtros para Grafos Heterofílicos

Implementación de un modelo de clasificación de nodos basado en principios de Procesamiento de Señales en Grafos (GSP).

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)

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

**Operadores de Desplazamiento de Grafo (GSOs):**

1. **GSO Paso-Bajo (S_LP)**: Adyacencia normalizada estilo GCN
   ```
   S_LP = D̃^(-1/2) · Ã · D̃^(-1/2)
   donde Ã = A + I
   ```

2. **GSO Paso-Alto (S_HP)**: Filtro de diferencia espectral
   ```
   S_HP = I - S_LP
   ```

## Instalación

```bash
# Crear entorno
conda create -n adaptivemix python=3.10
conda activate adaptivemix

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Entrenamiento Básico

```bash
python train.py --dataset Cora --epochs 200 --verbose
```

### Benchmark de AdaptiveMixGNN

```bash
# Solo nuestro modelo en todos los datasets
python benchmark.py

# Comparativa con baselines (MLP, GCN, GAT)
python benchmark.py --baselines

# Con mas ejecuciones para estadisticas
python benchmark.py --baselines --runs 10
```

### Generación de Figuras

```bash
# Grafico de barras comparando baselines
python plot.py --baselines

# Histogramas de distribucion de alpha (requiere CSVs)
python plot.py --alpha

# Todas las figuras
python plot.py --all
```

### Extracción de Distribución de Alpha

Para generar los histogramas de alpha, primero hay que extraer los datos:

```bash
# Generar CSV para Cora
python train.py --dataset Cora --save_alpha_distribution --epochs 200
mv alpha_distribution_results.csv alpha_cora.csv

# Generar CSV para Texas
python train.py --dataset Texas --save_alpha_distribution --epochs 200
mv alpha_distribution_results.csv alpha_texas.csv

# Generar figura
python plot.py --alpha
```

## Argumentos de Línea de Comandos

### train.py

```
Arquitectura:
  --hidden_dim        Dimensión de capa oculta (default: 64)
  --num_layers        Número de capas GNN (default: 2)
  --dropout           Tasa de dropout (default: 0.5)

Entrenamiento:
  --epochs            Épocas de entrenamiento (default: 200)
  --lr                Tasa de aprendizaje (default: 0.01)
  --weight_decay      Regularización L2 (default: 5e-4)
  --patience          Paciencia para early stopping (default: 50)
  --warmup_epochs     Épocas de warmup para α (default: 20)

Dataset:
  --dataset           Cora, CiteSeer, Texas, Wisconsin, Cornell (default: Cora)

Logging:
  --save_alpha_distribution  Guardar distribución de α por nodo en CSV
  --log_alpha               Guardar evolución de α por época
  --verbose                 Imprimir progreso detallado
```

### benchmark.py

```
  --baselines         Incluir comparativa con MLP, GCN, GAT
  --runs              Número de ejecuciones (default: 5)
  --epochs            Épocas por ejecución (default: 200)
```

## Resultados

### Precisión en Clasificación de Nodos

| Dataset | Tipo | Precisión Test | α Promedio |
|---------|------|----------------|------------|
| Cora | Homofílico | 79.54 ± 0.33% | 0.897 |
| CiteSeer | Homofílico | 68.14 ± 0.57% | 0.842 |
| Texas | Heterofílico | 80.00 ± 1.32% | 0.480 |
| Wisconsin | Heterofílico | 80.78 ± 0.78% | 0.450 |

### Comparativa con Baselines

| Modelo | Cora | CiteSeer | Texas | Wisconsin |
|--------|------|----------|-------|-----------|
| MLP | 56.54 ± 1.02 | 57.10 ± 1.07 | 77.84 ± 1.08 | 79.22 ± 1.57 |
| GCN | 81.22 ± 0.76 | 68.42 ± 0.52 | 62.70 ± 2.02 | 53.73 ± 2.00 |
| GAT | 80.44 ± 1.07 | 67.72 ± 0.95 | 61.08 ± 5.82 | 51.76 ± 5.05 |
| **AdaptiveMixGNN** | 79.54 ± 0.33 | 68.14 ± 0.57 | **80.00 ± 1.32** | **80.78 ± 0.78** |

### Validación de Hipótesis

| Dataset | Tipo | α Esperado | α Observado | Estado |
|---------|------|------------|-------------|--------|
| Cora | Homofílico | α > 0.5 | 0.897 | CONFIRMADO |
| CiteSeer | Homofílico | α > 0.5 | 0.842 | CONFIRMADO |
| Texas | Heterofílico | α < 0.5 | 0.480 | CONFIRMADO |
| Wisconsin | Heterofílico | α < 0.5 | 0.450 | CONFIRMADO |

## Estructura del Código

```
.
├── model.py           # Implementación de AdaptiveMixGNN
│   ├── compute_graph_shift_operators()  # Pre-calcular S_LP, S_HP
│   ├── AdaptiveMixGNNLayer              # Capa con mezcla α por nodo
│   ├── AdaptiveMixGNN                   # Modelo completo
│   └── get_optimizer()                  # Optimizador con LR diferenciado
│
├── train.py           # Script de entrenamiento
│   ├── Carga de datasets (Planetoid, WebKB)
│   ├── Training loop con early stopping
│   └── Extracción de distribución de alpha
│
├── benchmark.py       # Suite de benchmarks
│   ├── run_benchmark_ours()      # Solo AdaptiveMixGNN
│   └── run_benchmark_baselines() # Comparativa MLP/GCN/GAT
│
├── plot.py            # Generación de figuras
│   ├── plot_baselines_comparison()  # Gráfico de barras
│   └── plot_alpha_distribution()    # Histogramas de α
│
├── figuras/           # Figuras generadas
├── requirements.txt   # Dependencias
└── README.md          # Este archivo
```

## Referencias

### Procesamiento de Señales en Grafos
- Kipf & Welling (2017): Semi-Supervised Classification with GCNs
- Sandryhaila & Moura (2013): Discrete Signal Processing on Graphs
- Defferrard et al. (2016): Convolutional Neural Networks on Graphs

### Heterofilia en GNNs
- Zhu et al. (2020): Beyond Homophily in Graph Neural Networks
- Chien et al. (2021): Adaptive Universal Generalized PageRank GNN
- Bo et al. (2021): Beyond Low-frequency Information in GCNs

## Licencia

Este código se proporciona con fines de investigación y educación.
