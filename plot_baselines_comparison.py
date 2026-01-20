"""
Genera gráfico de barras comparando MLP, GCN, GAT, AdaptiveMixGNN
"""

import matplotlib.pyplot as plt
import numpy as np

# Resultados del benchmark
datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin']

# Accuracy media (del benchmark_baselines.py)
results = {
    'MLP': [56.54, 57.10, 77.84, 79.22],
    'GCN': [81.22, 68.42, 62.70, 53.73],
    'GAT': [80.44, 67.72, 61.08, 51.76],
    'AdaptiveMixGNN': [79.54, 68.14, 80.00, 80.78],
}

# Errores estándar
errors = {
    'MLP': [1.02, 1.07, 1.08, 1.57],
    'GCN': [0.76, 0.52, 2.02, 2.00],
    'GAT': [1.07, 0.95, 5.82, 5.05],
    'AdaptiveMixGNN': [0.33, 0.57, 1.32, 0.78],
}

# Colores
colors = {
    'MLP': '#95a5a6',           # Gris
    'GCN': '#e74c3c',           # Rojo
    'GAT': '#f39c12',           # Naranja
    'AdaptiveMixGNN': '#3498db', # Azul
}

# Crear figura
fig, ax = plt.subplots(figsize=(11, 5.5))

x = np.arange(len(datasets))
width = 0.18
multiplier = 0

for model, accuracy in results.items():
    offset = width * multiplier
    bars = ax.bar(x + offset, accuracy, width,
                  label=model, color=colors[model],
                  yerr=errors[model], capsize=3,
                  error_kw={'linewidth': 1})
    multiplier += 1

# Separador visual entre homofílicos y heterofílicos (entre índice 1 y 2)
separator_x = 1.5 + width * 1.5  # Centrado entre CiteSeer y Texas
ax.axvline(x=separator_x, color='#7f8c8d', linestyle='--', linewidth=1.5, alpha=0.6)

# Anotaciones de tipo de grafo
ax.text(0.75 + width*1.5, 94, 'Homofílicos', ha='center', fontsize=11,
        color='#27ae60', fontweight='bold')
ax.text(2.75 + width*1.5, 94, 'Heterofílicos', ha='center', fontsize=11,
        color='#c0392b', fontweight='bold')

# Configuración
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_title('Comparativa de Arquitecturas en Clasificación de Nodos',
             fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(loc='lower left', fontsize=10, ncol=2)
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Guardar
plt.savefig('figure_baselines_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_baselines_comparison.png', dpi=150, bbox_inches='tight')
print("Gráficas guardadas: figure_baselines_comparison.pdf/png")

plt.close()
