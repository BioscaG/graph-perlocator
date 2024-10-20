import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np


def load_data(graph_type=None, n_range=None, k_range=None, q_range=None, perl_vertex=None):
    # Definir el directorio base
    base_dir = Path('data')

    # Inicializar lista para almacenar los DataFrames
    all_data = []

    # Recorrer las subcarpetas y cargar los CSVs
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                folder = Path(root)
                folder_parts = folder.parts

                # Verificar si el nombre de la carpeta tiene el formato esperado
                try:
                    graph_type_folder = folder_parts[-2]
                    params = folder_parts[-1].split('_')

                    # Verificar que los valores sean convertibles a enteros
                    graph_type_curr = graph_type_folder
                    n_curr = int(params[1][1:]) if params[1][1:].isdigit() else None
                    k_curr = int(params[2][1:]) if params[2][1:].isdigit() else None
                    q_curr = int(params[3][1:]) if params[3][1:].isdigit() else None
                    perl_vertex_curr = int(params[4][5]) if params[4][5].isdigit() else None

                    # Filtrar según los parámetros proporcionados, ignorando los que sean None
                    if (graph_type is None or graph_type == graph_type_curr) and \
                            (n_range is None or (n_curr is not None and n_range[0] <= n_curr <= n_range[1])) and \
                            (k_range is None or (k_curr is not None and k_range[0] <= k_curr <= k_range[1])) and \
                            (q_range is None or (q_curr is not None and q_range[0] <= q_curr <= q_range[1])) and \
                            (perl_vertex is None or (perl_vertex_curr is not None and perl_vertex == perl_vertex_curr)):
                        # Cargar el CSV
                        df = pd.read_csv(file_path)
                        df['graph_type'] = graph_type_curr
                        df['n'] = n_curr
                        df['k'] = k_curr
                        df['q'] = q_curr
                        df['perl_vertex'] = perl_vertex_curr
                        all_data.append(df)
                except Exception as e:
                    print(f"Error al procesar {file_path}: {e}")

    # Combinar todos los DataFrames
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()  # Retornar DataFrame vacío si no se cargan datos


def plot_data(df):
    if df.empty:
        print("No se encontraron datos con los filtros seleccionados.")
        return

    # Agrupar los datos por valor de 'n' y calcular el valor máximo de cada curva
    n_values = sorted(df['n'].unique())
    max_values = {}

    for n in n_values:
        df_n = df[df['n'] == n]
        grouped = df_n.groupby('prob')['media'].mean()
        max_values[n] = grouped.max()  # Guardamos el valor máximo de cada curva

    # Ordenar las curvas en función de su valor máximo
    sorted_n_values = sorted(max_values, key=max_values.get)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_n_values)))

    lower_bound = np.zeros_like(df.groupby('prob')['media'].mean().values)  # Iniciar límite inferior en 0

    # Graficar en el orden de los valores máximos
    for i, n in enumerate(sorted_n_values):
        df_n = df[df['n'] == n]
        grouped = df_n.groupby('prob')['media'].mean()

        # Graficar la línea
        plt.plot(grouped.index, grouped.values, linestyle='-', color=colors[i], label=f'n={n}')

        # Colorear entre la curva actual y la curva anterior (lower_bound)
        plt.fill_between(grouped.index, lower_bound, grouped.values, color=colors[i], alpha=0.7)

        # Actualizar el límite inferior para la siguiente curva
        lower_bound = grouped.values

    plt.title('Número medio de componentes conexas para diferentes valores de N')
    #plt.title('Número medio de componentes conexas para N = 100')
    plt.xlabel('Probabilidad de percolación (p)')
    plt.ylabel('Media de componentes conexas')
    plt.legend(title='Valor de n')
    plt.grid(True)
    plot_name =  'plot.png'
    plt.savefig(plot_name)
    #plt.xlim(0, 0.2)

    plt.show()

# Ejemplo de uso:
# Filtrar por grafos de tipo 'g2d' y n entre 10 y 40
graph_type = input('graph type (g2d, rgg, g3d, ccg): ')
n_range_a = int(input('n_range a: '))
n_range_b = int(input('n_range b: '))
prob = int(input('number of prob: '))
inst_per_prob = int(input('intancies per prob: '))
perl_vertex = int(input('perlocar vertex (1 o 0): '))
data = load_data(graph_type=graph_type, n_range=(n_range_a, n_range_b), k_range=(inst_per_prob, inst_per_prob),q_range=(prob, prob),perl_vertex=perl_vertex)
plot_data(data)
