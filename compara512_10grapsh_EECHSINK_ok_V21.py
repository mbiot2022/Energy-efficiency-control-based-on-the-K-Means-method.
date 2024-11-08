import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from IPython.display import display, Markdown

# Definir caminho para salvar os gráficos
save_path = r'C:\Users\mauri\Documents\Mestrado_OFICIAL\DEFESA MESTRADO\comparaclusters'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Definir uma seed para reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Funções auxiliares e de cálculo
def minkowski_distance(point, center, lamb):
    return sum(abs(point[k] - center[k]) ** lamb for k in range(len(point))) ** (1 / lamb)

def generate_points(tot_sensors, limit):
    return [(random.uniform(0, limit), random.uniform(0, limit)) for _ in range(tot_sensors)]

def calculate_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def friis_transmission_power(pt, gt, gr, lambda_, d):
    return pt * (gt * gr * (lambda_ / (4 * math.pi * d)) ** 2) if d > 0 else float('inf')

def find_cluster_head(cluster, centroid):
    return min(cluster, key=lambda point: calculate_distance(point[0], centroid[0], point[1], centroid[1]))

def simulate_clusters(num_clusters, points, sink_position, lambda_, gt, gr, pt_transceiver):
    centroids = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_clusters)]
    clusters = [[] for _ in range(num_clusters)]

    while True:
        clusters = [[] for _ in range(num_clusters)]
        for x, y in points:
            distances = [minkowski_distance((x, y), centroid, 1) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append((x, y))

        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                mean_x = sum(x for x, y in cluster) / len(cluster)
                mean_y = sum(y for x, y in cluster) / len(cluster)
                new_centroids.append((mean_x, mean_y))
            else:
                new_centroids.append(centroids[i])

        if all(math.isclose(new_centroids[i][0], centroids[i][0], abs_tol=1e-3) and
               math.isclose(new_centroids[i][1], centroids[i][1], abs_tol=1e-3) for i in range(num_clusters)):
            break
        centroids = new_centroids

    cluster_powers = []
    for i, cluster in enumerate(clusters):
        if cluster:
            centroid = centroids[i]
            cluster_head = find_cluster_head(cluster, centroid)
            total_transmission_power = sum(
                friis_transmission_power(pt_transceiver, gt, gr, lambda_, calculate_distance(sensor[0], cluster_head[0], sensor[1], cluster_head[1]))
                for sensor in cluster if calculate_distance(sensor[0], cluster_head[0], sensor[1], cluster_head[1]) > 0)
            d_to_sink = calculate_distance(cluster_head[0], sink_position[0], cluster_head[1], sink_position[1])
            ch_to_sink_power = friis_transmission_power(pt_transceiver, gt, gr, lambda_, d_to_sink)
            cluster_powers.append({
                'cluster_index': i + 1,
                'cluster_head': cluster_head,
                'total_transmission_power': total_transmission_power,
                'ch_to_sink_power': ch_to_sink_power,
                'distance_to_sink': d_to_sink,
                'total_sensors': len(cluster)
            })

    return cluster_powers

# Código principal para gerar gráficos e resultados
def main():
    FREQ = 2.4 * 10**9
    Pt_transceiver = 0.001
    LAMBDA = 299792458 / FREQ
    GT = 1.0
    GR = 1.0
    sink_position = (100, 50)
    totSensors = 128
    points = generate_points(totSensors, 100)
    cluster_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

    all_cluster_data = {}
    total_rssf_power = 0  # Variável para acumular a potência total da RSSF

    for num_clusters in cluster_sizes:
        cluster_data = simulate_clusters(num_clusters, points, sink_position, LAMBDA, GT, GR, Pt_transceiver)
        all_cluster_data[num_clusters] = {'data': cluster_data}

        # Calcular a potência total da RSSF somando as potências de transmissão de sensores e CH-Sink
        total_rssf_power += sum(item['total_transmission_power'] + item['ch_to_sink_power'] for item in cluster_data)

    df_all_data = pd.concat(
        [pd.DataFrame(all_cluster_data[nc]['data']).assign(Clusters=nc) for nc in cluster_sizes],
        ignore_index=True
    )

    # Gráfico 1: Potências por Cluster com valores inteiros no eixo X
    for num_clusters in cluster_sizes:
        subset = df_all_data[df_all_data['Clusters'] == num_clusters]
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = subset['cluster_index'].astype(int)
        ax.bar(indices, subset['total_transmission_power'], label="Total Transmissão", color='blue')
        ax.bar(indices, subset['ch_to_sink_power'], label="CH-Sink", color='red', alpha=0.6)
        ax.set_title(f"Potências por Cluster ({num_clusters} Clusters)")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Potência (W)")
        ax.legend()
        ax.set_xticks(indices)
        plt.savefig(os.path.join(save_path, f"Potencias_{num_clusters}Clusters.png"), bbox_inches='tight')
        plt.show()
        display(Markdown(r"$Potência = \text{Total Transmissão} + \text{CH-Sink}$"))

    # Gráfico 2: Potência média de Sensores para CHs
    fig, ax = plt.subplots(figsize=(12, 8))
    avg_power_per_sensor = df_all_data.groupby('Clusters')['total_transmission_power'].mean()
    ax.plot(avg_power_per_sensor.index, avg_power_per_sensor.values, marker='o', linestyle='-', color='purple')
    ax.set_title("Potência Média de Sensores para CHs")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência Média de Transmissão (W)")
    plt.savefig(os.path.join(save_path, "Potencia_Media_Sensores_CH.png"), bbox_inches='tight')
    plt.show()
    display(Markdown(r"$\text{Potência Média} = \frac{\text{Total Transmissão dos Sensores}}{\text{Número de Sensores}}$"))

    # Potência de CH -> Sink
    fig, ax = plt.subplots()
    ax.bar(df_all_data['Clusters'], df_all_data['ch_to_sink_power'], color='cyan')
    ax.set_title("Potência de CH -> Sink")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência CH-Sink (W)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Potencia_CH_Sink_por_Cluster.png"))
    plt.show()
    display(Markdown(r"$\text{Potência CH-Sink} = \text{Potência do Cluster Head para o Sink}$"))

    # Correlação Potência CH-Sink / Potência Total (%)
    df_all_data['Relacao_CH_Sink_Potencia_Total'] = (df_all_data['ch_to_sink_power'] / df_all_data['total_transmission_power']) * 100
    df_all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all_data['Relacao_CH_Sink_Potencia_Total'] = df_all_data['Relacao_CH_Sink_Potencia_Total'].fillna(0)
    fig, ax = plt.subplots()
    ax.bar(df_all_data['Clusters'], df_all_data['Relacao_CH_Sink_Potencia_Total'], color='blue')
    ax.set_title("Correlação Potência CH-Sink / Potência Total (%)")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Correlação (%)")
    plt.xticks(rotation=45)
    ax.set_ylim(0, df_all_data['Relacao_CH_Sink_Potencia_Total'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Relacao_Potencia_CH_Sink_Total.png"))
    plt.show()
    display(Markdown(r"$\text{Correlação} = \frac{\text{Potência CH-Sink}}{\text{Potência Total}} \times 100\%$"))

    # Consumo energético de Sensores por Clusters na RSSF (%)
    results = []
    for num_clusters, cluster_data in all_cluster_data.items():
        data = cluster_data['data']
        pot_sensores_clusters_i_n = sum(item['total_transmission_power'] for item in data)
        pot_chs_sink = sum(item['ch_to_sink_power'] for item in data)
        pot_tot_rssf = pot_sensores_clusters_i_n + pot_chs_sink

        eficiencia_energetica = (pot_sensores_clusters_i_n / pot_tot_rssf) * 100
        results.append((num_clusters, eficiencia_energetica))

    df_results = pd.DataFrame(results, columns=['Número de Clusters', 'Eficiência Energética (%)'])
    df_results.sort_values(by='Número de Clusters', ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_results['Número de Clusters'], df_results['Eficiência Energética (%)'], marker='o', linestyle='-', color='purple')
    ax.set_title("Consumo energético de Sensores por Clusters na RSSF")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Eficiência Energética (%)")
    plt.xticks(df_results['Número de Clusters'], rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(save_path, "Nova_Eficiencia_Energetica.png"), bbox_inches='tight')
    plt.show()
    display(Markdown(r"$\text{EE\_Sensores} = \frac{\text{Pot\_Total\_Sensores}}{\text{Pot\_Total\_RSSF}} \times 100\%$"))
    display(Markdown("**Este gráfico representa o consumo energético dos sensores em relação à potência total da RSSF.**"))

    # Potência Total CH-Sink
    fig, ax = plt.subplots()
    ch_sink_power_total = df_all_data.groupby('Clusters')['ch_to_sink_power'].sum()
    ax.plot(ch_sink_power_total.index, ch_sink_power_total.values, marker='s', linestyle='--', color='red')
    ax.set_title("Potência Total CH-Sink")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência Total CH-Sink (W)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Potencia_Total_CH_Sink.png"))
    plt.show()
    display(Markdown(r"$\sum \text{Potência CH-Sink de Todos os Clusters}$"))
    display(Markdown("**Este gráfico mostra a potência total de transmissão do CH para o Sink em cada quantidade de clusters.**"))

    # Eficiência energética de Clusters na RSSF
    comparative_results = []
    for num_clusters, cluster_data in all_cluster_data.items():
        data = cluster_data['data']
        pot_sensores_e_ch = sum(item['total_transmission_power'] + item['ch_to_sink_power'] for item in data)
        eficiencia_comparativa = (pot_sensores_e_ch / total_rssf_power) * 100
        comparative_results.append((num_clusters, eficiencia_comparativa))

    df_comparativa = pd.DataFrame(comparative_results, columns=['Número de Clusters', 'Eficiência Energética (%)'])
    df_comparativa.sort_values(by='Número de Clusters', ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_comparativa['Número de Clusters'], df_comparativa['Eficiência Energética (%)'], marker='o', linestyle='-', color='green')
    ax.set_title("Eficiência energética de Clusters na RSSF")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Eficiência Energética (%)")
    plt.xticks(df_comparativa['Número de Clusters'], rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(save_path, "Eficiencia_Energetica_Comparativa.png"), bbox_inches='tight')
    plt.show()
    display(Markdown(r"$\text{EE\_QTD\_CLUSTERS} = \frac{\text{Pot\_Transmissão\_Sensores + Pot\_CH\_Sink}}{\text{Pot\_Tot\_RSSF}} \times 100\%$"))
    display(Markdown("**Este gráfico mostra a eficiência energética em percentual para diferentes quantidades de clusters na RSSF.**"))

    # Eficiência energética média dos Sensores -> Clusters Heads na RSSF (%)
    fig, ax = plt.subplots()
    avg_efficiency = (df_all_data.groupby('Clusters')['total_transmission_power'].mean() / df_all_data.groupby('Clusters')['total_transmission_power'].sum()) * 100
    ax.plot(avg_efficiency.index, avg_efficiency.values, marker='o', linestyle='-', color='purple')
    ax.set_title("Eficiência energética média dos Sensores -> Clusters Heads na RSSF (%)")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Eficiência Energética (%)")
    plt.xticks(avg_efficiency.index, avg_efficiency.index, rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Eficiencia_Energetica_Media_Sensores.png"), bbox_inches='tight')
    plt.show()
    display(Markdown(r"$\text{Eficiência Média} = \frac{\text{Potência Média Sensores-CH}}{\text{Potência Total}} \times 100\%$"))
    display(Markdown("**Este gráfico apresenta a eficiência energética média dos sensores considerando a relação de potência entre sensores e Cluster Heads para diferentes quantidades de clusters.**"))

    # Eficiência energética dos Sensores -> Clusters Heads na RSSF
    fig, ax = plt.subplots()
    total_sensor_to_ch_efficiency = (df_all_data.groupby('Clusters')['total_transmission_power'].sum() / total_rssf_power) * 100
    ax.plot(total_sensor_to_ch_efficiency.index, total_sensor_to_ch_efficiency.values, marker='o', linestyle='-', color='purple')
    ax.set_title("Eficiência energética dos Sensores -> Clusters Heads na RSSF (%)")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Eficiência Energética (%)")
    plt.xticks(total_sensor_to_ch_efficiency.index, total_sensor_to_ch_efficiency.index, rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Eficiencia_Energetica_Total_Sensores_CH.png"), bbox_inches='tight')
    plt.show()
    display(Markdown(r"$\text{Eficiência Total Sensores-CH} = \frac{\text{Potência Total Sensores para CH}}{\text{Potência Total RSSF}} \times 100\%$"))
    display(Markdown("**Este gráfico mostra a eficiência energética total dos sensores para Cluster Heads em relação à potência total da RSSF.**"))

    print("Final:", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

if __name__ == "__main__":
    main()
