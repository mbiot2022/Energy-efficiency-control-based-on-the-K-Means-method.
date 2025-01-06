#!/usr/bin/env python
# coding: utf-8
"""
Programa: eficiencia_energetica_IoT_128_clusters_V38.py
Simulação de uma Rede de Sensores Inteligentes em clusters, baseada no método K-Means e na Lei de Friis, com controle de eficiência energética.
@author: Maurício Brigato
Mestrado em Engenharia Elétrica FESJ ICTS
Unesp Universidade Estadual Paulista "Júlio de Mesquita Filho"
https://www2.unesp.br/
https://www.sorocaba.unesp.br/#!/pos-graduacao/--engenharia-eletrica-local/
date: 25/10/2024"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Definir caminho para salvar os gráficos
save_path = r'C:\Users\mauri\Documents\Mestrado_OFICIAL\DEFESA MESTRADO\eficiencia_128'
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
    cluster_sizes = [4, 8, 16, 32, 64, 128]

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

    # Gráficos de Potências por Cluster
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
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.savefig(os.path.join(save_path, f"Potencias_{num_clusters}Clusters.png"), bbox_inches='tight')
        plt.show()

   

    # Gráfico: Eficiência energética total dos Sensores -> Clusters Heads
    fig, ax = plt.subplots()
    total_sensor_to_ch_efficiency = (df_all_data.groupby('Clusters')['total_transmission_power'].sum() / total_rssf_power) * 100
    ax.plot(total_sensor_to_ch_efficiency.index, total_sensor_to_ch_efficiency.values, marker='o', linestyle='-', color='orange')
    ax.set_title("Eficiência energética total dos Sensores -> Clusters Heads na RSSF")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Eficiência Energética (%)")
    plt.xticks(total_sensor_to_ch_efficiency.index, total_sensor_to_ch_efficiency.index, rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Eficiencia_Energetica_Sensores_Clusters_Heads.png"), bbox_inches='tight')
    plt.show()

     # Gráfico: Correlação Potência CH-Sink / Potência Total (%)
    df_all_data['Relacao_CH_Sink_Potencia_Total'] = (df_all_data['ch_to_sink_power'] / df_all_data['total_transmission_power']) * 100
    df_all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all_data['Relacao_CH_Sink_Potencia_Total'] = df_all_data['Relacao_CH_Sink_Potencia_Total'].fillna(0)
    fig, ax = plt.subplots()
    correlation = df_all_data.groupby('Clusters')['Relacao_CH_Sink_Potencia_Total'].mean()
    ax.plot(correlation.index, correlation.values, marker='o', linestyle='-', color='green')
    ax.set_title("Correlação Potência CH-Sink / Potência Total (%)")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Correlação (%)")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_path, "Relacao_Potencia_CH_Sink_Total.png"))
    plt.show()

    

    # Gráfico: Potência média de Sensores para CHs
    fig, ax = plt.subplots(figsize=(12, 8))
    avg_power_per_sensor = df_all_data.groupby('Clusters')['total_transmission_power'].mean()
    ax.plot(avg_power_per_sensor.index, avg_power_per_sensor.values, marker='o', linestyle='-', color='purple')
    ax.set_title("Potência Média de Sensores para CHs")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência Média de Transmissão (W)")
    plt.savefig(os.path.join(save_path, "Potencia_Media_Sensores_CH.png"), bbox_inches='tight')
    plt.show()

    # Gráfico: Potência Total CH-Sink
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

  

    # Gráfico: Correlação Potência Sensores-CH pela Potência Total da RSSF
    fig, ax = plt.subplots()
    avg_efficiency = (df_all_data.groupby('Clusters')['total_transmission_power'].mean() / df_all_data.groupby('Clusters')['total_transmission_power'].sum()) * 100
    ax.plot(avg_efficiency.index, avg_efficiency.values, marker='o', linestyle='-', color='purple')
    ax.set_title("Correlação Potência Sensores-CH / Potência Total da RSSF")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Correlação (%)")
    plt.xticks(avg_efficiency.index, avg_efficiency.index, rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Eficiencia_Energetica_Media_Sensores.png"), bbox_inches='tight')
    plt.show()
    
 # Gráfico: Eficiência Energética de Clusters na RSSF
    comparative_results = []
    for num_clusters, cluster_data in all_cluster_data.items():
        data = cluster_data['data']
        pot_sensores_e_ch = sum(item['total_transmission_power'] + item['ch_to_sink_power'] for item in data)
        eficiencia_comparativa = (pot_sensores_e_ch / total_rssf_power) * 100
        comparative_results.append((num_clusters, eficiencia_comparativa))

    df_comparativa = pd.DataFrame(comparative_results, columns=['Número de Clusters', 'Eficiência Energética (%)'])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_comparativa['Número de Clusters'], df_comparativa['Eficiência Energética (%)'], marker='o', linestyle='-', color='green')
    ax.set_title("Eficiência Energética de Clusters na RSSF")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Eficiência Energética (%)")
    plt.xticks(df_comparativa['Número de Clusters'], rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(save_path, "Eficiencia_Energetica_Clusters.png"), bbox_inches='tight')
    plt.show()


    # Gráfico 10: Eficiência Energética de Clusters na RSSF
    inverse_energy_consumption = []
    for num_clusters, cluster_data in all_cluster_data.items():
        total_energy = sum(item['total_transmission_power'] + item['ch_to_sink_power'] for item in cluster_data['data'])
        inverse_consumption = 1 / total_energy if total_energy != 0 else float('inf')
        inverse_energy_consumption.append((num_clusters, inverse_consumption))

    df_inverse_consumption = pd.DataFrame(inverse_energy_consumption, columns=['Número de Clusters', '1/Energia consumida pelos clusters (W)'])
    df_inverse_consumption.sort_values(by='Número de Clusters', ascending=True, inplace=True)
    print("Finalizado:", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

if __name__ == "__main__":
    main()
