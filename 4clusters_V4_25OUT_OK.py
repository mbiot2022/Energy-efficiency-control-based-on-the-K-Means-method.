#!/usr/bin/env python
# coding: utf-8
"""
4clusters_V4_25OUT_OK.py
Simulação de uma Rede de Sensores Inteligentes em clusters, baseada no método K-Means e na Lei de Friis, com controle de eficiência energética.
@author: Maurício Brigato
Mestrado em Engenharia Elétrica FESJ ICTS
Unesp Universidade Estadual Paulista "Júlio de Mesquita Filho"
https://www2.unesp.br/
https://www.sorocaba.unesp.br/#!/pos-graduacao/--engenharia-eletrica-local/
date: 25/10/2024
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

def minkowski_distance(point, center, lamb):
    return sum(abs(point[k] - center[k]) ** lamb for k in range(len(point))) ** (1 / lamb)

def generate_points(tot_sensors, limit):
    return [(random.uniform(0, limit), random.uniform(0, limit)) for _ in range(tot_sensors)]

def calculate_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def friis_transmission_power(pt, gt, gr, lambda_, d):
    if d == 0:
        return float('inf')  # Evita divisão por zero
    return pt * (gt * gr * (lambda_ / (4 * math.pi * d)) ** 2)

def find_cluster_head(cluster, centroid):
    closest_point = min(cluster, key=lambda point: calculate_distance(point[0], centroid[0], point[1], centroid[1]))
    return closest_point

def main():
    print('------------------------4CLUSTERS - INÍCIO: --->')
    inicio = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    print("Início:", inicio)

    # Variáveis de inicialização
    FREQ = 2.4 * 10**9
    PrdBm = -80
    Pt_transceiver = 0.001
    Pr = 10 ** (PrdBm / 10) / 1000
    LAMBDA = 299792458 / FREQ
    GT = 1.0
    GR = 1.0
    area_limit = 100

    # Caminho para salvar os gráficos
    save_path = r'C:\Users\mauri\Documents\Mestrado_OFICIAL\DEFESA MESTRADO'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Coordenada da estação base
    sink_position = (100, 50)

    totSensors = 128
    points = generate_points(totSensors, area_limit)

    # Inicialização dos centroides
    centroids = [(random.uniform(0, area_limit), random.uniform(0, area_limit)) for _ in range(4)]
    print("Centroides iniciais:", centroids)

    # Para armazenar a trajetória de cada centroide
    trajectory = [[] for _ in range(4)]

    var_iteration = int(input('Defina o total de iterations: '))
    iteration = 0
    while iteration < var_iteration:
        iteration += 1
        clusters = [[] for _ in range(4)]

        # Atribuir cada ponto ao cluster mais próximo
        for x, y in points:
            distances = [minkowski_distance((x, y), centroid, 1) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append((x, y))

        # Atualizar os centroides e registrar a trajetória
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                mean_x = sum(x for x, _ in cluster) / len(cluster)
                mean_y = sum(y for _, y in cluster) / len(cluster)
                new_centroids.append((mean_x, mean_y))
                trajectory[i].append((mean_x, mean_y))
            else:
                new_centroids.append((random.uniform(0, area_limit), random.uniform(0, area_limit)))
                trajectory[i].append(centroids[i]) 

        # Verificar se os centroides pararam de mover
        if all(math.isclose(new_centroids[i][0], centroids[i][0], abs_tol=1e-3) and
               math.isclose(new_centroids[i][1], centroids[i][1], abs_tol=1e-3) for i in range(4)):
            break

        centroids = new_centroids

    # Calcular as potências de transmissão dos cluster heads para a estação base
    cluster_powers = []
    for i, cluster in enumerate(clusters):
        if cluster:
            centroid = centroids[i]
            cluster_head = find_cluster_head(cluster, centroid)
            total_transmission_power = 0

            for sensor in cluster:
                d = calculate_distance(sensor[0], cluster_head[0], sensor[1], cluster_head[1])
                if d > 0:
                    transmission_power = friis_transmission_power(Pt_transceiver, GT, GR, LAMBDA, d)
                    total_transmission_power += transmission_power

            # Calcular a potência do cluster head para a estação base
            d_to_sink = calculate_distance(cluster_head[0], sink_position[0], cluster_head[1], sink_position[1])
            ch_to_sink_power = friis_transmission_power(Pt_transceiver, GT, GR, LAMBDA, d_to_sink)

            cluster_powers.append({
                'cluster_index': i + 1,
                'cluster_head': cluster_head,
                'total_transmission_power': total_transmission_power,
                'ch_to_sink_power': ch_to_sink_power,
                'distance_to_sink': d_to_sink,
                'total_sensors': len(cluster)
            })

    # Ordenar os clusters pela potência total gasta em ordem crescente
    cluster_powers_sorted = sorted(cluster_powers, key=lambda x: x['total_transmission_power'])

    # Cores e marcadores consistentes para os clusters
    colors = ['blue', 'green', 'orange', 'purple']
    markers = ['o', 's', 'D', '^']

    # Gráfico 1: Trajetória dos centroides e identificação dos cluster heads
    plt.figure(figsize=(10, 10))
    for i, (cluster, traj) in enumerate(zip(clusters, trajectory)):
        x_vals = [x for x, y in cluster]
        y_vals = [y for x, y in cluster]
        plt.scatter(x_vals, y_vals, color=colors[i], marker=markers[i], label=f'Cluster {i+1}')

        traj_x = [x for x, y in traj]
        traj_y = [y for x, y in traj]
        plt.plot(traj_x, traj_y, '--', color=colors[i], label=f'Trajetória Centroide {i+1}', marker='o')

        # Marca o ponto final do centroide
        plt.scatter(traj_x[-1], traj_y[-1], color='black', marker='x', s=100)

        # Identifica o cluster head
        cluster_head = find_cluster_head(cluster, (traj_x[-1], traj_y[-1]))
        plt.scatter(cluster_head[0], cluster_head[1], color='cyan', edgecolor='black', marker='*', s=150, label=f'Cluster Head {i+1}')
        plt.text(cluster_head[0], cluster_head[1], f'CH{i+1}', fontsize=12, verticalalignment='bottom')

    plt.scatter(sink_position[0], sink_position[1], color='red', marker='s', s=150, label='Base Station (Sink)')
    plt.title("Trajetória dos Centroides e Cluster Heads")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Grafico1_Trajetoria_Centroides.png"))
    plt.show()

    # Gráfico 2: Distribuição dos clusters com círculos de alcance e cluster heads
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, cluster in enumerate(clusters):
        x_vals = [x for x, y in cluster]
        y_vals = [y for x, y in cluster]
        ax.scatter(x_vals, y_vals, color=colors[i], marker=markers[i], label=f'Cluster {i+1}')

        cx, cy = centroids[i]
        ax.scatter(cx, cy, color='black', marker='x', s=100)

        # Identifica o cluster head no segundo gráfico
        cluster_head = cluster_powers[i]['cluster_head']
        ax.scatter(cluster_head[0], cluster_head[1], color='cyan', edgecolor='black', marker='*', s=150, label=f'Cluster Head {i+1}')
        ax.text(cluster_head[0], cluster_head[1], f'CH{i+1}', fontsize=12, verticalalignment='bottom')

    ax.scatter(sink_position[0], sink_position[1], color='red', marker='s', s=150, label='Base Station (Sink)')
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_title("Clusters com Alcance Máximo e Cluster Heads")
    ax.set_xlabel("Eixo X")
    ax.set_ylabel("Eixo Y")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Grafico2_Distribuicao_Clusters.png"))
    plt.show()

    # Gráfico 3: Potência de transmissão dos cluster heads para a base station
    ch_to_sink_powers = [data['ch_to_sink_power'] for data in cluster_powers_sorted]
    plt.figure(figsize=(8, 6))
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, 5), ch_to_sink_powers, color=colors)
    plt.xlabel("Cluster")
    plt.ylabel("Potência de Transmissão para Base Station (W)")
    plt.title("Potência de Transmissão dos Cluster Heads para Base Station")
    plt.xticks(range(1, 5), [f'CH{i+1}' for i in range(4)])
    plt.savefig(os.path.join(save_path, "Grafico3_Potencia_Cluster_Heads.png"))
    plt.show()

    # Exibir os resultados de potência para cada cluster em ordem de menor para maior potência total gasta
    for i, power_data in enumerate(cluster_powers_sorted):
        print(f"Cluster {power_data['cluster_index']}:")
        print(f"  Cluster Head: {power_data['cluster_head']}")
        print(f"  Distância do Cluster Head ao Sink: {power_data['distance_to_sink']:.2f} metros")
        print(f"  Potência Total de Transmissão: {power_data['total_transmission_power']:.15f} W")
        print(f"  Potência de Transmissão do CH para Base Station: {power_data['ch_to_sink_power']:.15f} W")
        
        if i < len(cluster_powers_sorted) - 1:
            next_power_data = cluster_powers_sorted[i + 1]
            print(f"  Comparação com o próximo CH:")
            print(f"    Próximo CH Distância ao Sink: {next_power_data['distance_to_sink']:.2f} metros")
            print(f"    Potência de Transmissão deste CH é {'maior' if power_data['ch_to_sink_power'] > next_power_data['ch_to_sink_power'] else 'menor'} que a do próximo CH")
        print("-" * 50)

    # Criar quadro comparativo dos clusters
    clusters_data = {
        "Cluster": [data['cluster_index'] for data in cluster_powers_sorted],
        "Total Sensores": [data['total_sensors'] for data in cluster_powers_sorted],
        "Distância do CH ao Sink (m)": [data['distance_to_sink'] for data in cluster_powers_sorted],
        "Potência Total Gasta (W)": [data['total_transmission_power'] for data in cluster_powers_sorted]
    }

    # Criar um DataFrame
    df_clusters = pd.DataFrame(clusters_data)

    # Exibir o DataFrame ordenado
    print("\nQuadro Comparativo dos Clusters:")
    print(df_clusters.to_string(index=False))

    # Gráfico 4: Potência Total Gasta por Cluster (em ordem crescente)
    df_clusters_sorted = df_clusters.sort_values("Potência Total Gasta (W)").reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.bar(df_clusters_sorted.index + 1, df_clusters_sorted["Potência Total Gasta (W)"], color='blue', alpha=0.7)
    plt.xlabel("Cluster (Ordenado por Potência Total)")
    plt.ylabel("Potência Total Gasta (W)")
    plt.title("Potência Total Gasta por Cluster (Ordem Crescente)")
    plt.xticks(df_clusters_sorted.index + 1, [f'Cluster {i}' for i in df_clusters_sorted["Cluster"]])
    plt.savefig(os.path.join(save_path, "Grafico4_Potencia_Total_Clusters.png"))
    plt.show()

    print('------------------------4CLUSTERS - FINAL: --->')
    fim = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    print("Início:", inicio)
    print("Final:", fim)

if __name__ == "__main__":
    main()
