import networkx as nx
import matplotlib.pyplot as plt

class NetworkAnalyzer:
    """Analyse des propriétés de réseaux sociaux"""
    
    def __init__(self, agents):
        self.agents = agents
        self.G = self._build_full_network()
    
    def _build_full_network(self):
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.id)
            for friend_id in agent.friends:
                G.add_edge(agent.id, friend_id)
        return G
    
    def compute_small_world_coefficient(self):
        """Coefficient de clustering / longueur de chemin (Primer)"""
        C = nx.average_clustering(self.G)
        L = nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else float('inf')
        # Réseau aléatoire équivalent
        G_random = nx.erdos_renyi_graph(len(self.G.nodes()), nx.density(self.G))
        C_random = nx.average_clustering(G_random)
        L_random = nx.average_shortest_path_length(G_random)
        
        sigma = (C / C_random) / (L / L_random) if L_random > 0 else 0
        return sigma  # > 1 indique propriété "petit monde"
    
    def identify_superspreaders(self, epidemic_model):
        """Identifie les superspreaders selon centralité"""
        betweenness = nx.betweenness_centrality(self.G)
        # Croiser avec nombre de contaminations
        contaminations = {}
        for target, source in epidemic_model.transmission_memory.items():
            contaminations[source] = contaminations.get(source, 0) + 1
        
        # Top 10%
        sorted_agents = sorted(contaminations.items(), key=lambda x: x[1], reverse=True)
        return sorted_agents[:max(1, len(sorted_agents) // 10)]