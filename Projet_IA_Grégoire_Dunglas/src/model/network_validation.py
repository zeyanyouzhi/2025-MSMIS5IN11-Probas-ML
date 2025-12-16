"""
Validation des propriétés de réseaux sociaux contre données empiriques.
Sources :
- Facebook (Dunbar number)
- Six Degrees (Milgram)
- Scale-free (Barabási)
"""

import networkx as nx
import numpy as np
from model.metrics import compute_R0_dynamic, compute_generation_time, compute_dispersion_k
from collections import Counter
from scipy.stats import linregress
from model.metrics import validate_epidemic_realism

class NetworkValidator:
    """
    Valide le réalisme des réseaux générés en les comparant à des propriétés empiriques.
    """
    
    def __init__(self, agents):
        self.agents = agents
        self.G = self._build_network()

    def _build_network(self):
        """
        Construit un graphe représentant les relations entre agents.
        
        Returns:
            networkx.Graph: Graphe des relations sociales.
        """
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.id)
            for friend in agent.friends:
                G.add_edge(agent.id, friend)
        return G

    def validate_dunbar_number(self):
        """
        Vérifie si le réseau respecte le nombre de Dunbar (~150 relations stables max).
        
        Returns:
            bool: True si le réseau est réaliste, False sinon.
        """
        degrees = [self.G.degree[n] for n in self.G.nodes()]
        avg_degree = sum(degrees) / len(degrees)
        print(f"\n Validation Dunbar :")
        print(f"   - Degré moyen : {avg_degree:.1f} (attendu : 10-30)")
        print(f"   - Degré max : {max(degrees)} (attendu : <150)")
        
        close = sum(1 for d in degrees if d <= 5)
        friends = sum(1 for d in degrees if 5 < d <= 15)
        extended = sum(1 for d in degrees if 15 < d <= 50)
        
        print(f"   - Cercle proche (≤5) : {close/len(degrees)*100:.1f}%")
        print(f"   - Amis (5-15) : {friends/len(degrees)*100:.1f}%")
        print(f"   - Étendu (15-50) : {extended/len(degrees)*100:.1f}%")
        
        return 10 <= avg_degree <= 30

    def validate_six_degrees(self):
        """
        Vérifie si la distance moyenne entre nœuds est proche de 6 (Six Degrees of Separation).
        
        Returns:
            bool: True si la distance moyenne est réaliste, False sinon.
        """
        if not nx.is_connected(self.G):
            print("Le graphe n'est pas connexe.")
            return False
        avg_distance = nx.average_shortest_path_length(self.G)
        print(f"   - Distance moyenne : {avg_distance:.2f} (attendu : ~6)")
        return avg_distance <= 6

    def validate_scale_free(self):
        """
        Vérifie si le réseau suit une distribution de type "scale-free" (loi de puissance).
        
        Returns:
            bool: True si la distribution est réaliste, False sinon.
        """
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        degree_counts = Counter(degrees)
        
        # Filtrer degrés valides (k ≥ 3)
        valid_degrees = {k: v for k, v in degree_counts.items() if k >= 3 and v > 0}
        
        if len(valid_degrees) < 5:
            print("\nValidation Scale-Free : données insuffisantes")
            return False
        
        x = np.log(np.array([float(k) for k in sorted(valid_degrees.keys())], dtype=float))
        y = np.log(np.array([float(valid_degrees[k]) for k in sorted(valid_degrees.keys())], dtype=float))

        x = x.astype(np.float64)
        y = y.astype(np.float64)
        
        # Ajustement linéaire robuste
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            gamma = -float(slope)  # Exposant loi puissance
            r2 = r_value**2
            
            print(f"\nValidation Scale-Free :")
            print(f"   - Exposant γ : {gamma:.2f} (attendu : 2-3)")
            print(f"   - R² ajustement : {r2:.3f} (attendu : >0.7)")
            # Indicateur de qualité de l'ajustement linéaire
            print(f"   - p-value : {p_value:.4f}")
            
            is_valid = (2 <= gamma <= 3.5) and (r2 > 0.7) and (p_value < 0.05)
            
            if is_valid:
                print(f"Réseau scale-free validé")
            else:
                print(f"Distribution s'écarte du scale-free")
            
            return is_valid
        
        except Exception as e:
            print(f"\nValidation Scale-Free : erreur ajustement ({e})")
            return False
        
    def validate_small_world(self):
        """
        Vérifie si le réseau présente des propriétés de "petit monde" (clustering élevé, chemins courts).
        
        Returns:
            bool: True si le réseau a une topologie de petit monde, False sinon.
        """
        if not nx.is_connected(self.G):
            print(" Le graphe n'est pas connexe.")
            return False
        
        clustering_coefficient = nx.average_clustering(self.G)
        average_path_length = nx.average_shortest_path_length(self.G)
        
        # Générer un réseau aléatoire pour comparaison
        random_graph = nx.erdos_renyi_graph(len(self.G.nodes()), nx.density(self.G))
        random_clustering = nx.average_clustering(random_graph)
        random_path_length = nx.average_shortest_path_length(random_graph) if nx.is_connected(random_graph) else float('inf')
        
        sigma = (clustering_coefficient / random_clustering) / (average_path_length / random_path_length)
        
        print(f"\n Validation Small-World :")
        print(f"   - Coefficient de clustering : {clustering_coefficient:.3f} (attendu : >0.3)")
        print(f"   - Longueur de chemin moyenne : {average_path_length:.3f} (attendu : <6)")
        print(f"   - Sigma : {sigma:.3f} (attendu : >1)")
        
        return sigma > 1

    def validate_clustering(self):
        """
        Vérifie si le coefficient de clustering est élevé par rapport à un réseau aléatoire.
        
        Returns:
            bool: True si le clustering est réaliste, False sinon.
        """
        clustering = nx.average_clustering(self.G)
        
        # Réseau aléatoire équivalent pour comparaison
        G_random = nx.erdos_renyi_graph(len(self.G.nodes()), nx.density(self.G))
        clustering_random = nx.average_clustering(G_random)
        
        print(f"\nValidation Clustering :")
        print(f"   - Coefficient clustering : {clustering:.3f} (attendu : 0.3-0.6)")
        print(f"   - Réseau aléatoire : {clustering_random:.3f}")
        if clustering_random > 0:
            print(f"   - Ratio : {clustering/clustering_random:.1f}x (attendu : >20x)")
        else:
            print("- Ratio : N/A (clustering aléatoire nul)")
        
        return 0.3 <= clustering <= 0.6

    def run_all_validations(self):
        """Lance toutes les validations et donne score global"""
        results = {
            'dunbar': self.validate_dunbar_number(),
            'six_degrees': self.validate_six_degrees(),
            'scale_free': self.validate_scale_free(),
            'clustering': self.validate_clustering()
        }
        
        score = sum(results.values()) / len(results) * 100
        print(f" SCORE GLOBAL DE RÉALISME : {score:.0f}%")
        
        if score >= 75:
            print("Réseau hautement réaliste !")
        elif score >= 50:
            print("Réseau partiellement réaliste, améliorations possibles")
        else:
            print("Réseau peu réaliste, modifications nécessaires")
        
        return results
    
    def validate_connections(self, agents, city):
        """Vérifie que tous les modules communiquent correctement"""
        errors = []
        warnings = []
        
        # Test 1 : Tous les agents ont current_location
        for agent in agents:
            if not hasattr(agent, 'current_location'):
                errors.append(f"Agent {agent.id} manque current_location")
        
        # Test 2 : city_env.all_agents est synchronisé
        if city.all_agents != agents:
            warnings.append("city.all_agents désynchronisé avec agents")
        
        # Test 3 : Réseaux sociaux sont bidirectionnels
        non_bidirectional_friends = []
        non_bidirectional_colleagues = []
        
        for agent in agents:
            # Amis
            for friend_id in agent.friends:
                friend = next((a for a in agents if a.id == friend_id), None)
                if friend and agent.id not in friend.friends:
                    non_bidirectional_friends.append((agent.id, friend_id))
            
            # Collègues
            for colleague_id in agent.colleagues:
                colleague = next((a for a in agents if a.id == colleague_id), None)
                if colleague and agent.id not in colleague.colleagues:
                    non_bidirectional_colleagues.append((agent.id, colleague_id))
        
        if non_bidirectional_friends:
            errors.append(f"{len(non_bidirectional_friends)} liens d'amitié non-bidirectionnels")
        
        if non_bidirectional_colleagues:
            warnings.append(f"{len(non_bidirectional_colleagues)} liens de collègues non-bidirectionnels")
        
        # Test 4 : Pas de liens vers soi-même
        self_links = []
        for agent in agents:
            if agent.id in agent.friends or agent.id in agent.colleagues:
                self_links.append(agent.id)
        
        if self_links:
            errors.append(f"{len(self_links)} agents avec liens vers eux-mêmes")
        
        # Affichage
        if errors:
            print("ERREURS CRITIQUES :")
            for e in errors:
                print(f"  - {e}")
        
        if warnings:
            print("AVERTISSEMENTS :")
            for w in warnings:
                print(f"  - {w}")
        
        if not errors and not warnings:
            print("Tous les modules sont correctement connectés")
            print(f"   - {len(agents)} agents")
            print(f"   - {sum(len(a.friends) for a in agents) // 2} liens d'amitié")
            print(f"   - {sum(len(a.colleagues) for a in agents) // 2} liens de collègues")
            print(f"   - {sum(len(a.family) for a in agents) // 2} liens familiaux")

    
    def validate_network_consistency(self, agents):
        """
        Vérifie que tous les modules communiquent correctement
        """
        errors = []
        warnings = []
        
        # Test 1 : Bidirectionnalité des liens
        for agent in agents:
            for friend_id in agent.friends:
                friend = next((a for a in agents if a.id == friend_id), None)
                if friend and agent.id not in friend.friends:
                    errors.append(f"Lien non-bidirectionnel : {agent.id} → {friend_id}")
        
        # Test 2 : Pas de liens vers soi-même
        for agent in agents:
            if agent.id in agent.friends:
                errors.append(f"Agent {agent.id} : auto-lien détecté")
        
        # Test 3 : IDs valides
        agent_ids = {a.id for a in agents}
        for agent in agents:
            invalid = [fid for fid in agent.friends if fid not in agent_ids]
            if invalid:
                errors.append(f"Agent {agent.id} : amis inexistants {invalid}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0
        }
    
    def validate_spatial_consistency(self, agents):
        """
        Vérifie que tous les agents ont des localisations cohérentes.
        
        Returns:
            dict: {
                'errors': [liste d'erreurs],
                'warnings': [liste d'avertissements],
                'is_valid': bool
            }
        """
        errors = []
        warnings = []
        
        valid_locations = {
            'home', 'work', 'restaurant', 'public', 'transports', 
            'gym', 'supermarket', 'hospital'
        }
        
        for agent in agents:
            # Vérifier existence current_location
            if not hasattr(agent, 'current_location'):
                errors.append(f"Agent {agent.id} : manque current_location")
            elif agent.current_location not in valid_locations:
                errors.append(f"Agent {agent.id} : current_location='{agent.current_location}' invalide")
            
            # Vérifier cohérence avec home_quarter
            if not hasattr(agent, 'home_quarter'):
                errors.append(f"Agent {agent.id} : manque home_quarter")
            elif agent.home_quarter not in ['Nord', 'Sud', 'Est', 'Ouest']:
                warnings.append(f"Agent {agent.id} : home_quarter='{agent.home_quarter}' non standard")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0
        }

    def validate_degree_distribution(self, agents):
        """
        Vérifie que la distribution suit une loi de puissance (scale-free).
        
        Returns:
            dict: {
                'gamma': float,  # Exposant loi puissance (attendu : 2-3)
                'R2': float,     # Coefficient ajustement (attendu : >0.7)
                'is_scale_free': bool
            }
        """        
        degrees = [len(a.friends) for a in agents]
        degree_counts = Counter(degrees)
        # Filtrer k >= 3
        valid_degrees = {k: v for k, v in degree_counts.items() if k >= 3 and v > 0}
        if len(valid_degrees) < 5:
            return {'gamma': None, 'R2': None, 'is_scale_free': False}
        # Log-log regression
        x = np.log([float(k) for k in sorted(valid_degrees.keys())])
        y = np.log([float(valid_degrees[k]) for k in sorted(valid_degrees.keys())])
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        gamma = -float(slope)
        r2 = r_value**2
        is_valid = (2 <= gamma <= 3.5) and (r2 > 0.7)
        return {
            'gamma': gamma,
            'R2': r2,
            'p_value': p_value,
            'is_scale_free': is_valid
        }

class EpidemicValidator:
    def __init__(self, epidemic_model, agents):
        self.epidemic = epidemic_model
        self.agents = agents
    
    def validate_all(self):
        """Wrapper consolidé vers metrics.py"""
        return validate_epidemic_realism(self.epidemic, day=10)