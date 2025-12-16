"""
Construction de réseaux sociaux réalistes basés sur :
- Modèle de Watts-Strogatz (small-world) : clusters + raccourcis
- Modèle de Barabási-Albert (scale-free) : hubs naturels (influenceurs)
Inspiré de Primer et recherche en épidémiologie
"""

import random
import networkx as nx
from collections import defaultdict
import community as community_louvain
from model.utils import calculate_distance
import math
import numpy as np

class SocialNetworkBuilder:
    """
    Construit des réseaux sociaux réalistes pour la simulation épidémique
    """
    
    def __init__(self, agents):
        self.agents = agents
        self.agent_dict = {a.id: a for a in agents}
        self.bidirectional_prob = 0.65
        self.turnover_liens = 0.07  # Taux de turnover des liens
        self._interaction_history = {}  # Historique des interactions pour la décadence des liens
        
    def build_family_network(self):
        """
        Réseaux familiaux : clusters très denses (déjà fait dans main.py)
        """
        # Déjà implémenté correctement dans main.py (lignes 54-75)
        pass
    
    def build_friends_network_realistic(self, target_avg_degree=20, initial_encounters=15000):
        """
        Génération par collisions sociales réalistes (Vázquez 2007, Nature Physics)
        
        Principe scientifique :
        - Rencontres pairwise répétées avec biais spatial/démographique
        - Accumulation d'affinité → conversion en lien si seuil franchi
        - Heavy-tailed degree distribution émergente
        
        Paramètres calibrés sur Facebook (Ugander et al. 2011) :
        - Degré médian : 99
        - Degré moyen : ~190 (mais on cible 15-25 pour simulation)
        """
        """
          OBSOLÈTE : Utilisée uniquement pour tests
        Le pipeline principal utilise _build_barabasi_albert_base()
        """
        print("AVERTISSEMENT : Utilisation fonction obsolète")
        # ========== PHASE 0 : Réinitialisation ==========
        for agent in self.agents:
            agent.friends = set()
        
        # Mémoire cumulative des rencontres (ne jamais reset entre iterations)
        encounter_count = getattr(self, '_persistent_encounters', {})
        affinity_scores = getattr(self, '_persistent_affinity', {})
        
        print(f"🎲 Génération par {initial_encounters} collisions sociales...")
        
        # ========== PHASE 1 : Collisions avec biais réalistes ==========
        for iteration in range(initial_encounters):
            # === A. SÉLECTION PONDÉRÉE ===
            # 60% voisinage spatial, 30% homophilie démographique, 10% pur hasard
            sampling_mode = random.choices(
                ['spatial', 'demographic', 'random'], 
                weights=[0.6, 0.3, 0.1]
            )[0]
            
            if sampling_mode == 'spatial':
                # Choisir agent1 aléatoire, agent2 dans rayon 20 unités
                agent1 = random.choice(self.agents)
                neighbors = [a for a in self.agents 
                            if a.id != agent1.id 
                            and agent1.home_quarter == a.home_quarter]  # Même quartier
                agent2 = random.choice(neighbors) if neighbors else random.choice(
                    [a for a in self.agents if a.id != agent1.id]
                )
            
            elif sampling_mode == 'demographic':
                # Même tranche d'âge (±10 ans) OU même métier
                agent1 = random.choice(self.agents)
                candidates = [a for a in self.agents 
                            if a.id != agent1.id 
                            and (abs(a.age - agent1.age) <= 10 
                                or a.job == agent1.job)]
                agent2 = random.choice(candidates) if candidates else random.choice(
                    [a for a in self.agents if a.id != agent1.id]
                )
            
            else:  # random
                agent1, agent2 = random.sample(self.agents, 2)
            
            # === B. ACCUMULATION AFFINITÉ ===
            pair = tuple(sorted([agent1.id, agent2.id]))
            encounter_count[pair] = encounter_count.get(pair, 0) + 1
            
            # Calculer affinité incrémentielle (change à chaque rencontre)
            base_affinity = agent1.likeability(agent2)
            
            # Bonus si rencontres fréquentes (effet "mere exposure")
            exposure_bonus = min(0.3, 0.05 * (encounter_count[pair] - 1))
            affinity_scores[pair] = base_affinity + exposure_bonus
        
        # Persister pour prochains appels
        self._persistent_encounters = encounter_count
        self._persistent_affinity = affinity_scores
        
        # ========== PHASE 2 : Conversion affinité → liens ==========
        friendships_formed = 0
        
        for (id1, id2), nb_meetings in encounter_count.items():
            agent1 = self.agent_dict[id1]
            agent2 = self.agent_dict[id2]
            
            # Déjà amis ? Skip
            if id2 in agent1.friends:
                continue
            
            # === RÈGLES DE CONVERSION (calibrées empiriquement) ===
            # 1. Minimum 3 rencontres (Granovetter : weak ties need repetition)
            if nb_meetings < 3:
                continue
            
            # 2. Formule probabiliste réaliste
            affinity = affinity_scores.get((id1, id2), 0)
            
            # Seuil BEAUCOUP plus strict (Dunbar : max 10-30 amis proches)
            threshold = {
                'leader': 0.25,      # ↑ de 0.1 à 0.25
                'anxieux': 0.60,     # ↑ de 0.4 à 0.60
                'suiveur': 0.35,     # ↑ de 0.2 à 0.35
                'rebelle': 0.30,     # ↑ de 0.15 à 0.30
                'calme': 0.40        # ↑ de 0.25 à 0.40
            }.get(agent1.psychology, 0.35)

            exponent = (affinity - threshold) * math.log(nb_meetings + 1) * 0.8  # ÷ 2.5
            friendship_prob = 1 / (1 + math.exp(-exponent))

            # AJOUT : Plafond dur selon Dunbar
            if len(agent1.friends) >= 30:
                friendship_prob *= 0.1  # Réduire drastiquement si déjà 30 amis
            
            if random.random() < friendship_prob:
                agent1.friends.add(agent2.id)
                agent2.friends.add(agent1.id)
                friendships_formed += 1
        
        # ========== PHASE 3 : Vérification cible ==========
        current_avg = sum(len(a.friends) for a in self.agents) / len(self.agents)
        
        print(f" {friendships_formed} amitiés → {current_avg:.1f} amis/agent")
        
        # Si trop bas, relancer avec budget augmenté (+50%)
        if current_avg < target_avg_degree * 0.75:
            print(f"  Degré {current_avg:.1f} < cible {target_avg_degree}, relance avec +100% budget...")
            return self.build_friends_network_realistic(
                target_avg_degree, 
                int(initial_encounters * 2.5)  # Doubler au lieu de 1.5x
            )        
        return current_avg

    def apply_preferential_attachment(self, n_new_links=500):
        """
        Attachement préférentiel (Barabási-Albert) : les agents populaires attirent plus
        
         CORRECTION : Ne créer de liens QUE si pas déjà amis
        
        Principe scientifique (Nature, 1999) :
        - "Les riches deviennent plus riches" (Matthew effect)
        - Probabilité de recevoir un lien ∝ (degré + 1)
        - Crée des hubs naturels selon loi de puissance
        
        Inspirations :
        - Barabási & Albert (1999) : Emergence of scaling in random networks
        - Primer (YouTube) : Visualisation attachement préférentiel
        - Newman (2018) : Networks (livre de référence)
        
        Args:
            n_new_links: Nombre de liens à ajouter
        """
        print(f"⚡ Application attachement préférentiel ({n_new_links} liens)...")
        
        links_added = 0
        attempts = 0
        max_attempts = n_new_links * 10  #  Limite pour éviter boucle infinie

        patience = 0
        max_patience = n_new_links // 2  # Tolérance avant arrêt

        while links_added < n_new_links and attempts < max_attempts:
            if attempts > 0 and attempts % 100 == 0:
                patience += 1
                if patience > max_patience:
                    print(f"  Arrêt anticipé : {links_added}/{n_new_links} liens créés")
                    break
            attempts += 1
            
            # Choisir agent source (tous équiprobables)
            source = random.choice(self.agents)
            
            candidates = [a for a in self.agents 
                        if a.id != source.id 
                        and a.id not in source.friends]

            if not candidates:
                continue            

            degrees = [(a, (len(a.friends) + 1)**3.0) for a in candidates]
            total_degree = sum(d for _, d in degrees)

            for i, (agent, base_score) in enumerate(degrees):
                multiplier = 1.0
                current_degree = len(agent.friends)
                # Super-bonus pour agents déjà très connectés (effet Matthew renforcé)
                if current_degree > 50:
                    multiplier = 25
                elif current_degree > 30:
                    multiplier = 12
                elif current_degree > 20:
                    multiplier = 5
                
                # Bonus psychologie/métier
                if agent.psychology == 'leader':
                    multiplier *= 5
                elif agent.psychology == 'rebelle':
                    multiplier *= 1.8

                if agent.job in ['enseignant', 'commerçant']:
                    multiplier *= 2.5
                elif agent.job == 'médecin':
                    multiplier *= 2.2
                
                degrees[i] = (agent, base_score * multiplier)

            total_degree = sum(d for _, d in degrees)

            if total_degree == 0:
                continue
            
            # Sélection pondérée
            rand = random.random() * total_degree
            cumsum = 0
            target = None
            for agent, degree in degrees:
                cumsum += degree
                if rand <= cumsum:
                    target = agent
                    break
            
            if target:
                # Créer lien potentiellement bidirectionnel
                source.friends.add(target.id)
                if random.random() < self.bidirectional_prob:
                    target.friends.add(source.id)
                links_added += 1
        
        print(f"   {links_added} liens créés par attachement préférentiel ({attempts} tentatives)")

    def enforce_degree_constraints(self, min_degree=5, max_degree=150):
        """
        Applique contraintes de Dunbar (limites cognitives)
        
        Sources scientifiques :
        - Dunbar (1992) : 150 relations stables max
        - Hill & Dunbar (2003) : 5 proches, 15 bons amis, 50 amis, 150 connaissances
        """
        print(f"Application contraintes Dunbar ({min_degree}-{max_degree})...")
        
        removed = 0
        added = 0
        
        for agent in self.agents:
            # === CAS 1 : Trop d'amis (>50) ===
            if len(agent.friends) > max_degree:
                if len(agent.friends) > 200:
                    # Garder top 150 liens les plus forts
                    friend_scores = []
                    for fid in agent.friends:
                        friend = self.agent_dict[fid]
                        affinity = agent.likeability(friend)
                        common = len(agent.friends & friend.friends)
                        score = affinity + 0.3 * common
                        friend_scores.append((fid, score))
                    
                    friend_scores.sort(key=lambda x: x[1], reverse=True)
                    to_keep = set(fid for fid, _ in friend_scores[:150])
                else:
                    # Si <= 200 amis, on garde tous les amis (ne supprime rien)
                    to_keep = set(agent.friends)
                
                # Supprimer les autres (bidirectionnel)
                for fid in list(agent.friends):
                    if fid not in to_keep:
                        agent.friends.discard(fid)
                        self.agent_dict[fid].friends.discard(agent.id)
                        removed += 1
            
            # === CAS 2 : Trop peu d'amis (<min_degree) ===
            elif len(agent.friends) < min_degree:
                # Ajouter des liens avec voisins proches
                # Création de connexions avec des agents géographiquement proches
                candidates = [a for a in self.agents 
                            if a.id != agent.id 
                            and a.id not in agent.friends
                            and calculate_distance(agent.home_quarter, a.home_quarter) == True]
                
                need = min_degree - len(agent.friends)
                selected = random.sample(candidates, min(need, len(candidates)))
                
                for new_friend in selected:
                    agent.friends.add(new_friend.id)
                    new_friend.friends.add(agent.id)
                    added += 1
        
        print(f"   {removed} liens supprimés, {added} liens ajoutés")

    def add_weak_ties(self, rewiring_prob=0.03, max_attempts=10):
        """
        Ajout de liens faibles via rewiring à la Watts–Strogatz.
        Préserve le degré moyen et la connexité globale.
        """

        # Construire le graphe à partir des relations d'amitié actuelles
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.id)
            for friend_id in agent.friends:
                G.add_edge(agent.id, friend_id)
        agents = self.agent_dict

        rewired = 0

        # Liste figée des arêtes (évite modification pendant itération)
        edges = list(G.edges())

        for u, v in edges:
            if random.random() > rewiring_prob:
                continue

            # On choisit aléatoirement une extrémité à rebrancher
            source = u if random.random() < 0.5 else v
            target = v if source == u else u

            # Ne pas casser la connexité
            if G.degree(target) <= 1:
                continue

            # Suppression temporaire
            G.remove_edge(source, target)

            success = False

            for _ in range(max_attempts):
                candidate = random.choice(list(agents.keys()))

                if candidate == source:
                    continue
                if G.has_edge(source, candidate):
                    continue

                # Évite de reconnecter localement (faible lien = longue distance)
                if nx.has_path(G, source, candidate):
                    path_len = nx.shortest_path_length(G, source, candidate)
                    if path_len < 3:
                        continue

                # Ajout du nouveau lien
                G.add_edge(source, candidate)
                success = True
                rewired += 1
                break

            # Échec → on restaure l’arête initiale
            if not success:
                G.add_edge(source, target)

        print(f"Ajout de liens faibles (Watts-Strogatz) : {rewired} liens rewirés")

    def add_weak_ties_2(self, rewiring_prob=0.05):
        """
        Ajoute des liens faibles (weak ties) selon modèle Watts-Strogatz
        
        Principe scientifique (Granovetter, 1973 + Watts & Strogatz, 1998) :
        - Les liens faibles (connaissance lointaine) créent des raccourcis
        - Permettent propriété "small-world" : clustering élevé + chemin court
        - 10% de rewiring = optimal pour small-world (Watts & Strogatz)
        
        Inspirations :
        - Granovetter (1973) : "The Strength of Weak Ties"
        - Watts & Strogatz (1998) : "Collective dynamics of small-world networks" (Nature)
        - Fouloscopie : "Les réseaux petit monde expliqués"
        
        Args:
            rewiring_prob: Probabilité de créer un raccourci (0.05-0.15 optimal)
        """
        print(f"Ajout de liens faibles (rewiring={rewiring_prob})...")
        
        rewired = 0
        for agent in self.agents:
            # Nombre de raccourcis = 10% des amis
            n_shortcuts = max(1, int(len(agent.friends) * rewiring_prob)) # Optimal à 0.05
            
            for _ in range(n_shortcuts):
                # Agents éloignés spatialement ET socialement
                distant = [
                    a for a in self.agents
                    if a.id != agent.id
                    and a.id not in agent.friends
                    and agent.home_quarter != a.home_quarter
                    and len(agent.friends & a.friends) == 0
                    and agent.comm_id != a.comm_id
                ]
                
                if distant:
                    new_friend = random.choice(distant)
                    agent.friends.add(new_friend.id)
                    new_friend.friends.add(agent.id)
                    rewired += 1        
        print(f"{rewired} liens rewirés (raccourcis longue distance)")
                        
    def apply_triadic_closure(self, closure_rate=0.25, max_iterations=5):
        """
        Fermeture triadique itérative (Rapoport 1953, Granovetter 1973)
        
        Principe : "L'ami de mon ami devient mon ami"
        Réalité empirique : 25-40% des amitiés se forment via amis communs
        
        Paramètres calibrés sur Facebook (Ugander 2011) :
        - ~30% des nouvelles amitiés = triadic closure
        """
        print(f"Triadic closure itérative (taux={closure_rate}, {max_iterations} passes)...")
        
        total_new = 0
        
        for iteration in range(max_iterations):
            new_this_round = 0
            
            for agent in random.sample(self.agents, len(self.agents)):  # Ordre aléatoire
                # Collecter amis d'amis
                friends_of_friends = set()
                for friend_id in agent.friends:
                    friend = self.agent_dict[friend_id]
                    friends_of_friends.update(friend.friends)
                
                # Retirer soi-même et amis existants
                friends_of_friends.discard(agent.id)
                friends_of_friends -= agent.friends
                
                if not friends_of_friends:
                    continue
                
                # Trier par nombre d'amis communs (priorité aux liens forts)
                fof_ranked = []
                for fof_id in friends_of_friends:
                    fof = self.agent_dict[fof_id]
                    common = len(agent.friends & fof.friends)
                    fof_ranked.append((fof, common))
                
                fof_ranked.sort(key=lambda x: x[1], reverse=True)
                
                # Fermeture proportionnelle aux amis communs
                for fof, common_count in fof_ranked[:8]:
                    if common_count < 3:
                        continue

                    closure_prob = closure_rate * (1 + 0.15 * common_count)
                    closure_prob *= max(0.1, 1 - len(agent.friends)/30)  # Décroissance si > 20 amis

                    if len(agent.friends) >= 15:
                        closure_prob *= 0.03
                    
                    if random.random() < closure_prob:
                        agent.friends.add(fof.id)
                        fof.friends.add(agent.id)
                        new_this_round += 1
                        
                        # Stop si degré devient trop élevé (réalisme Dunbar)
                        if len(agent.friends) > 50:
                            break
            
            total_new += new_this_round
            print(f"  Pass {iteration+1} : +{new_this_round} liens")
            
            # Convergence : si <5% de nouveaux liens, arrêter
            if new_this_round < 0.05 * len(self.agents):
                print(f"Convergence atteinte")
                break
        
        print(f"Total : {total_new} nouvelles amitiés par triadic closure")

    def build_colleagues_network_scale_free(self, m=3):
        """
        Réseau de collègues selon modèle de Barabási-Albert (scale-free)
        
        Principe : Attachement préférentiel (les "populaires" attirent plus)
        → Crée des hubs naturels (managers, leaders d'équipe)
        
        Paramètre :
        - m : nombre de connexions pour chaque nouvel agent
        """
        # Grouper par lieu de travail
        workplaces = {}
        for agent in self.agents:
            key = (agent.work_quarter, agent.job)
            workplaces.setdefault(key, []).append(agent)
        
        # Pour chaque lieu, créer un réseau scale-free
        for workplace_agents in workplaces.values():
            if len(workplace_agents) < 2:
                continue
            
            # Initialiser avec m+1 agents complètement connectés
            initial_size = min(m + 1, len(workplace_agents))
            for i in range(initial_size):
                workplace_agents[i].colleagues = set(
                    a.id for a in workplace_agents[:initial_size] if a.id != workplace_agents[i].id
                )
            
            # Ajouter les autres agents avec attachement préférentiel
            for agent in workplace_agents[initial_size:]:
                agent.colleagues = set()
                
                # Calculer probabilités (proportionnelles au degré)
                existing_agents = workplace_agents[:workplace_agents.index(agent)]
                degrees = [len(a.colleagues) + 1 for a in existing_agents]  # +1 pour éviter division par 0
                total_degree = sum(degrees)
                
                # Sélectionner m collègues selon attachement préférentiel
                selected = []
                for _ in range(min(m, len(existing_agents))):
                    rand = random.random() * total_degree
                    cumsum = 0
                    for i, a in enumerate(existing_agents):
                        cumsum += degrees[i]
                        if rand <= cumsum and a not in selected:
                            selected.append(a)
                            break
                
                # Créer les liens bidirectionnels
                for colleague in selected:
                    agent.colleagues.add(colleague.id)
                    colleague.colleagues.add(agent.id)
    
    def apply_homophily_filtering(self):
        """
        Filtre les liens faibles selon homophilie (Primer : "birds of a feather")
        Les gens restent amis avec ceux qui leur ressemblent
        """
        for agent in self.agents:
            # Garder seulement les amis avec affinité > seuil variable
            filtered_friends = set()
            for friend_id in agent.friends:
                friend = self.agent_dict[friend_id]
                affinity = agent.likeability(friend)
                
                # Seuil adaptatif selon psychologie
                threshold = {
                    'leader': 0.2,    # Leaders = réseau large
                    'suiveur': 0.4,   # Suiveurs = sélectifs
                    'anxieux': 0.5,   # Anxieux = cercle restreint
                    'rebelle': 0.3,   # Rebelles = diversité
                    'calme': 0.35
                }.get(agent.psychology, 0.3)
                
                if affinity > threshold:
                    filtered_friends.add(friend_id)
            
            agent.friends = filtered_friends    

    def identify_network_hubs(self):
        """
        Identifie les hubs (super-connecteurs) dans chaque réseau
        Utile pour stratégies de vaccination ciblée
        """
        # Construire graphe NetworkX pour analyse
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.id)
            for friend_id in agent.friends:
                G.add_edge(agent.id, friend_id)
            for col_id in agent.colleagues:
                G.add_edge(agent.id, col_id)
        
        # Centralité de degré (nombre de connexions)
        degree_centrality = nx.degree_centrality(G)
        
        # Centralité d'intermédiarité (nœuds "pont")
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Top 10% de chaque métrique
        hubs_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        hubs_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        
        n_hubs = max(1, len(self.agents) // 10)
        
        return {
            'degree_hubs': [agent_id for agent_id, _ in hubs_degree[:n_hubs]],
            'betweenness_hubs': [agent_id for agent_id, _ in hubs_betweenness[:n_hubs]],
            'centrality_scores': {'degree': degree_centrality, 'betweenness': betweenness_centrality}
        }
    
    def compute_network_metrics(self):
        """
        Calcule les métriques de réseau pour validation scientifique
        """
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.id)
            for friend_id in agent.friends:
                G.add_edge(agent.id, friend_id)
        
        if not nx.is_connected(G):
            # Prendre la plus grande composante connexe
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        metrics = {
            'clustering_coefficient': nx.average_clustering(G),
            'average_path_length': nx.average_shortest_path_length(G),
            'diameter': nx.diameter(G),
            'density': nx.density(G),
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        }
        
        # Calculer le coefficient "small-world" (σ)
        # σ > 1 indique propriété "petit monde"
        G_random = nx.erdos_renyi_graph(G.number_of_nodes(), nx.density(G))
        C_random = nx.average_clustering(G_random)
        L_random = nx.average_shortest_path_length(G_random) if nx.is_connected(G_random) else float('inf')
        
        if C_random <= 0 or L_random <= 0 or np.isinf(C_random) or np.isinf(L_random):
            metrics['small_world_sigma'] = None
        else:
            metrics['small_world_sigma'] = (metrics['clustering_coefficient'] / C_random) / (metrics['average_path_length'] / L_random)
        
        return metrics    

    def detect_communities(self):
        """
        Détection de communautés (Louvain algorithm)
        Inspiré Fouloscopie : groupes sociaux naturels
        
        Returns:
            dict: {community_id: [agent_ids]}
        """
        import community as community_louvain  # pip install python-louvain
        
        # Construire graphe complet
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.id)
            for friend_id in agent.friends:
                G.add_edge(agent.id, friend_id, weight=1.5)  # Amis = lien fort
            for col_id in agent.colleagues:
                G.add_edge(agent.id, col_id, weight=1.0)  # Collègues = lien moyen
        
        # Détection de communautés
        partition = community_louvain.best_partition(G)
        
        # Organiser par communauté
        communities = {}
        for agent_id, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(agent_id)
        
        return communities

    def merge_small_communities(self, communities, min_size=8):
        """
        Fusionne communautés <8 membres avec voisines
        Cible : 10-20 communautés de 10-20 membres (Dunbar)
        """
        print(f"Fusion communautés <{min_size} membres...")
        
        to_merge = [cid for cid, members in communities.items() if len(members) < min_size]
        
        for cid in to_merge:
            members = communities[cid]
            
            # Trouver communauté voisine la plus connectée
            neighbor_counts = {}
            for agent_id in members:
                agent = self.agent_dict[agent_id]
                for friend_id in agent.friends:
                    for other_cid, other_members in communities.items():
                        if other_cid != cid and friend_id in other_members:
                            neighbor_counts[other_cid] = neighbor_counts.get(other_cid, 0) + 1
            
            if neighbor_counts:
                best_cid = max(neighbor_counts.items(), key=lambda x: x[1])[0]
                communities[best_cid].extend(members)
                print(f"Communauté {cid} ({len(members)} membres) → {best_cid}")
            
            del communities[cid]
        
        print(f"{len(to_merge)} communautés fusionnées")
        return communities

    def assign_community_leaders(self, communities):
        """
        Assigne un leader par communauté (influenceur local)
        Critères : centralité + psychologie leader
        """
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.id)
            for friend_id in agent.friends:
                G.add_edge(agent.id, friend_id)
        
        betweenness = nx.betweenness_centrality(G)
        
        community_leaders = {}
        for comm_id, members in communities.items():
            # Trier par centralité * facteur psychologie
            candidates = []
            for agent_id in members:
                agent = self.agent_dict[agent_id]
                score = betweenness.get(agent_id, 0)
                # Bonus si psychologie leader
                if agent.psychology == 'leader':
                    score *= 2.0
                # Bonus si métier influent
                if agent.job in ['enseignant', 'médecin', 'cadre']:
                    score *= 1.5
                candidates.append((agent_id, score))
            
            # Leader = meilleur score
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates:
                community_leaders[comm_id] = candidates[0][0]
        
        return community_leaders
    
    def update_friendships_over_time(self, interaction_matrix, days_elapsed=1):
        """
        Évolution dynamique des amitiés (Primer : feedback loops)
        
        Principe scientifique (Burt 2004, Kossinets & Watts 2006) :
        - 5-10 interactions → lien faible
        - 15-20 interactions → ami
        - <2 interactions/mois → rupture
        
        Args:
            interaction_matrix : {(id1, id2): count}
            days_elapsed : nombre de jours depuis dernier appel
        """
        new_friendships = 0
        broken_friendships = 0
        
        # === CRÉATION DE LIENS ===
        for key, count in interaction_matrix.items():
            agent_id1, agent_id2 = key[:2]  # On ignore le lieu pour l’amitié
            agent1 = self.agent_dict[agent_id1]
            agent2 = self.agent_dict[agent_id2]
            
            # Déjà amis ? Skip
            if agent_id2 in agent1.friends:
                continue
            
            # Seuil adaptatif selon fréquence
            interactions_per_day = count / days_elapsed
            
            # Formule réaliste (Kossinets & Watts 2006) :
            # P(lien) ∝ interactions^1.5 × affinité
            affinity = agent1.likeability(agent2)
            
            # Seuil : 2 interactions/jour pendant 7 jours = quasi-certain
            threshold_interactions = 10  # Total sur période
            
            if count >= threshold_interactions:
                friendship_prob = min(0.9, 
                                    (count / threshold_interactions)**1.2 * affinity)
                
                if random.random() < friendship_prob:
                    agent1.friends.add(agent_id2)
                    agent2.friends.add(agent_id1)
                    new_friendships += 1
                    #print(f"Nouvelle amitié : {agent_id1} ↔ {agent_id2} "
                    #    f"({count} interactions en {days_elapsed}j)")
        
        # RUPTURE DE LIENS (manque d'entretien)
        for agent in self.agents:
            for friend_id in list(agent.friends):
                pair = tuple(sorted([agent.id, friend_id]))
                recent_interactions = interaction_matrix.get(pair, 0)
                
                if recent_interactions < 0.4:
                    rupture_prob = 0.001                    
                    # Bonus stabilité famille/collègues
                    friend = self.agent_dict[friend_id]
                    if friend_id in agent.nuclear_family:
                        rupture_prob = 0.0001  # Quasi-stable
                    elif friend_id in agent.colleagues:
                        rupture_prob = 0.003  # Turnover professionnel
                    else:
                        # Amis "purs" : bonus si forte affinité
                        friend = self.agent_dict[friend_id]
                        affinity = agent.likeability(friend)
                        if affinity > 0.6:
                            rupture_prob *= 0.3

                    if random.random() < rupture_prob:
                        agent.friends.discard(friend_id)
                        friend.friends.discard(agent.id)
                        broken_friendships += 1
                        #print(f"Amitié perdue : {agent.id} ↔ {friend_id} "
                        #    f"(manque d'interactions)")

        # NOUVEAU : Mécanisme homéostatique (Dunbar)
        target_degree = 15
        for agent in self.agents:
            current_degree = len(agent.friends)
            
            if current_degree < target_degree * 0.7:  # Si < 10.5 amis
                deficit = int((target_degree - current_degree) * 0.3)  # Compenser 30%
                
                # Chercher amis d'amis
                friends_of_friends = set()
                for fid in agent.friends:
                    friend = self.agent_dict.get(fid)
                    if friend:
                        friends_of_friends.update(friend.friends)
                
                friends_of_friends.discard(agent.id)
                friends_of_friends -= agent.friends
                
                # Ajouter liens faibles
                candidates = list(friends_of_friends)[:deficit]
                for cid in candidates:
                    agent.friends.add(cid)
                    self.agent_dict[cid].friends.add(agent.id)

        print(f"{sum(1 for a in self.agents if len(a.friends) < target_degree * 0.7)} agents compensés")
        print(f"Bilan : +{new_friendships} amitiés, -{broken_friendships} pertes")


    def strengthen_family_bonds(self):
        """
        Renforce les liens familiaux (toujours les plus forts)
        Fouloscopie : famille = refuge en cas de crise
        """
        for agent in self.agents:
            # Famille cohabitante = lien maximal
            for fam_id in agent.family:
                fam = self.agent_dict.get(fam_id)
                if fam and fam.home == agent.home:
                    # S'assurer qu'ils sont aussi "amis" (double lien)
                    if fam_id not in agent.friends:
                        agent.friends.add(fam_id)
                        fam.friends.add(agent.id)

    def fix_bidirectional_links(self):
        """
        CORRECTION CRITIQUE : Force la bidirectionnalité des liens d'amitié
        
        Problème détecté : "26 liens d'amitié non-bidirectionnels"
        Cause : build_friends_network_realistic() peut créer liens unilatéraux
        
        Solution : Parcourir tous les agents et synchroniser les liens
        """
        print("Correction des liens non-bidirectionnels...")
        
        fixed_count = 0
        
        # NOUVELLE APPROCHE : Double passe
        # Passe 1 : Nettoyer les liens invalides
        for agent in self.agents:
            invalid_friends = set()
            for friend_id in agent.friends:
                if friend_id not in self.agent_dict:
                    invalid_friends.add(friend_id)
            agent.friends -= invalid_friends
            fixed_count += len(invalid_friends)
        
        # Passe 2 : Forcer la bidirectionnalité
        for agent in self.agents:
            for friend_id in list(agent.friends):
                friend = self.agent_dict.get(friend_id)
                if friend and agent.id not in friend.friends:
                    friend.friends.add(agent.id)
                    fixed_count += 1
        
        print(f"{fixed_count} liens corrigés (bidirectionnalité garantie)")
        
        # Vérification finale STRICTE
        errors = []
        for agent in self.agents:
            for friend_id in agent.friends:
                friend = self.agent_dict.get(friend_id)
                if not friend:
                    errors.append(f"Agent {agent.id} → ami inexistant {friend_id}")
                elif agent.id not in friend.friends:
                    errors.append(f"Lien non-réciproque : {agent.id} → {friend_id}")
        
        if errors:
            print(f"{len(errors)} erreurs persistantes :")
            for e in errors[:5]:  # Afficher 5 premières
                print(f"     - {e}")
            raise RuntimeError("Liens non-bidirectionnels détectés après correction")
        else:
            print("Tous les liens sont bidirectionnels")

    def remove_non_bidirectional_links(self):
        """
        Supprime les liens d'amitié non-bidirectionnels (au lieu de les corriger).
        Après exécution, seuls les liens réciproques sont conservés.
        """
        print("Suppression des liens non-bidirectionnels...")
        removed_count = 0
        for agent in self.agents:
            to_remove = set()
            for friend_id in agent.friends:
                friend = self.agent_dict.get(friend_id)
                if not friend or agent.id not in friend.friends:
                    to_remove.add(friend_id)
            if to_remove:
                agent.friends -= to_remove
                removed_count += len(to_remove)
        print(f"{removed_count} liens non-bidirectionnels supprimés")
        

    def adjust_community_sizes(self, communities, taille_min=5, taille_max=20):
        # Fusionner les petites communautés avec la plus proche (en termes de liens)
        to_merge = [cid for cid, members in communities.items() if len(members) < taille_min]
        for cid in to_merge:
            members = communities[cid]
            # Chercher la communauté voisine la plus connectée
            neighbor_counts = {}
            for agent_id in members:
                agent = self.agent_dict[agent_id]
                for friend_id in agent.friends:
                    for other_cid, other_members in communities.items():
                        if other_cid != cid and friend_id in other_members:
                            neighbor_counts[other_cid] = neighbor_counts.get(other_cid, 0) + 1
            if neighbor_counts:
                best_cid = max(neighbor_counts.items(), key=lambda x: x[1])[0]
                communities[best_cid].extend(members)
            del communities[cid]

        # Diviser les trop grandes communautés
        new_communities = {}
        next_id = max(communities.keys()) + 1 if communities else 0
        for cid, members in list(communities.items()):
            if len(members) > taille_max:
                # Re-détecter des sous-communautés avec Louvain sur le sous-graphe
                subG = nx.Graph()
                subG.add_nodes_from(members)
                for agent_id in members:
                    agent = self.agent_dict[agent_id]
                    for friend_id in agent.friends:
                        if friend_id in members:
                            subG.add_edge(agent_id, friend_id)
                sub_partition = community_louvain.best_partition(subG)
                sub_comms = {}
                for node, sub_cid in sub_partition.items():
                    sub_comms.setdefault(sub_cid, []).append(node)
                for sub in sub_comms.values():
                    new_communities[next_id] = sub
                    next_id += 1
                del communities[cid]
        communities.update(new_communities)
        return communities
    
    def remove_random_links(self, turnover_rate=None):
        """
        Supprime des liens aléatoirement, surtout entre agents en désaccord (ex: anxieux/calme)
        """
        if turnover_rate is None:
            turnover_rate = self.turnover_liens
        for agent in self.agents:
            if agent.friends and random.random() < turnover_rate:
                friend_id = random.choice(list(agent.friends))
                friend = self.agent_dict.get(friend_id)
                if friend and hasattr(agent, 'state') and hasattr(friend, 'state') and agent.state != friend.state:
                    agent.friends.discard(friend_id)
                    friend.friends.discard(agent.id)

    def build_family_links(self):
        """
        Assigne 3-5 membres de famille aléatoires à chaque agent
        """
        for agent in self.agents:
            family_size = random.randint(3, 5)
            while len(agent.family) < family_size:
                family_member = random.choice(self.agents)
                if family_member != agent and family_member.id not in agent.family:
                    agent.family.add(family_member.id)
                    family_member.family.add(agent.id)

    def build_colleague_links(self):
        """
        Assigne 5-10 collègues professionnels aléatoires à chaque agent
        """
        for agent in self.agents:
            colleague_size = random.randint(5, 10)
            while len(agent.colleagues) < colleague_size:
                colleague = random.choice(self.agents)
                if colleague != agent and colleague.id not in agent.colleagues:
                    agent.colleagues.add(colleague.id)
                    colleague.colleagues.add(agent.id)

    def monitor_network_quality(self):
        """
        Surveille qualité en temps réel (alertes précoces)
        """
        degrees = [len(a.friends) for a in self.agents]
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)
        
        # Alertes
        warnings = []
        
        if avg_degree < 10:
            warnings.append(f"Degré moyen {avg_degree:.1f} < 10 (Dunbar minimum)")
        
        if max_degree < 30:
            warnings.append(f"Pas de hubs (max={max_degree}, attendu >50)")
        
        if min_degree == 0:
            isolated = sum(1 for d in degrees if d == 0)
            warnings.append(f"{isolated} agents isolés")
        
        # Coefficient de variation (variance/moyenne)
        cv = np.std(degrees) / avg_degree if avg_degree > 0 else 0
        if cv < 0.5:
            warnings.append(f"Distribution trop homogène (CV={cv:.2f}, attendu >0.8)")
        
        if warnings:
            print("\nALERTES QUALITÉ RÉSEAU :")
            for w in warnings:
                print(f"  {w}")
        else:
            print("\nRéseau conforme (surveillance continue)")
        
        return len(warnings) == 0

    def _build_barabasi_albert_base(self, m=3):
        """
        Construction Barabási-Albert pure (attachement préférentiel sans bruit)
        
        Args:
            m: Nombre de liens par nouvel agent (standard: 2-3)
        
        Source: Barabási & Albert (1999), Science
        """
        print(f"Barabási-Albert (m={m})...")
        
        # Réinitialiser TOUS les liens
        for agent in self.agents:
            agent.friends = set()
        
        # Phase 1 : Initialiser avec m0 agents complètement connectés
        m0 = max(m + 1, 5)
        initial_agents = self.agents[:m0]
        
        for i, agent in enumerate(initial_agents):
            for other in initial_agents:
                if other.id != agent.id:
                    agent.friends.add(other.id)
        
        print(f"Initialisation : {m0} agents complètement connectés")
        
        # Phase 2 : Ajouter agents un par un avec attachement préférentiel
        links_added = 0
        
        for new_agent in self.agents[m0:]:
            # Calculer probabilités (proportionnelles au degré)
            existing_agents = self.agents[:self.agents.index(new_agent)]
            degrees = [(a, len(a.friends) + 1) for a in existing_agents]
            total_degree = sum(d for _, d in degrees)
            
            if total_degree == 0:
                continue
            
            # Sélectionner m agents selon attachement préférentiel
            selected = []
            for _ in range(m):
                rand = random.random() * total_degree
                cumsum = 0
                
                for agent, degree in degrees:
                    cumsum += degree
                    if rand <= cumsum and agent not in selected:
                        selected.append(agent)
                        break
            
            # Créer les liens (bidirectionnels)
            for target in selected:
                new_agent.friends.add(target.id)
                target.friends.add(new_agent.id)
                links_added += 1
        
        avg_degree = sum(len(a.friends) for a in self.agents) / len(self.agents)
        print(f"{links_added} liens créés, degré moyen : {avg_degree:.1f}")

    def _prune_weak_links(self, target_avg_degree):
        """
        Élagage forcé des liens faibles pour atteindre la cible
        
        Stratégie : Supprimer les liens à faible affinité jusqu'à atteindre le degré cible
        """
        current_avg = sum(len(a.friends) for a in self.agents) / len(self.agents)
        target_total_links = int(target_avg_degree * len(self.agents) / 2)
        current_total_links = sum(len(a.friends) for a in self.agents) // 2
        links_to_remove = current_total_links - target_total_links
        
        if links_to_remove <= 0:
            return
        
        print(f"Élagage : supprimer {links_to_remove} liens...")
        
        # Collecter tous les liens avec leur score d'affinité
        all_links = []
        for agent in self.agents:
            for friend_id in agent.friends:
                if agent.id < friend_id:  # Éviter doublons
                    friend = self.agent_dict[friend_id]
                    affinity = agent.likeability(friend)
                    all_links.append((agent.id, friend_id, affinity))
        
        # Trier par affinité croissante (supprimer les plus faibles)
        all_links.sort(key=lambda x: x[2])
        
        # Supprimer les N liens les plus faibles
        removed = 0
        for agent_id, friend_id, _ in all_links[:links_to_remove]:
            agent = self.agent_dict[agent_id]
            friend = self.agent_dict[friend_id]
            
            # Ne pas descendre sous 5 amis (Dunbar minimum)
            if len(agent.friends) > 5 and len(friend.friends) > 5:
                agent.friends.discard(friend_id)
                friend.friends.discard(agent_id)
                removed += 1
            
            if removed >= links_to_remove:
                break
        
        final_avg = sum(len(a.friends) for a in self.agents) / len(self.agents)
        print(f"{removed} liens supprimés → degré moyen : {final_avg:.1f}")

    def build_complete_network(self, target_avg_degree=20, enable_validation=True):
        """
        Pipeline refondu pour obtenir réseau scale-free réaliste
        
        Stratégie : Créer MOINS de liens, mais mieux distribués
        """
        print("CONSTRUCTION RÉSEAU SCALE-FREE (Pipeline optimisé)")
        
        # === PHASE 1 : BASE MINIMALE (Barabási-Albert pur) ===
        print("\n[1/7] Attachement préférentiel initial (base scale-free)...")
        # COMMENCER avec attachement préférentiel, PAS collisions
        # m = 2-3 liens par nouvel agent (Barabási-Albert standard)
        m = max(2, target_avg_degree // 6)
        self._build_barabasi_albert_base(m=m)
        
        # === PHASE 2 : CLUSTERING LOCAL (Triadic closure limité) ===
        print("\n[2/7] Triadic closure (clustering local)...")
        # UNE SEULE passe, taux réduit
        self.apply_triadic_closure(closure_rate=0.15, max_iterations=3)
        
        # === PHASE 3 : CONTRAINTES DUNBAR STRICTES ===
        print("\n[3/7] Application contraintes Dunbar...")
        self.enforce_degree_constraints(min_degree=5, max_degree=50)
        
        # === PHASE 4 : WEAK TIES MINIMALISTES ===
        print("\n[4/7] Weak ties (raccourcis minimalistes)...")
        self.add_weak_ties(rewiring_prob=0.08)
        self.apply_triadic_closure(closure_rate=0.05, max_iterations=3)
        
        # === PHASE 5 : NETTOYAGE FINAL ===
        print("\n[5/7] Nettoyage et validation...")
        self.fix_bidirectional_links()

        # === PHASE 6 : GÉNÉRATION CONNAISSANCES ===
        print("\n[6/7] Génération connaissances...")
        for agent in self.agents:
            # Amis d'amis = connaissances potentielles
            fof = set()
            for fid in agent.friends:
                fof.update(self.agent_dict[fid].friends)
            
            fof.discard(agent.id)
            fof -= agent.friends
            
            # Garder 30-50 connaissances
            n_connaissances = random.randint(30, 50)
            agent.connaissances = set(random.sample(list(fof), min(n_connaissances, len(fof))))
            
            #Force du lien
            for aid in agent.friends:
                agent.tie_strength[aid] = random.uniform(0.7, 1.0)  # Amis proches
            for aid in agent.connaissances:
                agent.tie_strength[aid] = random.uniform(0.1, 0.4)  # Liens faibles

        # === PHASE 7 : CONTRAINTES DUNBAR STRICTES ===
        print("\n[7/7] Application contraintes Dunbar...")
        self.enforce_degree_constraints(min_degree=5, max_degree=150)

        # Vérification finale
        final_avg = sum(len(a.friends) for a in self.agents) / len(self.agents)
        print(f"\nDegré moyen final : {final_avg:.1f}")
        
        if final_avg > target_avg_degree * 1.3:
            print(f"ALERTE : Degré {final_avg:.1f} > cible {target_avg_degree}")
            print("→ Élagage forcé des liens faibles...")
            self._prune_weak_links(target_avg_degree)
        
        return self.compute_network_metrics() if enable_validation else {}

    def compute_network_overlap(self):
        """
        Calcule le chevauchement entre couches sociales.
        
        Empirique (Szell et al. 2010) : 15-25% des amis sont aussi collègues.
        """
        overlap = 0
        for agent in self.agents:
            overlap += len(agent.friends & agent.colleagues)
        
        total_colleagues = sum(len(a.colleagues) for a in self.agents)
        ratio = overlap / total_colleagues if total_colleagues > 0 else 0
        
        print(f"Chevauchement amis/collègues : {ratio:.1%} (attendu : 15-25%)")
        return ratio
    
    def apply_relationship_decay(self, days_elapsed=1):
        """
        Affaiblit les liens non-entretenus (inspiré Dunbar 2018).
        
        Principe : Sans interaction pendant >30 jours, probabilité rupture = 5%/jour
        """
        for agent in self.agents:
            for friend_id in list(agent.friends):
                pair = tuple(sorted([agent.id, friend_id]))
                last_interaction_day = self._interaction_history.get(pair, 0)
                days_since = days_elapsed - last_interaction_day
                
                if days_since > 30:  # Seuil Dunbar : 1 mois sans contact
                    decay_prob = 0.05  # 5% par jour
                    if random.random() < decay_prob:
                        agent.friends.discard(friend_id)
                        self.agent_dict[friend_id].friends.discard(agent.id)