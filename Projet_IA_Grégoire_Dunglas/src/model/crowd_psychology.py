# Modélisation de la psychologie des foules

import random
from model.network_builder import SocialNetworkBuilder
from collections import Counter

class CrowdPsychology:
    def __init__(self, agents):
        self.agents = agents
        self.panic_threshold = 0.25  # Panic plus précoce (réaliste)
        self.calm_propagation_rate = 0.15  # Nouveau : vitesse de retour au calme
        self.information_cascades = []

    def update_behaviors(self):
        total = len(self.agents)
        anxieux_count = sum(1 for a in self.agents if a.psychology == 'anxieux')
        infecte_count = sum(1 for a in self.agents if a.status == 'infecté')
        
        # CASCADE INFORMATIONNELLE (Primer : un agent change → effet domino)
        if anxieux_count / total > self.panic_threshold:
            cascade_size = 0
            for agent in self.agents:
                if agent.psychology == 'calme':
                    anxious_neighbors = sum(1 for nid in agent.neighbors 
                                        for a in self.agents 
                                        if a.id == nid and a.psychology == 'anxieux')
                    cascade_prob = min(0.6, 0.15 + 0.08 * anxious_neighbors)
                    if random.random() < cascade_prob:
                        agent.psychology = 'anxieux'
                        cascade_size += 1
            
            if cascade_size > 0:
                self.information_cascades.append({
                    'tour': len(self.information_cascades),
                    'size': cascade_size,
                    'trigger': 'threshold_panic'
                })
        
        if anxieux_count / total < self.panic_threshold * 0.5:  # Si <12.5% anxieux
            calming_agents = [a for a in self.agents if a.psychology == 'anxieux']
            
            for agent in calming_agents:
                # Probabilité de retour au calme selon contexte
                calm_prob = 0.05  # 5% par tour de base
                
                # Accélération si entouré de calmes
                calm_neighbors = sum(1 for nid in agent.neighbors 
                                    for a in self.agents 
                                    if a.id == nid and a.psychology == 'calme')
                if calm_neighbors > len(agent.neighbors) * 0.7:  # >70% calmes
                    calm_prob += 0.15  # Bonus 15%
                
                # Influence des leaders
                leader_friends = [a for fid in agent.friends 
                                for a in self.agents 
                                if a.id == fid and a.psychology == 'leader']
                if leader_friends:
                    calm_prob += 0.10 * len(leader_friends)
                
                if random.random() < calm_prob:
                    agent.psychology = 'calme'
                    print(f"Agent {agent.id} redevient calme (anxiété résolue)")        
        # INFLUENCE DES LEADERS (Fouloscopie : figures d'autorité)
        leaders = [a for a in self.agents if a.psychology == 'leader']
        for agent in self.agents:
            if agent.psychology == 'suiveur' and leaders:
                influential_leaders = sorted(
                    [l for l in leaders if l.comm_id == agent.comm_id or l.id in agent.friends],
                    key=lambda l: l.trust_level,
                    reverse=True
                )
                if influential_leaders:
                    closest_leader = influential_leaders[0]
                    if random.random() < 0.5 * closest_leader.trust_level:
                        # Adopte le comportement du leader
                        agent.psychology = 'calme' if closest_leader.status != 'infecté' else 'anxieux'
        
        network = SocialNetworkBuilder(self.agents)
        network.update_friendships_over_time(interaction_matrix={},days_elapsed=1)
                            
    def apply_community_influence(self, communities, community_leaders):
        """
        Influence des leaders de communauté sur leurs membres
        Fouloscopie : effet d'entraînement, polarisation
        
        Args:
            communities (dict): {comm_id: [agent_ids]}
            community_leaders (dict): {comm_id: leader_id}
        """
        agent_dict = {a.id: a for a in self.agents}
        
        for comm_id, members in communities.items():
            leader_id = community_leaders.get(comm_id)
            if not leader_id:
                continue
            
            leader = agent_dict.get(leader_id)
            if not leader:
                continue
            
            # Le leader influence sa communauté
            for member_id in members:
                if member_id == leader_id:
                    continue
                
                member = agent_dict.get(member_id)
                if not member:
                    continue
                
                # Influence psychologique (50% de chance)
                if member.psychology == 'suiveur' and random.random() < 0.5:
                    member.psychology = leader.psychology
                
                # Influence sur la confiance (plus subtil)
                if member.trust_level < leader.trust_level:
                    member.trust_level += 0.05  # Convergence progressive
                elif member.trust_level > leader.trust_level:
                    member.trust_level -= 0.05

    def detect_echo_chambers(self, communities):
        """
        Détecte les chambres d'écho (Primer : information bubbles)
        Communautés où >80% partagent même psychologie = polarisation
        """
        agent_dict = {a.id: a for a in self.agents}
        echo_chambers = []
        
        for comm_id, members in communities.items():
            if len(members) < 5:  # Trop petit pour être significatif
                continue
            
            # Compter psychologies
            psychologies = [agent_dict[mid].psychology for mid in members if mid in agent_dict]
            psy_counts = Counter(psychologies)
            
            # Si psychologie majoritaire >80%
            most_common_psy, count = psy_counts.most_common(1)[0]
            if count / len(members) > 0.8:
                echo_chambers.append({
                    'community_id': comm_id,
                    'dominant_psychology': most_common_psy,
                    'homogeneity': count / len(members),
                    'size': len(members)
                })
        
        return echo_chambers
    
    def apply_confinement(self, agent, confinement):
        """Confinement adaptatif selon psychologie"""
        if not confinement:
            return
        
        # Probabilités réalistes (Flaxman et al. 2020, Nature)
        compliance_rates = {
            'anxieux': 0.85,    # Très strict
            'calme': 0.60,      # Modéré
            'suiveur': 0.50,
            'leader': 0.20,     # Peu respectueux
            'rebelle': 0.05     # Presque rien
        }
        
        compliance = compliance_rates.get(agent.psychology, 0.50)
        if random.random() < compliance:
            agent.current_location = 'home'
