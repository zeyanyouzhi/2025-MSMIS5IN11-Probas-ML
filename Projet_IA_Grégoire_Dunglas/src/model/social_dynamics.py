# Mécanismes de défense collective contre la maladie

import random

class SocialDefenseMechanisms:
    """
    Simule les comportements collectifs de défense inspirés de Fouloscopie :
    - Formation de bulles sociales
    - Groupes de protection mutuelle
    - Leaders d'opinion sur la santé
    - Propagation de bonnes pratiques
    """
    
    def __init__(self, agents):
        self.agents = agents
        self.social_bubbles = []  # Bulles de confinement volontaire
        self.health_leaders = []  # Agents influents sur la santé
        
    def form_social_bubbles(self):
        """
        Formation de bulles avec propriété "petit monde" (Primer)
        Clusters denses + quelques liens faibles inter-clusters
        """
        bubbles_formed = []
        processed = set()
        
        for agent in self.agents:
            if agent.id in processed:
                continue
            
            # Bubble = famille proche + 1-2 amis proches
            bubble = {agent.id} | agent.family
            
            # Amis proches (liens forts)
            if agent.friends:
                friend_scores = [(fid, agent.likeability(next(a for a in self.agents if a.id == fid))) 
                                for fid in agent.friends]
                friend_scores.sort(key=lambda x: x[1], reverse=True)
                bubble.update([fid for fid, score in friend_scores[:2] if score > 0.7])
            
            # AJOUT : Liens faibles (10% chance de connexion inter-bulles)
            if random.random() < 0.1:
                random_agent = random.choice([a for a in self.agents if a.id not in processed])
                bubble.add(random_agent.id)
            
            bubbles_formed.append(bubble)
            processed.update(bubble)
        
        self.social_bubbles = bubbles_formed
        return bubbles_formed
        
    def identify_health_leaders(self):
        """
        Identifie les agents qui deviennent leaders d'opinion sur la santé
        Basé sur : profession médicale, psychologie leader, confiance élevée
        """
        self.health_leaders = [
            a for a in self.agents 
            if (a.job in ['médecin', 'infirmier', 'enseignant'] 
                or a.psychology == 'leader')
            and getattr(a, 'trust_level', 0.5) > 0.7
        ]
        return self.health_leaders
    
    def propagate_health_behaviors(self):
        """
        Les leaders propagent les bonnes pratiques dans leurs réseaux
        Ex: port du masque, distanciation, respect du confinement
        """
        for leader in self.health_leaders:
            # Influencer les followers
            all_contacts = leader.family | leader.friends | leader.colleagues | leader.neighbors
            
            for contact_id in all_contacts:
                contact = next((a for a in self.agents if a.id == contact_id), None)
                if contact and random.random() < 0.4:
                    # Augmente la confiance et adopte comportement prudent
                    if hasattr(contact, 'trust_level'):
                        contact.trust_level = min(1.0, contact.trust_level + 0.1)
                    if contact.psychology in ['impulsif', 'rebelle']:
                        if random.random() < 0.3:
                            contact.psychology = 'prudent'
    
    def apply_social_stigma(self, epidemic_model):
        """
        Les agents évitent les malades visibles (Fouloscopie : évitement spatial)
        """
        for agent in self.agents:
            if agent.status == 'infecté' and agent.symptom_severity == 'severe':
                # Réduction drastique des contacts
                agent.friends = set(list(agent.friends)[:2])  # Garde seulement 2 amis proches
                agent.colleagues = set()  # Arrêt de travail

    def enforce_bubble_isolation(self, epidemic_model):
        """
        Les bulles sociales réduisent les contacts externes
        Diminue la probabilité de contamination inter-bulles
        """
        # Identifier quelle bulle pour chaque agent
        agent_to_bubble = {}
        for i, bubble in enumerate(self.social_bubbles):
            for agent_id in bubble:
                agent_to_bubble[agent_id] = i
        
        # Réduire contacts hors bulle
        for agent in self.agents:
            if agent.id in agent_to_bubble:
                bubble_idx = agent_to_bubble[agent.id]
                # Limiter amis et collègues hors bulle
                agent.friends = {fid for fid in agent.friends 
                                if fid in self.social_bubbles[bubble_idx]}