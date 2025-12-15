"""
Surveillance en temps réel de la qualité du réseau et de l'épidémie.
"""

import numpy as np
from collections import Counter
from model.metrics import compute_R0_dynamic

class SimulationMonitor:
    """
    Surveille les métriques critiques pendant la simulation.
    
    Sources scientifiques :
        - Lloyd-Smith et al. (2005) : Super-spreading events
        - Pastor-Satorras & Vespignani (2001) : Epidemic thresholds
    """
    
    def __init__(self, agents, epidemic_model):
        self.agents = agents
        self.epidemic = epidemic_model
        self.alerts = []

    def check_network_health(self):
        """
        Vérifie la santé du réseau social en analysant les degrés des agents.
        """
        degrees = [len(a.friends) for a in self.agents]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        isolated = sum(1 for d in degrees if d == 0)
        
        warnings = []
        
        if avg_degree < 10:
            warnings.append(f"Degré moyen {avg_degree:.1f} < 10 (Dunbar minimum)")
        
        if max_degree < 30:
            warnings.append(f"Pas de hubs (max={max_degree}, attendu >50)")
        
        if isolated > 0:
            warnings.append(f"{isolated} agents isolés")
        
        cv = np.std(degrees) / avg_degree if avg_degree > 0 else 0
        if cv < 0.5:
            warnings.append(f"Distribution trop homogène (CV={cv:.2f}, attendu >0.8)")
        
        if warnings:
            print("\nALERTES RÉSEAU :")
            for w in warnings:
                print(f"  {w}")
                self.alerts.append(('network', w))
        
        return len(warnings) == 0
    
    def check_epidemic_progress(self, day):
        """Vérifie la progression de l'épidémie"""
        infected = sum(1 for a in self.agents if a.status in ['infecté', 'incubation'])
        
        warnings = []
        
        if day > 5 and infected == 0:
            warnings.append("Épidémie éteinte prématurément")
        
        if day > 3 and infected == len([a for a in self.agents if a.status == 'infecté']):
            warnings.append("Aucune nouvelle infection (R0 probablement < 1)")
        
        # Vérifier matrice de contacts
        if hasattr(self.epidemic, 'transmission_memory'):
            transmissions = len(self.epidemic.transmission_memory)
            if day > 5 and transmissions == 0:
                warnings.append("Aucune transmission enregistrée → vérifier callback")
        
        if warnings:
            print(f"\n ALERTES ÉPIDÉMIE (jour {day}) :")
            for w in warnings:
                print(f"  {w}")
                self.alerts.append(('epidemic', w))

        if day > 5:
            r0 = compute_R0_dynamic(self.epidemic, day)
            
            if r0 < 0.5:
                warnings.append(f"R₀={r0:.2f} trop faible → extinction imminente")
            elif r0 > 4.0:
                warnings.append(f"R₀={r0:.2f} trop élevé → explosion irréaliste")
            
            # Vérifier transmissions enregistrées
            if len(self.epidemic.transmission_memory) == 0:
                warnings.append("CRITIQUE : Aucune transmission depuis début → bug callback")
        return len(warnings) == 0
    
    def check_location_consistency(self):
        """Vérifie que tous les agents ont des lieux valides"""
        valid_locations = {
            'home', 'work', 'restaurant', 'public', 'transports', 
            'gym', 'supermarket', 'hospital'
        }
        
        errors = []
        
        for agent in self.agents:
            loc = getattr(agent, 'current_location', None)
            
            if loc is None:
                errors.append(f"Agent {agent.id} : current_location = None")
            elif loc not in valid_locations:
                errors.append(f"Agent {agent.id} : current_location = '{loc}' invalide")
        
        if errors:
            print("\nERREURS LOCALISATION :")
            for e in errors[:10]:  # Afficher 10 premières
                print(f"  {e}")
            raise RuntimeError(f"{len(errors)} agents avec localisation invalide")
        
        return True