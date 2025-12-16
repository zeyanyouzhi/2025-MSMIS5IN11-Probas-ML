import random
from typing import Tuple, Optional, List

class MobilityManager:
    """
    Gère les déplacements réalistes des agents dans l'environnement urbain.
    """
    
    def __init__(self, city_env):
        self.city_env = city_env
        self.activity_to_place = {
            'restaurant': ['restaurant', 'cafe'],
            'cafe': ['cafe', 'restaurant'],
            'shopping': ['supermarche', 'supermarket'],
            'supermarket': ['supermarche', 'supermarket'],
            'sport': ['gym', 'parc'],
            'gym': ['gym'],
            'parc': ['parc'],
            'cinema': ['cinema'],
            'public': ['parc', 'cinema'],
        }

    def decide_location(self, agent, time_of_day: str, day_of_week: str, 
                        all_agents: List, confinement_active: bool = False) -> str:
        """
        Décide du lieu où un agent se rend en fonction de son profil et des contraintes.
        
        Args:
            agent: L'agent pour lequel décider.
            time_of_day (str): Moment de la journée (matin, après-midi, soir).
            day_of_week (str): Jour de la semaine.
            all_agents (List): Liste de tous les agents.
            confinement_active (bool): Si le confinement est actif.
        
        Returns:
            str: Lieu choisi.
        """
        if time_of_day is None:
            time_of_day = 'morning'
        
        if confinement_active:
            compliance_rates = {
                'anxieux': 0.95,
                'calme': 0.70,
                'suiveur': 0.60,
                'leader': 0.25,
                'rebelle': 0.05
            }
            if random.random() < compliance_rates.get(agent.psychology, 0.60):
                if time_of_day == 'afternoon' and random.random() < 0.14:
                    return 'supermarket'
                return 'home'
        
        if hasattr(agent, 'weekly_routine'):
            preferred_activity = agent.weekly_routine[day_of_week][time_of_day]
            
            # Anxieux suivent routine strictement
            if agent.psychology == 'anxieux' and random.random() < 0.85:
                return self._activity_to_location(agent, preferred_activity)
            
            # Leaders sortent plus
            elif agent.psychology == 'leader' and random.random() < 0.7:
                return self._activity_to_location(agent, preferred_activity)
            
            # Rebelles cassent routine
            elif agent.psychology == 'rebelle' and random.random() < 0.5:
                return self._activity_to_location(agent, preferred_activity)
            
            # Suiveurs/Calmes suivent routine
            elif agent.psychology in ['suiveur', 'calme'] and random.random() < 0.8:
                return self._activity_to_location(agent, preferred_activity)
            
            # Fallback
            if hasattr(agent, 'current_location') and agent.current_location:
                return agent.current_location
            else:
                return 'home'
        
        return 'home'

    def _activity_to_location(self, agent, activity: str) -> str:
        """
        Convertit activité planifiée en lieu symbolique
        """
        mapping = {
            'work': 'work' if agent.job not in ['retraité', 'étudiant'] else 'home',
            'home': 'home',
            'restaurant': 'restaurant',
            'sport': 'gym',
            'shopping': 'supermarket',
            'public': 'public',
            'transports': 'transports',
            'cafe': 'restaurant',
            'family_visit': 'home',
            'cinema': 'public',
            'bar': 'restaurant'
        }
        return mapping.get(activity, 'home')

    def _get_favorite_or_nearby(self, agent, place_key: str, candidates: list = None) -> str:
        """Résolution robuste des lieux favoris"""
        if not hasattr(agent, 'favorite_places'):
            agent.favorite_places = {}
        
        if agent.favorite_places.get(place_key):
            return agent.favorite_places[place_key]
        
        city_env = getattr(agent, 'city_env', None) or self.city_env
        
        if not city_env:
            agent.favorite_places[place_key] = 'home'
            return 'home'
        
        # Essayer type exact
        if hasattr(city_env, 'get_random_location'):
            loc = city_env.get_random_location(place_key, agent.home_quarter)
            if loc:
                agent.favorite_places[place_key] = loc
                return loc
        
        # Fallback sur types similaires
        similar_types = self.activity_to_place.get(place_key, [place_key])
        
        for similar in similar_types:
            if hasattr(city_env, 'get_random_location'):
                loc = city_env.get_random_location(similar, agent.home_quarter)
                if loc:
                    agent.favorite_places[place_key] = loc
                    return loc
        
        # Chercher dans toute la ville
        for similar in similar_types:
            if hasattr(city_env, 'get_random_location'):
                loc = city_env.get_random_location(similar)
                if loc:
                    agent.favorite_places[place_key] = loc
                    return loc
        
        # Dernier fallback
        agent.favorite_places[place_key] = 'home'
        return 'home'
    
    def _social_outing(self, agent) -> str:
        """Sortie loisir selon passions"""
        if not self.city_env.public_places:
            return agent.home_quarter
        
        if 'cinéma' in agent.passions and random.random() < 0.3:
            return random.choice(self.city_env.public_places)
        elif agent.psychology in ['rebelle', 'leader']:
            return random.choice(self.city_env.gathering_points or self.city_env.public_places)
        else:
            return random.choice(self.city_env.public_places)
    
    def _meet_friend(self, agent, all_agents):
        """Rencontre avec ami (synchronisation spatiale)"""
        friends_list = [a for a in all_agents if a.id in agent.friends]
        if not friends_list:
            return agent.home_quarter, 'home'
        
        friend = random.choice(friends_list)
        meeting_spot = random.choice(['restaurant', 'public', 'cafe'])
        
        # Synchronisation spatiale
        if random.random() < 0.7:
            friend.current_location = meeting_spot
        
        return meeting_spot, meeting_spot
    
    def _get_random_colleague(self, agent, all_agents: List):
        """Récupère collègue aléatoire"""
        if all_agents is None:
            return None
        colleagues = [a for a in all_agents if a.id in agent.colleagues]
        return random.choice(colleagues) if colleagues else None

    def simulate_random_encounters(self, agent, all_agents):
        """
        Simule rencontres aléatoires (Vázquez 2007)
        """
        if agent.current_location not in ['restaurant', 'public', 'transports']:
            return []
        
        # Agents au même lieu
        colocated = [
            a for a in all_agents 
            if a.id != agent.id 
            and getattr(a, 'current_location') == agent.current_location
        ]
        
        # Échantillonner 2-5 rencontres
        n_encounters = random.randint(2, min(5, len(colocated)))
        return random.sample(colocated, n_encounters) if colocated else []