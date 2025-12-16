# Modélisation des agents

import random

class Agent:
    def __init__(self, id, status='sain', psychology='calme', symptom_severity='none', home_quarter=None, work_quarter=None, age=None, passions=None, job=None, city_env=None):
        self.id = id
        self.status = status
        self.psychology = psychology
        self.symptom_severity = symptom_severity
        self.city_env = city_env
        self.home_quarter = home_quarter
        self.current_location = 'home'
        self.current_location_detail = self.home_quarter
        self.work_quarter = work_quarter
        self.mobility_pattern = self._choose_mobility_pattern()
        self.nuclear_family = set()
        self.extended_family = set()
        self.friends = set()
        self.connaissances = set()
        self.colleagues = set()
        self.neighbors = set()
        self.tie_strength = {}
        self.profile = random.choice(['suiveur', 'leader', 'explorateur', 'prudent', 'impulsif'])
        self.vision_range = random.randint(2, 8)
        self.attention_to_signals = random.uniform(0.1, 1.0)
        # Ajout âge, passions, métier
        self.age = age if age is not None else random.randint(8, 80)
        # Passions selon l'âge (simplifié)
        passions_by_age = [
            (range(8, 18), ['jeux vidéo', 'sport', 'musique', 'dessin']),
            (range(18, 30), ['voyages', 'musique', 'sport', 'cinéma', 'jeux vidéo']),
            (range(30, 50), ['lecture', 'cuisine', 'randonnée', 'cinéma', 'voyages']),
            (range(50, 81), ['jardinage', 'lecture', 'cuisine', 'voyages', 'peinture'])
        ]
        if passions is not None:
            self.passions = passions
        else:
            for age_range, passions_list in passions_by_age:
                if self.age in age_range:
                    self.passions = random.sample(passions_list, k=2)
                    break
            else:
                self.passions = ['lecture']
        # Liste de métiers (simplifié)
        jobs = ['enseignant', 'ingénieur', 'médecin', 'étudiant', 'retraité', 'artiste', 'commerçant', 'ouvrier', 'cadre', 'infirmier']
        self.job = job if job is not None else random.choice(jobs)
        # Lieux favoris dynamiques selon l'environnement
        self.favorite_places = {
            'restaurant': None,
            'cafe': None,
            'supermarche': None,
            'gym': None,
            'parc': None,
            'cinema': None
        }
        self.comm_id = -1  # Identifiant de la communauté sociale

        # Mémoire des visites (pour fidélisation progressive)
        self.place_visit_count = {place: 0 for place in self.favorite_places.keys()}

        # Routine hebdomadaire détaillée
        self.weekly_routine = self._generate_weekly_routine()
        self.trust_level = random.uniform(0.3, 1.0)  # Niveau de confiance aux mesures sanitaires
        self.anxiety_level = random.uniform(0.0, 1.0)  # Niveau d'anxiété initial
        self.state = 'calme' if self.psychology != 'anxieux' else 'anxieux'

        self.vaccine_hesitancy = random.uniform(0, 1)  # 0=pro-vaccin, 1=anti-vaccin
        self.risk_perception = random.uniform(0.3, 1.0)  # Perception du danger
        self.compliance_to_rules = random.uniform(0, 1)  # Respect des consignes
        
        # Initialisation selon psychologie
        if self.psychology == 'anxieux':
            self.risk_perception *= 1.5
            self.compliance_to_rules *= 1.4
        elif self.psychology == 'rebelle':
            self.compliance_to_rules *= 0.3
            self.vaccine_hesitancy *= 1.3
        elif self.psychology == 'leader':
            self.vaccine_hesitancy *= 0.7
        self.is_vaccinated = False
        self.vaccination_day = None
        self.vaccine_type = None
        self.décroissance_anxiété = 0.05  # Taux de réduction de l'anxiété
        # Ajout des attributs prénom et nom
        liste_prénom = ['Alice', 'Bob', 'Charlie', 'Diane', 'Eve', 'Frank', 'Grace', 'Hugo', 'Ivy', 'Jack', 'Chlo', 'Lucas', 'Mia', 'Noah', 'Olivia', 'Paul', 'Quinn', 'Rose', 'Sam', 'Tina', 'Sophie', 'Olym', 'Tacx', 'Haechi', 'Ele', 'Tito', 'Grégoire', 'Loïc', 'Louis', 'Emma', 'Léa', 'Anaëlle', 'Victoire', 'Oscar', 'Renan', 'Pauly', 'Paul', 'Clément', 'Georgette', 'Coralie', 'Antoine', 'Aurélie', 'Dimitri']
        liste_nom = ['Martin', 'Bernard', 'Thomas', 'Petit', 'Robert', 'Richard', 'Durand', 'Dubois', 'Moreau', 'Laurent', 'Shvili', 'Slyze', 'Diurnal', 'Martine', 'Leroy', 'Roux', 'David', 'Morel', 'Fournier', 'Girard', 'Bonnet', 'Dupont', 'Lambert', 'Fontaine', 'Rousseau', 'Vincent', 'Muller', 'Lemoine', 'Blanc', 'Garnier', 'Faure', 'Chevalier', 'Chombre','Dupont', 'Cros', 'Dusautoir', 'Segonds', 'Capilla']
        self.prenom = random.choice(liste_prénom)
        self.nom = random.choice(liste_nom)
        

    def update_anxiety(self, infected_in_community):
        """
        Réduit l'anxiété si le nombre d'infectés dans la communauté baisse à zéro.
        """
        if self.state == "anxieux" and infected_in_community == 0:
            self.anxiety_level -= self.décroissance_anxiété
            if self.anxiety_level < 0.2:
                self.state = "calme"

    def likeability(self, other):
        """
        Calcule le score d'affinité (likeability) entre deux agents selon Fouloscopie :
        - Similarité de psychologie
        - Proximité spatiale
        - Attention aux signaux
        - Proximité d'âge
        - Passions communes
        - Métier identique
        """
        score = 0
        # Psychologie
        if self.psychology == other.psychology:
            score += 0.5
        # Lieux communs fréquentés
        common_places = 0
        if self.home_quarter == other.home_quarter:
            common_places += 1
        if self.work_quarter == other.work_quarter:
            common_places += 1
        score += 0.2 * common_places
        # Attention aux signaux
        score += 0.2 * ((self.attention_to_signals + other.attention_to_signals) / 2)
        # Proximité d'âge (plus c'est proche, plus c'est fort)
        age_diff = abs(self.age - other.age)
        score += max(0, 0.3 - 0.01 * age_diff)
        # Passions communes
        if hasattr(self, 'passions') and hasattr(other, 'passions'):
            common_passions = set(self.passions) & set(other.passions)
            score += 0.25 * len(common_passions)
        # Même métier
        if hasattr(self, 'job') and hasattr(other, 'job') and self.job == other.job:
            score += 0.35
        return score

    def _generate_weekly_routine(self):
        """
        Génère routine hebdomadaire basée sur :
        - Âge (étudiants ≠ retraités ≠ actifs)
        - Métier (horaires variables)
        - Psychologie (leaders sortent plus)
        - Passions (sport régulier, cinéma le weekend)
        
        Sources : Études mobilité urbaine (INSEE), Fouloscopie (patterns collectifs)
        """
        routine = {
            'weekday': {
                'morning': 'work',
                'midday': 'work',
                'afternoon': 'work',
                'evening': 'home',
                'night': 'home'
            },
            'weekend': {
                'morning': 'home',
                'midday': 'home',
                'afternoon': 'home',
                'evening': 'home',
                'night': 'home'
            }
        }
        
        # ========== SPÉCIFICITÉS PAR ÂGE ==========
        
        # ÉTUDIANTS (8-25 ans)
        if 8 <= self.age <= 25:
            routine['weekday']['morning'] = 'work'  # Cours
            routine['weekday']['midday'] = 'restaurant' if random.random() < 0.6 else 'work'
            routine['weekday']['afternoon'] = 'work'
            routine['weekday']['evening'] = 'public' if random.random() < 0.5 else 'home'
            
            # Weekend : sorties fréquentes
            routine['weekend']['morning'] = 'home'
            routine['weekend']['midday'] = 'cafe' if random.random() < 0.4 else 'home'
            routine['weekend']['afternoon'] = 'public' if random.random() < 0.7 else 'sport'
            routine['weekend']['evening'] = 'public' if random.random() < 0.6 else 'restaurant'
        
        # ACTIFS (26-60 ans)
        elif 26 <= self.age <= 60:
            # Métiers avec horaires spécifiques
            if self.job in ['médecin', 'infirmier']:
                # Personnel soignant : horaires décalés possibles
                if random.random() < 0.3:  # 30% en horaires de nuit
                    routine['weekday']['evening'] = 'work'
                    routine['weekday']['night'] = 'work'
                    routine['weekday']['morning'] = 'home'  # Repos après nuit
            
            elif self.job == 'commerçant':
                # Commerçants : travail le samedi
                routine['weekend']['morning'] = 'work'
                routine['weekend']['midday'] = 'work'
                routine['weekend']['afternoon'] = 'work'
            
            elif self.job in ['enseignant', 'cadre']:
                # Horaires de bureau classiques
                routine['weekday']['midday'] = 'restaurant' if random.random() < 0.4 else 'work'
            
            # Soirées selon psychologie
            if self.psychology == 'leader':
                routine['weekday']['evening'] = 'restaurant' if random.random() < 0.6 else 'home'
                routine['weekend']['evening'] = 'public' if random.random() < 0.7 else 'restaurant'
            elif self.psychology == 'anxieux':
                routine['weekday']['evening'] = 'home'
                routine['weekend']['evening'] = 'home'
            else:
                routine['weekday']['evening'] = 'home' if random.random() < 0.7 else 'restaurant'
                routine['weekend']['evening'] = 'public' if random.random() < 0.4 else 'home'
        
        # RETRAITÉS (>60 ans)
        else:
            # Journées plus flexibles
            routine['weekday']['morning'] = 'home' if random.random() < 0.6 else 'shopping'
            routine['weekday']['midday'] = 'home' if random.random() < 0.7 else 'cafe'
            routine['weekday']['afternoon'] = 'home' if random.random() < 0.5 else 'public'
            routine['weekday']['evening'] = 'home'
            
            # Weekend : visites familiales
            routine['weekend']['morning'] = 'home'
            routine['weekend']['midday'] = 'family_visit' if random.random() < 0.3 else 'home'
            routine['weekend']['afternoon'] = 'home' if random.random() < 0.6 else 'public'
        
        # ========== AJUSTEMENTS PAR PASSIONS ==========
        # Adaptation des routines en fonction des centres d'intérêt

        # ========== AJUSTEMENTS PAR PSYCHOLOGIE ==========
        # Ajustements comportementaux selon le profil psychologique
        
        if 'sport' in self.passions:
            # 2-3 séances par semaine
            routine['weekday']['evening'] = 'sport' if random.random() < 0.4 else routine['weekday']['evening']
            routine['weekend']['afternoon'] = 'sport' if random.random() < 0.5 else routine['weekend']['afternoon']
        
        # Cinéphiles
        if 'cinéma' in self.passions:
            routine['weekend']['evening'] = 'public' if random.random() < 0.6 else routine['weekend']['evening']
        
        # Cuisiniers (sortent moins au restaurant)
        if 'cuisine' in self.passions:
            routine['weekday']['midday'] = 'home' if random.random() < 0.7 else routine['weekday']['midday']
        
        # Voyageurs (explorent plus)
        if 'voyages' in self.passions:
            routine['weekend']['afternoon'] = 'public' if random.random() < 0.7 else routine['weekend']['afternoon']
        
        # ========== AJUSTEMENTS PAR PSYCHOLOGIE ==========
        
        if self.psychology == 'rebelle':
            # Rebelles cassent la routine
            for day_type in ['weekday', 'weekend']:
                for time in routine[day_type].keys():
                    if random.random() < 0.2:  # 20% de chaos
                        routine[day_type][time] = random.choice(['public', 'restaurant', 'cafe'])
        
        elif self.psychology == 'suiveur':
            # Suiveurs imitent la majorité (routine conventionnelle)
            # Déjà le cas par défaut, pas de changement
            pass
        
        return routine

    def decide_vaccination(self, epidemic_severity, agent_dict):
        """
        Décision individuelle de vaccination
        
        Facteurs (Betsch et al. 2018, PNAS) :
        - Gravité perçue de l'épidémie
        - Hésitation vaccinale
        - Influence sociale (amis vaccinés)
        
        Args:
            epidemic_severity : float [0,1] (% population infectée)
            agent_dict : dict {id: Agent} dictionnaire de tous les agents
        
        Returns:
            bool : True si accepte vaccination
        """
        # Facteur 1 : Gravité perçue
        severity_factor = epidemic_severity * self.risk_perception
        
        # Facteur 2 : Influence sociale
        if self.friends:
            vaccinated_friends = sum(
                1 for fid in self.friends 
                    if fid in agent_dict and agent_dict[fid] == 'R'
            )
            social_influence = vaccinated_friends / len(self.friends)
        else:
            social_influence = 0
        
        # Formule décisionnelle (régression logistique empirique)
        decision_score = (
            0.4 * severity_factor +
            0.3 * social_influence +
            0.3 * (1 - self.vaccine_hesitancy)
        )
        
        # Seuil adaptatif selon psychologie
        threshold = {
            'leader': 0.4,
            'anxieux': 0.3,
            'rebelle': 0.8,
            'suiveur': 0.5,
            'calme': 0.5
        }.get(self.psychology, 0.5)
        
        return decision_score > threshold

    def decide_confinement(self, local_infection_rate):
        """
        Décision de rester chez soi
        
        Args:
            local_infection_rate : float (% infectés dans quartier)
        
        Returns:
            bool : True si reste home
        """
        # Leaders et rebelles ignorent confinement
        if self.psychology in ['leader', 'rebelle']:
            if random.random() > 0.2:  # 80% désobéissent
                return False
        
        # Anxieux confinement strict
        if self.psychology == 'anxieux':
            return True
        
        # Autres : fonction de la gravité locale
        confinement_prob = (
            0.5 * local_infection_rate +
            0.3 * self.compliance_to_rules +
            0.2 * self.risk_perception
        )
        
        return random.random() < confinement_prob

    def _choose_mobility_pattern(self):
        # Définir un schéma de mobilité selon le profil
        if self.psychology == 'calme':
            return ['home', 'work']
        elif self.psychology == 'anxieux':
            return ['home', 'work', 'home']
        elif self.psychology == 'leader':
            return ['work', 'public', 'home']
        elif self.psychology == 'rebelle':
            return ['public', 'home', 'work']
        elif self.psychology == 'suiveur':
            return ['home', 'work', 'public']
        return ['home', 'work']
    
    def update_status(self, new_status):
        self.status = new_status

    def __repr__(self):
        return (f"Agent(id={self.id}, status={self.status}, psychology={self.psychology}, "
                f"age={self.age}, job={self.job}, passions={self.passions}, "
                f"position={self.current_location}, home={self.home_quarter}, work={self.work_quarter}, "
                f"nuclear_family={list(self.nuclear_family)}, extended_family={list(self.extended_family)}, friends={list(self.friends)}, "
                f"colleagues={list(self.colleagues)}, neighbors={list(self.neighbors)})")
