# Modélisation de l'épidémie
import math
import random
import numpy as np


class EpidemicModel:
    def __init__(self, agents, infection_radius=1, infection_prob=0.8, city_env=None, interaction_callback=None):
        self.agents = agents
        self.infection_radius = infection_radius
        self.infection_prob = infection_prob
        self.infection_duration = 7
        self.agent_states = {}
        self.incubation_timers = {} 
        self.infection_timers = {}
        self.city_env = city_env
        self.transmission_memory = {}
        # Paramètres covid
        self.incubation_period_mean = 6.5  # Médiane 6.5 jours
        self.incubation_period_std = 2.0   # Écart-type réaliste

        self.infection_duration_mean = 14   # 14 jours contagieux
        self.infection_duration_std = 3     # Variabilité inter-individuelle
        self.presymptomatic_transmission_prob = 0.44

        # Distribution lognormale (plus réaliste que normale)
        self.use_lognormal_periods = True
        self.agent_incubation_periods = {}  # Durée par agent
        self.agent_infection_day = {}  # Jour d'infection de chaque agent
        self.interaction_callback = interaction_callback  # Callback pour enregistrer les interactions
        self.agent_transmissibility = {}  # β individuel par agent (gamma distribution)
        self._initialize_heterogeneous_transmissibility()
        self.effet_leader = 0.15  # Réduction du risque si leader santé dans la communauté
        for agent in agents:
            if agent.status == 'infecté':
                self.agent_states[agent.id] = 'I'  # Infectious
                self.infection_timers[agent.id] = 0
            else:
                self.agent_states[agent.id] = 'S'  # Susceptible
        self.compteur_log_familial = 0  # Compteur pour logging des transmissions familiales
        self.compteur_log_risque = 0  # Compteur pour logging des risques
        self.compteur_log_status = 0  # Compteur pour logging des statuts
        self.compteur_contacts = 0  # Compteur pour logging des contacts
        self.compteur_log_entree_aerosols = 0  # Compteur pour logging des entrées dans aérosols
        self.compteur_log_immunite = 0  # Compteur pour logging des pertes d'immunité
        self.variants = {
            'alpha': {'transmissibility': 1.0, 'severity': 1.0},  # Souche originale
            'beta': {'transmissibility': 1.5, 'severity': 0.9},   # Plus contagieux
            'gamma': {'transmissibility': 1.3, 'severity': 1.2}   # Plus sévère
        }
        self.agent_variant = {}  # {agent_id: variant_name}
        self.mutation_prob = 0.1  # 10% chance mutation par transmission
        self.vaccinated_agents = set()
        self.vaccination_history = {}


    def vaccinate_agent(self, agent, current_day, vaccine_type='pfizer'):
        """
        Vaccine un agent avec immunité PARFAITE
        
        Efficacité selon littérature :
        - Pfizer/BioNTech : 95% (Polack et al. 2020, NEJM)
        - Moderna : 94% (Baden et al. 2021, NEJM)
        - AstraZeneca : 70% (Voysey et al. 2021, Lancet)
        
        Args:
            agent: Agent à vacciner
            current_day: Jour de vaccination
            vaccine_type: Type de vaccin (défaut: pfizer)
        
        Returns:
            bool: True si vaccination réussie
        """
        # Vérifications
        if agent.is_vaccinated:
            return False  # Déjà vacciné
        
        if self.agent_states.get(agent.id) in ['I', 'E']:
            return False  # Déjà infecté (trop tard)
        
        #  VACCINATION = IMMUNITÉ PARFAITE
        agent.is_vaccinated = True
        agent.vaccination_day = current_day
        agent.vaccine_type = vaccine_type
        
        # Passage direct à l'état R (immunisé)
        self.agent_states[agent.id] = 'V'  # Nouvel état : Vacciné
        agent.update_status('vacciné')
        
        # Enregistrement
        self.vaccinated_agents.add(agent.id)
        self.vaccination_history[agent.id] = current_day
        
        return True

    def set_network_hubs(self, hubs):
            """
            Enregistre les hubs pour modéliser leur rôle de superspreaders
            """
            self.network_hubs = hubs

    def trigger_new_wave(self, day, variant_name='delta'):
        """
        Déclenche une nouvelle vague avec variant émergent
        
        Inspiré de : Davies et al. (2021) - Omicron immune escape
        """
        # Sélectionner 3-5 "patient zéros" (voyageurs internationaux)
        n_seeds = random.randint(3, 5)
        candidates = [a for a in self.agents if self.agent_states.get(a.id) == 'S']
        
        if len(candidates) < n_seeds:
            return False
        
        seeds = random.sample(candidates, n_seeds)
        
        for agent in seeds:
            self._infect_agent(agent, day, variant=variant_name, source_id=None)
            print(f"  🌊 Nouvelle vague : Agent {agent.id} infecté ({variant_name})")
        
        # Enregistrer début de vague
        if not hasattr(self, 'waves'):
            self.waves = []
        
        self.waves.append({
            'day': day,
            'variant': variant_name,
            'seeds': [a.id for a in seeds]
        })
        
        return True

    def propagate(self, current_day=0, time_of_day='morning'):
        """
        Modèle SEIR avec transmission CO-LOCALISÉE STRICTE
        
        PRINCIPE SCIENTIFIQUE (Brockmann & Helbing 2013) :
        - La transmission nécessite PROXIMITÉ SPATIALE effective
        - Les liens sociaux définissent la PROBABILITÉ conditionnelle
        - La co-localisation est le GATE absolu
        """
        
        # === PHASE 1 : Collecter agents contagieux (E + I) ===
        contagious = [
            a for a in self.agents 
            if self.agent_states.get(a.id) in ['E', 'I']
        ]
        
        if not contagious:
            return
        
        # === PHASE 2 : Grouper agents par lieu ACTUEL ===
        locations = {}
        for agent in self.agents:
            loc = getattr(agent, 'current_location', 'home')
            locations.setdefault(loc, []).append(agent)
        
        # === DEBUG : Afficher la répartition des agents par lieu ===
        print(f"\n[DEBUG] Répartition des agents par lieu (jour {current_day}, {time_of_day}):")
        for loc, agents_present in locations.items():
            n_contagieux = sum(1 for a in agents_present if self.agent_states.get(a.id) in ['E', 'I'])
            print(f"  - {loc}: {len(agents_present)} présents, {n_contagieux} contagieux")

        # === PHASE 3 : Propagation INTRA-LIEU UNIQUEMENT ===
        for location, agents_present in locations.items():
            # Filtrer contagieux dans CE lieu
            local_contagious = [a for a in agents_present if self.agent_states.get(a.id) in ['E', 'I']]
            
            if not local_contagious:
                continue
            
            # Risque de base du lieu
            base_risk = self.city_env.get_risk_index(location) if self.city_env else 0.5
            
            for inf in local_contagious:
                state = self.agent_states[inf.id]
                
                # Charge virale selon état
                if state == 'E':  # Incubation
                    days_since_infection = current_day - self.agent_infection_day.get(inf.id, current_day)
                    incubation_progress = days_since_infection / self.incubation_timers.get(inf.id, 5)
                    viral_load = min(0.8, 0.2 + 0.6 * incubation_progress)
                else:  # I (symptomatique)
                    days_infected = self.infection_timers.get(inf.id, 0)
                    viral_load = 1.0 * np.exp(-0.15 * max(0, days_infected - 2))
                
                variant = self.agent_variant.get(inf.id, 'alpha')
                if variant not in self.variants:
                    # Ajout d'un variant inconnu avec transmissibilité par défaut
                    self.variants[variant] = {'transmissibility': 1.0}
                variant_boost = self.variants[variant]['transmissibility']
                
                effective_prob = 50 * self.infection_prob * base_risk * viral_load * variant_boost
                
                # === TRANSMISSION SELON TYPE DE LIEU + LIEN SOCIAL ===
                
                if location == 'home':
                    # Transmission familiale (cohabitants)
                    for target in agents_present:
                        if (target.id == inf.id or 
                            self.agent_states.get(target.id) != 'S'):
                            continue
                        
                        if target.is_vaccinated:
                            # Échappement immunitaire variants Beta/Gamma
                            if variant in ['beta', 'gamma']:
                                escape_prob = 0.15  # 15% échappement
                                if random.random() < escape_prob:
                                    pass  # Infection possible
                                else:
                                    continue
                            else:
                                continue  # Protection totale vs Alpha

                        # Vérifier lien familial ET même domicile
                        if (target.id in inf.nuclear_family and 
                            target.home_quarter == inf.home_quarter):
                            
                            if time_of_day in ['evening', 'night', 'morning']:
                                time_multiplier = {'evening': 0.7, 'night': 0.9, 'morning': 0.5}[time_of_day]
                                daily_prob = 0.30
                                prob = 80 * (1 - (1 - daily_prob) ** (1/14))
                                prob *= viral_load * time_multiplier
                                
                                if random.random() < prob:
                                    self._infect_agent(target, current_day, variant=variant, source_id=inf.id)
                                    if self.interaction_callback:
                                        self.interaction_callback(inf.id, target.id, location)
                
                elif location == 'work':
                    # Transmission professionnelle (collègues)
                    if time_of_day in ['morning', 'midday', 'afternoon']:
                        for target in agents_present:
                            if (target.id == inf.id or 
                                self.agent_states.get(target.id) != 'S'):
                                continue
                            
                            # Vérifier lien professionnel ET même lieu de travail
                            if (target.id in inf.colleagues and 
                                target.work_quarter == inf.work_quarter):
                                
                                workplace_prob = effective_prob * (
                                    1.5 if inf.job in ['enseignant', 'commerçant', 'médecin'] else 0.8
                                )
                                
                                if random.random() < workplace_prob:
                                    self._infect_agent(target, current_day, variant=variant, source_id=inf.id)
                                    if self.interaction_callback:
                                        self.interaction_callback(inf.id, target.id, location)
                
                elif location in [
                    'restaurant', 'cafe', 'public', 'transports', 'gym', 'supermarche', 'supermarket', 'event', 'parc', 'cinema', 'sport']:
                    # Contacts avec amis présents
                    friends_present = [a for a in agents_present 
                                    if a.id in inf.friends and 
                                    self.agent_states.get(a.id) == 'S']
                    
                    for friend in friends_present:
                        friend_prob = effective_prob * 80 * 1.2  # Bonus proximité sociale
                        if random.random() < friend_prob:
                            self._infect_agent(friend, current_day, variant=variant, source_id=inf.id)
                            if self.interaction_callback:
                                self.interaction_callback(inf.id, friend.id, location)
                    
                    # Contacts aléatoires (échantillonnage)
                    potential_contacts = [a for a in agents_present 
                                        if a.id != inf.id 
                                        and a.id not in inf.friends
                                        and self.agent_states.get(a.id) == 'S']
                    
                    n_contacts = max(2, int(len(potential_contacts) * random.uniform(0.3, 0.5)))
                    random_contacts = random.sample(potential_contacts, min(n_contacts, len(potential_contacts)))
                    
                    place_multiplier = {
                        'restaurant': 0.8, 'transports': 0.9, 'gym': 0.7,
                        'supermarket': 0.4, 'public': 0.5, 'event': 1.2
                    }.get(location, 0.6)
                    
                    for contact in random_contacts:
                        if random.random() < effective_prob * place_multiplier:
                            self._infect_agent(contact, current_day, variant=variant, source_id=inf.id)
                            if self.interaction_callback:
                                self.interaction_callback(inf.id, contact.id, location)
        
        # === PHASE 4 : Transitions d'états ===
        self._update_seir_states(current_day)

    def _infect_agent(self, agent, current_day, variant='alpha', source_id=None):
        """Infecte un agent susceptible (passage S → E)"""
        if source_id is not None:
            self.transmission_memory[agent.id] = source_id
        incubation_days = int(np.random.lognormal(
            mean=np.log(5.5),  # Médiane 5.5 jours
            sigma=0.4          # Écart-type réduit (plage 3-9 jours)
        ))
        incubation_days = max(3, min(9, incubation_days))
        
        self.agent_states[agent.id] = 'E'
        self.incubation_timers[agent.id] = incubation_days
        self.agent_infection_day[agent.id] = current_day
        agent.update_status('incubation')
        if random.random() < self.mutation_prob:
            # Sélection naturelle : variants plus transmissibles s'imposent
            if variant == 'alpha':
                new_variant = random.choices(
                    ['beta', 'gamma'],
                    weights=[0.6, 0.4]  # Beta plus compétitif
                )[0]
            else:
                new_variant = variant  # Variants mutés restent stables
            
            self.agent_variant[agent.id] = new_variant
            print(f"Mutation : {variant} → {new_variant} (agent {agent.id})")
        else:
            self.agent_variant[agent.id] = variant

        if source_id is not None:
            source_variant = self.agent_variant.get(source_id, 'alpha')
            
            # Beta/Gamma plus compétitifs (remplacent Alpha)
            if source_variant in ['beta', 'gamma']:
                self.agent_variant[agent.id] = source_variant
            elif random.random() < self.mutation_prob:
                # Mutation classique
                new_variant = random.choices(
                    ['beta', 'gamma'],
                    weights=[0.6, 0.4]
                )[0]
                self.agent_variant[agent.id] = new_variant
                print(f"Mutation : {source_variant} → {new_variant} (agent {agent.id})")
            else:
                self.agent_variant[agent.id] = source_variant

    def _update_seir_states(self, current_day):
        """
        Transitions SEIR avec état V (vacciné)
        
        États possibles :
        - S : Susceptible
        - E : Exposé (incubation)
        - I : Infectieux (symptomatique)
        - R : Guéri (immunité naturelle temporaire)
        - V : Vacciné (immunité permanente)
        """
        for agent in self.agents:
            state = self.agent_states.get(agent.id)
            
            # E → I
            if state == 'E':
                self.incubation_timers[agent.id] -= 1
                if self.incubation_timers[agent.id] <= 0:
                    self.agent_states[agent.id] = 'I'
                    agent.update_status('infecté')
                    
                    inf_days = int(np.random.lognormal(
                        np.log(self.infection_duration_mean), 0.3
                    ))
                    self.infection_timers[agent.id] = max(7, min(21, inf_days))
            
            # I → R
            elif state == 'I':
                self.infection_timers[agent.id] -= 1
                if self.infection_timers[agent.id] <= 0:
                    self.agent_states[agent.id] = 'R'
                    agent.update_status('immunisé')
            
            # R → S (perte immunité naturelle après 180 jours)
            elif state == 'R':
                days_since_recovery = current_day - self.agent_infection_day.get(agent.id, 0)
                
                if days_since_recovery > 2:  # 6 mois
                    current_variant = self.agent_variant.get(agent.id, 'alpha')
                    reinfection_prob = {
                        'alpha': 0.05,
                        'beta': 0.15,
                        'gamma': 0.20
                    }.get(current_variant, 0.10)
                    
                    recent_exposure = any(
                        self.agent_infection_day.get(fid, 0) > current_day - 7
                        for fid in agent.friends
                    )
                    
                    if recent_exposure and random.random() < reinfection_prob:
                        self.agent_states[agent.id] = 'S'
                        agent.update_status('sain')
                        # Nouvelle souche possible
                        if random.random() < 0.3:
                            self.agent_variant[agent.id] = random.choice(['beta', 'gamma'])

    def _initialize_heterogeneous_transmissibility(self):
        """
        Distribution gamma de transmissibilité (Lloyd-Smith et al. 2005)
        
        Principe : k<1 → 20% infectent 80% (loi de Pareto épidémique)
        
        Calibration COVID-19 (Endo et al. 2020) :
        - k ≈ 0.1 (forte dispersion)
        - 10% des cas = 80% des transmissions
        """        
        # Paramètres pour k≈0.15 (super-spreading)
        shape = 0.15  # k (dispersion parameter)
        scale = 1.0 / shape  # θ = 1/k
        
        self.agent_transmissibility = {}
        
        for agent in self.agents:
            # Tirer β individuel (gamma distribution)
            beta_individual = np.random.gamma(shape, scale)
            
            # Facteurs biologiques/comportementaux
            if agent.psychology == 'leader':
                beta_individual *= 2.5  # Leaders = super-spreaders sociaux
            elif agent.psychology == 'anxieux':
                beta_individual *= 0.3  # Isolement volontaire
            
            if agent.job in ['enseignant', 'commerçant']:
                beta_individual *= 1.8  # Exposition professionnelle
            
            # Âge (enfants moins contagieux)
            if agent.age < 12:
                beta_individual *= 0.5
            
            self.agent_transmissibility[agent.id] = beta_individual
        
        # Validation distribution
        betas = list(self.agent_transmissibility.values())
        betas_sorted = sorted(self.agent_transmissibility.items(), key=lambda x: x[1], reverse=True)
        superspreader_ids = [aid for aid, beta in betas_sorted[:len(self.agents)//10] if beta > 2.5]

        print(f"{len(superspreader_ids)} super-spreaders identifiés (β > 2.5)")
        self.superspreaders = set(superspreader_ids)
        print(f"Distribution β : médiane={np.median(betas):.2f}, "
            f"P90={np.percentile(betas, 90):.2f}")