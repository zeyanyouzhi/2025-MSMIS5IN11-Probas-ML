# Point d'entrée du projet

import sys
import os
import random
import importlib
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import model.agent
importlib.reload(model.agent)

from model.agent import Agent
from model.epidemic import EpidemicModel
from model.crowd_psychology import CrowdPsychology
from model.environment import CityEnvironment
from model.social_dynamics import SocialDefenseMechanisms
from model.network_builder import SocialNetworkBuilder
from model.metrics import compute_R0_dynamic, validate_epidemic_realism, compute_attack_rate
from model.mobility import MobilityManager
from model.monitoring import SimulationMonitor
from model.network_validation import NetworkValidator, EpidemicValidator
from model.visualisation import (
    plot_social_network,
    plot_psychology_distribution,
    plot_bubble_network,
    plot_optimized_social_network,
    export_social_network_to_gexf,
    plot_degree_distribution,
    plot_seir_dynamics,
    plot_degree_distribution_loglog,
    plot_r0_evolution,
    plot_superspreading_distribution,
    plot_variant_timeline,
    plot_epidemic_map,
    plot_epidemic_network_spread,
    export_social_network_to_gexf_exhaustif,
    plot_epidemic_community_spread,
    plot_epidemic_community_timeline,
    plot_interaction_heatmap,
    plot_seirv_dynamics, 
    plot_vaccination_impact
)

# Matrice d'interactions pour évolution des réseaux
interaction_matrix = {}

def record_interaction(agent1_id, agent2_id, location):
    """
    Enregistre une interaction entre deux agents pour le calcul de R0 et l'évolution du réseau.
    
    Args:
        agent1_id (int): ID du premier agent.
        agent2_id (int): ID du second agent.
        location (str): Lieu de l'interaction.
    """
    key = tuple(sorted([agent1_id, agent2_id]) + [location])
    interaction_matrix[key] = interaction_matrix.get(key, 0) + 1
    
if __name__ == "__main__":
    # Visualisation des réseaux sociaux pour un échantillon d'agents
    # Étape 1 : Vérification de l'import et initialisation
    print("Test import Agent :", Agent)
    jours_max = 20 # Nombre de jours à simuler
    # Étape 2 : Création dynamique d'agents
    x = 200  # nombre d'agents à simuler
    possible_psychologies = ['calme', 'anxieux', 'leader', 'rebelle', 'suiveur']
    agents = []
    # Initialisation avec au moins quelques agents infectés
    n_infecte_init = max(1, int(0.02 * x))  # 2% infectés au départ
    infecte_indices = set(random.sample(range(x), n_infecte_init))

    city = CityEnvironment()
    # Création des agents
    for i in range(x):
        psychology = random.choice(possible_psychologies)
        status = 'infecté' if i in infecte_indices else 'sain'
        age = random.randint(8, 80)
        passions_by_age = [
            (range(8, 18), ['jeux vidéo', 'sport', 'musique', 'dessin']),
            (range(18, 30), ['voyages', 'musique', 'sport', 'cinéma', 'jeux vidéo']),
            (range(30, 50), ['lecture', 'cuisine', 'randonnée', 'cinéma', 'voyages']),
            (range(50, 81), ['jardinage', 'lecture', 'cuisine', 'voyages', 'peinture'])
        ]
        for age_range, passions_list in passions_by_age:
            if age in age_range:
                passions = random.sample(passions_list, k=2)
                break
        else:
            passions = ['lecture']
        jobs = ['enseignant', 'ingénieur', 'médecin', 'étudiant', 'retraité', 'artiste', 'commerçant', 'ouvrier', 'cadre', 'infirmier']
        if 8 <= age <= 18:
            job = 'étudiant'
        else:
            job = random.choice([j for j in jobs if j != 'étudiant'])
        # Assigner quartiers aléatoires
        quarters = ['Nord', 'Sud', 'Est', 'Ouest']
        home_quarter = random.choice(quarters)

        # 70% travaillent dans même quartier, 30% ailleurs
        if random.random() < 0.7:
            work_quarter = home_quarter
        else:
            work_quarter = random.choice([q for q in quarters if q != home_quarter])

        agent = Agent(
            id=i,
            status=status,
            psychology=psychology,
            home_quarter=home_quarter,
            work_quarter=work_quarter,
            age=age,
            passions=passions,
            job=job,
            city_env=city
        )
        agents.append(agent)
        city.all_agents = agents

    def build_transmission_tree():
        """Construit l'arbre de transmission pour analyse (Primer style)"""
        G = nx.DiGraph()
        for target_id, source_id in epidemic.transmission_memory.items():
            G.add_edge(source_id, target_id)
        return G
    # Regroupement des agents par domicile
    home_dict = {}
    for agent in agents:
        home_dict.setdefault(agent.home_quarter, []).append(agent.id)

    # Création de familles réalistes (2 à 6 membres, plusieurs familles possibles par domicile)
    family_dict = {}  # agent_id -> set(family_ids)
    for home_quarter, members in home_dict.items():
        random.shuffle(members)
        families = []
        i = 0
        while i < len(members):
            remaining = len(members) - i
            if remaining < 2:
                # Dernier agent seul dans le domicile
                family = set(members[i:i+remaining])
                families.append(family)
                i += remaining
            else:
                n_family = random.randint(2, min(6, remaining))
                family = set(members[i:i+n_family])
                families.append(family)
                i += n_family
        for family in families:
            for agent_id in family:
                family_dict[agent_id] = family - {agent_id}

    for agent in agents:
        agent.family = family_dict.get(agent.id, set())
    work_dict = {}
    for agent in agents:
        work_dict.setdefault((agent.work_quarter, agent.job), []).append(agent.id)
    
    # Construction des réseaux sociaux réalistes
    network_builder = SocialNetworkBuilder(agents)
    
    # Réseau d'amis : modèle small-world (Watts-Strogatz)
    metrics = network_builder.build_complete_network(target_avg_degree=30, enable_validation=True)

    degree_validation = NetworkValidator(agents).validate_degree_distribution(agents=agents)
    print(f"\nValidation loi de puissance :")
    print(f"   - Gamma : {degree_validation['gamma']:.2f} (attendu : 2-3)")
    print(f"   - R² : {degree_validation['R2']:.3f} (attendu : >0.7)")
    print(f"   - Scale-free : {'' if degree_validation['is_scale_free'] else 'non validé'}")

    print("\nMétriques intermédiaires après construction :")
    degrees_temp = [len(a.friends) for a in agents]
    print(f"   - Degré moyen : {sum(degrees_temp)/len(degrees_temp):.1f}")
    print(f"   - Degré max : {max(degrees_temp)}")
    print(f"   - Agents avec 0 amis : {sum(1 for d in degrees_temp if d == 0)}")

    # Réseau de collègues : modèle scale-free (Barabási-Albert)
    network_builder.build_colleagues_network_scale_free(m=2)
    network_builder.apply_homophily_filtering()

    # Identifier les hubs (super-connecteurs)
    hubs = network_builder.identify_network_hubs()
    print(f"\nHubs identifiés :")
    print(f"   - Top connecteurs (degré) : {hubs['degree_hubs'][:5]}")
    print(f"   - Top ponts (betweenness) : {hubs['betweenness_hubs'][:5]}")

    # Détection de communautés (Fouloscopie : groupes sociaux)
    try:
        communities = network_builder.detect_communities()
        communities = network_builder.adjust_community_sizes(communities, taille_min=5, taille_max=20)
        communities = network_builder.merge_small_communities(communities, min_size=5)
        community_leaders = network_builder.assign_community_leaders(communities)
        history_by_community = {cid: {} for cid in communities.keys()}
        
        print(f"\n Communautés détectées : {len(communities)}")
        for comm_id, members in list(communities.items()):  # Afficher 5 premières
            for agent in agents:
                if agent.id in members:
                    agent.comm_id = comm_id
            leader_id = community_leaders.get(comm_id)
            leader = next((a for a in agents if a.id == leader_id), None)
            print(f"   - Communauté {comm_id} : {len(members)} membres, leader = Agent {leader_id} ({leader.psychology if leader else '?'})")
        
    except ImportError:
        print("\nModule python-louvain non installé, communautés non détectées")
        print("Installation : pip install python-louvain")
        communities = {}
        community_leaders = {}
        history_by_community = {}

    # Métriques de validation
    metrics = network_builder.compute_network_metrics()
    print(f"\nMétriques du réseau :")
    print(f"   - Coefficient de clustering : {metrics['clustering_coefficient']:.3f}")
    print(f"   - Longueur de chemin moyenne : {metrics['average_path_length']:.2f}")
    print(f"   - Coefficient small-world (σ) : {metrics['small_world_sigma']:.2f}" if metrics['small_world_sigma'] else "   - σ : N/A")
    print(f"   - Diamètre du réseau : {metrics['diameter']}")
    
    def is_neighbor(home1, home2, threshold=10):
        """
        Détermine si deux agents sont voisins.
        - Si home1/home2 sont des str (quartiers), voisins si identiques.
        - Si ce sont des tuples (current_locations), voisins si distance <= threshold.
        """
        if home1 is None or home2 is None:
            return False
        if isinstance(home1, str) and isinstance(home2, str):
            return home1 == home2
        if isinstance(home1, (tuple, list)) and isinstance(home2, (tuple, list)):
            dx = home1[0] - home2[0]
            dy = home1[1] - home2[1]
            return (dx**2 + dy**2)**0.5 <= threshold
        return False

    for agent in agents:
        agent.neighbors = set(
            [a.id for a in agents if a.id != agent.id and is_neighbor(agent.home_quarter, a.home_quarter)]
        )
    
    # Génération aléatoire de familles de sang (optionnel)
    blood_families = []
    remaining_agents = set(a.id for a in agents)
    while remaining_agents:
        n_blood = random.randint(1, 5)
        group = set(random.sample(list(remaining_agents), min(n_blood, len(remaining_agents))))
        blood_families.append(group)
        remaining_agents -= group
    blood_family_dict = {}
    for family in blood_families:
        for agent_id in family:
            blood_family_dict[agent_id] = family - {agent_id}
    for agent in agents:
        agent.blood_family = blood_family_dict.get(agent.id, set())

    #Affichage de chaque agent
    #for agent in agents:
        # print(agent)

    print("\n--- Les graphiques sont enregistrés dans le dossier outputs/ ---")
    # Un seul graphique par type de réseau (pour alléger)
    plot_social_network(agents, 'family', 'Réseau familial (échantillon)', 'network_famille.png', color_by_psy=True)
    plot_social_network(agents, 'friends', 'Réseau d’amis (échantillon)', 'network_amis.png', color_by_psy=True)
    plot_social_network(agents, 'colleagues', 'Réseau de collègues (échantillon)', 'network_collegues.png', color_by_psy=True)
    plot_social_network(agents, 'neighbors', 'Réseau de voisins (échantillon)', 'network_voisins.png', color_by_psy=True)
    # Répartition des profils psychologiques avant simulation
    plot_psychology_distribution(agents, 'Répartition des profils psychologiques (avant)', 'psychologie_avant.png', sample_size=20)

    # Mesures sanitaires : confinement et vaccination
    confinement = False
            
    # Assigner types de masques selon confiance
    for agent in agents:
        if agent.trust_level > 0.8:
            agent.mask_type = 'FFP2'
        elif agent.trust_level > 0.5:
            agent.mask_type = 'chirurgical'
        elif agent.trust_level > 0.3:
            agent.mask_type = 'tissu'
        else:
            agent.mask_type = None  # Pas de masque


    # Étape 3 : Initialisation des modèles
    epidemic = EpidemicModel(
        agents, 
        infection_prob=0.5,
        city_env=city, 
        interaction_callback=record_interaction
    )
    epidemic.set_network_hubs(hubs)

    #  VÉRIFICATION : Infectés initiaux
    print(f"\nVÉRIFICATION INFECTÉS INITIAUX :")
    initial_infected = [a for a in agents if a.status in ['infecté', 'incubation']]
    print(f"  - Nombre : {len(initial_infected)}")
    for inf in initial_infected[:3]:
        print(f"  - Agent {inf.id} : status={inf.status}, timer={epidemic.infection_timers.get(inf.id, 0)}")

    monitor = SimulationMonitor(agents, epidemic)
    # Vérifications initiales
    monitor.check_network_health()
    monitor.check_location_consistency()

    crowd = CrowdPsychology(agents)
    # Mécanismes de défense sociale
    social_defense = SocialDefenseMechanisms(agents)
    social_defense.form_social_bubbles()
    social_defense.identify_health_leaders()

    print(f"\n{len(social_defense.social_bubbles)} bulles sociales formées")
    print(f"{len(social_defense.health_leaders)} leaders de santé identifiés")
    plot_bubble_network(agents, social_defense.social_bubbles, 'bulles_sociales.png')
    plot_optimized_social_network(agents, social_defense.social_bubbles, 'bulles_sociales_optimisées.png')
    # Utiliser le premier identifiant de communauté détecté, ou 0 si non disponible
    comm_id = next(iter(communities.keys()), 0) if communities else 0
    export_social_network_to_gexf(agents, social_defense.social_bubbles, comm_id, 'reseau_social.gexf')

    # Étape 4 : Simulation multi-tours avec visualisation
    history_sain = []
    history_infecte = []
    history_immunise = []
    history_anxieux = []
    history_contagieux = []
    history_vaccines = []

    # Appeler après la création des agents
    network_builder.fix_bidirectional_links()

    spatial_check = NetworkValidator(agents).validate_spatial_consistency(agents=agents)
    network_check = NetworkValidator(agents).validate_network_consistency(agents)

    if not spatial_check['is_valid']:
        print("\nERREURS SPATIALES :")
        for err in spatial_check['errors'][:5]:
            print(f"  - {err}")
        raise RuntimeError("Incohérence spatiale détectée")

    if not network_check['is_valid']:
        print("\nERREURS RÉSEAU :")
        for err in network_check['errors'][:5]:
            print(f"  - {err}")
        raise RuntimeError("Réseau invalide")

    print("Validation complète réussie")

    print("\n--- Simulation sur plusieurs tours ---")
    old_prob = epidemic.infection_prob  # Initialisation avec une valeur float valide
    # Simulation sur 20 jours, 5 moments par jour
    times_of_day = ['morning', 'midday', 'afternoon', 'evening', 'night']

    # Principe de Pareto : 20% des agents ont 80% des contacts
    for agent in agents:
        contact_multiplier = 1.0
        if agent.psychology == 'leader':
            contact_multiplier = 3.0  # Leaders = 3x plus de contacts
        elif agent.job in ['commerçant', 'enseignant']:
            contact_multiplier = 2.5
        elif agent.psychology == 'anxieux':
            contact_multiplier = 0.4  # Anxieux = isolement
        
        agent.contact_rate = contact_multiplier

    # Tracking SEIR pour visualisation
    history_seir = {'S': [], 'E': [], 'I': [], 'R': [], 'V': []}
    history_r0 = []
    r0_by_community = {cid: {} for cid in communities.keys()}

    for day in range(1, jours_max + 1):
        if day % 4 == 0:
            export_social_network_to_gexf(agents, social_defense.social_bubbles, comm_id, f'reseau_social_jour_{day}.gexf')
        # Tous les 5 jours : évolution des réseaux sociaux
        if day % 5 == 0 and day > 0:  # Tous les 5 jours (plus stable)
            print(f"   Mise à jour des réseaux sociaux (jour {day})")
            network_builder.update_friendships_over_time(interaction_matrix, days_elapsed=5)
            interaction_matrix.clear()
            monitor.check_network_health()
            monitor.check_epidemic_progress(day)
            plot_epidemic_network_spread(agents, epidemic, day, f'epidemic_spread_day_{day}.png')
            plot_epidemic_community_spread(agents, epidemic, communities, day,f'epidemic_community_day{day}.png')

        day_of_week = 'weekend' if day % 7 in [6, 0] else 'weekday'
        print(f"\n=== JOUR {day} ({'Weekend' if day_of_week == 'weekend' else 'Semaine'}) ===")        
        for time in times_of_day:
            print(f"  {time}:")
            
            # Événement déclencheur tous les 5 jours au moment 'evening'
            if time == 'evening' and random.random() < 0.1:
                event_type = random.choice(['wedding', 'conference', 'concert'])
                participants = random.sample(agents, int(0.4 * len(agents)))  # 40% participent
                
                for agent in participants:
                    agent.current_location = 'event'
                    epidemic.infection_prob *= 2.0  # Transmission ++
                
                print(f"Événement : {event_type} ({len(participants)} participants)")

            mobility_manager = MobilityManager(city)

            if day % 10 == 0 and random.random() < 0.3:  # Tous les 10 jours, 30% chance
                event_type = random.choice(['concert', 'conference', 'match'])
                participants = random.sample(agents, int(0.8 * len(agents)))  # 80% participent
                event_location = city.get_random_location('gathering')
                
                print(f"Gros Événement : {event_type} ({len(participants)} participants)")
                
                for agent in participants:
                    agent.current_location = 'event'
                    epidemic.infection_prob *= 4.0  # Transmission +++

                # Propagation intensifiée (Lloyd-Smith 2005)
                epidemic.propagate(day, 'evening')

            # Déclencher vagues tous les 30 jours si R0 < 0.5
            if day % 30 == 0 and day > 30:
                r0 = compute_R0_dynamic(epidemic, day)
                if r0 < 0.5:
                    new_variant = f"variant_gen{day//30}"
                    epidemic.trigger_new_wave(day, variant_name=new_variant)
                    print(f"  🌊 Nouvelle vague déclenchée par {new_variant} (R0={r0:.2f})")

            # Déplacement selon l'heure
            for agent in agents:
                # CORRECTION : Confinement strict (agents anxieux/calmes restent home)
                if confinement and agent.psychology in ['anxieux', 'calme']:
                    agent.current_location = 'home'
                else:
                    # Mobilité normale (fonction retourne maintenant juste un str)
                    agent.current_location = mobility_manager.decide_location(
                        agent, time, day_of_week, agents, confinement_active=confinement
                    )
                    if agent.id in [34, 42]:  # Infectés initiaux (exemples)
                        print(f"Agent {agent.id} : current_location={agent.current_location}")
                
                # Mise à jour optionnelle du détail géographique
                if agent.current_location == 'home':
                    agent.current_location_detail = agent.home_quarter
                elif agent.current_location == 'work':
                    agent.current_location_detail = agent.work_quarter

            #  DEBUG LIEUX (supprimer après fix)
            if day <= 3:
                print(f"\n  DEBUG lieux (jour {day}, {time}) :")
                infectes = [a for a in agents if a.status in ['infecté', 'incubation']]
                for inf in infectes[:2]:  # 2 premiers infectés
                    lieu = getattr(inf, 'current_location', 'MANQUANT')
                    print(f"    Infecté {inf.id}: lieu={lieu}, "
                        f"amis={len(inf.friends)}, famille={len(inf.family)}")
                
                # Vérifier quelques sains
                sains_sample = [a for a in agents if a.status == 'sain'][:3]
                for sain in sains_sample:
                    lieu = getattr(sain, 'current_location', 'MANQUANT')
                    print(f"    Sain {sain.id}: lieu={lieu}")            

        # Calcul taux infection local (pour décisions)
        local_rates = {}
        for quarter in ['Nord', 'Sud', 'Est', 'Ouest']:
            agents_in_quarter = [a for a in agents if a.home_quarter == quarter]
            if agents_in_quarter:
                infected = sum(1 for a in agents_in_quarter 
                            if epidemic.agent_states.get(a.id) in ['E', 'I'])
                local_rates[quarter] = infected / len(agents_in_quarter)
        
        for time in times_of_day:
            # Décisions individuelles
            # S'assurer que mobility_manager est défini avant la boucle
            mobility_manager = MobilityManager(city)
            for agent in agents:
                # Vaccination (si campagne en cours et jour 10)
                if day % 10 == 0:                    
                    # Stratégie : Vacciner top 20% centralité
                    hub_scores = {}
                    for agent in agents:
                        degree_score = len(agent.friends) + len(agent.colleagues)
                        job_bonus = 2.0 if agent.job in ['enseignant', 'commerçant', 'médecin'] else 1.0
                        hub_scores[agent.id] = degree_score * job_bonus
                    
                    n_vaccines = int(0.20 * len(agents))
                    top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:n_vaccines]
                    
                    vaccinated = 0
                    for agent_id, score in top_hubs:
                        agent = next(a for a in agents if a.id == agent_id)
                        
                        #  UTILISER LA NOUVELLE FONCTION
                        success = epidemic.vaccinate_agent(agent, day, vaccine_type='pfizer')
                        if success:
                            vaccinated += 1
                            print("\nCAMPAGNE DE VACCINATION (immunité parfaite)")
                            print(f"    {vaccinated}/{n_vaccines} agents vaccinés")
                            print(f"     Immunité : PARFAITE (0% réinfection)")
                            print(f"    Couverture : {vaccinated/len(agents)*100:.1f}%")
                # Confinement
                local_rate = local_rates.get(agent.home_quarter, 0)
                if agent.decide_confinement(local_rate):
                    agent.current_location = 'home'
                else:
                    # Mobilité normale
                    agent.current_location = mobility_manager.decide_location(
                        agent, time, day_of_week, agents
                    )
            
            # Propagation épidémique
            epidemic.propagate(current_day=day, time_of_day=time)

            # Propagation des bonnes pratiques (tous les 2 jours)
            if day % 2 == 0:
                social_defense.propagate_health_behaviors()
                #social_defense.enforce_bubble_isolation(epidemic)
            
            # Restaurer probabilité après rassemblement
            if day % 5 == 0 and time == 'evening':
                epidemic.infection_prob = old_prob
            
            # Calcul et affichage de R0 dynamique
            if day % 5 == 0:  # Validation tous les 5 jours
                validate_epidemic_realism(epidemic, day)
                cohort_size = sum(1 for _, infection_day in epidemic.agent_infection_day.items() 
                                if day - 14 <= infection_day <= day - 7)
                r0 = compute_R0_dynamic(epidemic, day)
                print(f"   Cohorte J-14 à J-7 : {cohort_size} infectés → R0={r0:.2f}")

        # Psychologie une fois par jour (soir)
        crowd.update_behaviors()

        # Tous les 3 jours : influence des leaders de communauté
        if day % 3 == 0 and communities:
            crowd.apply_community_influence(communities, community_leaders)
            
            # Tous les 10 jours : détecter chambres d'écho
            if day % 10 == 0:
                echo_chambers = crowd.detect_echo_chambers(communities)
                if echo_chambers:
                    print(f"{len(echo_chambers)} chambres d'écho détectées (polarisation)")

        # Visualisation spatiale tous les 3 jours
        if day % 3 == 0:
            plot_epidemic_map(agents, epidemic, day, f'epidemic_map_day{day}.png')

        if day == 10:
            print("\nCAMPAGNE DE VACCINATION (ciblage optimal)")
            
            # Stratégie scientifique (Cohen et al. 2003 PRL) :
            # Vacciner top 20% centralité = 80% réduction transmission
            
            # 1. Calculer centralité combinée (degré + betweenness)
            hub_scores = {}
            for agent in agents:
                degree_score = len(agent.friends) + len(agent.colleagues)
                # Bonus métier à risque
                job_bonus = 2.0 if agent.job in ['enseignant', 'commerçant', 'médecin'] else 1.0
                hub_scores[agent.id] = degree_score * job_bonus
            
            # 2. Vacciner top 20% (40 agents sur 200)
            n_vaccines = int(0.20 * len(agents))
            top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:n_vaccines]
            
            vaccinated = 0
            for agent_id, score in top_hubs:
                agent = next(a for a in agents if a.id == agent_id)
                if agent.status == 'sain':
                    agent.update_status('immunisé')
                    epidemic.agent_states[agent.id] = 'R'
                    vaccinated += 1
            
            print(f"{vaccinated} agents vaccinés (stratégie hub-targeting)")
            print(f"Score moyen hubs : {sum(s for _, s in top_hubs)/len(top_hubs):.1f}")

        # Stats quotidiennes
        s = sum(1 for a in agents if epidemic.agent_states.get(a.id) == 'S')
        e = sum(1 for a in agents if epidemic.agent_states.get(a.id) == 'E')
        i = sum(1 for a in agents if epidemic.agent_states.get(a.id) == 'I')
        r = sum(1 for a in agents if epidemic.agent_states.get(a.id) == 'R')
        v = sum(1 for a in agents if epidemic.agent_states.get(a.id) == 'V')
        
        history_seir['S'].append(s)
        history_seir['E'].append(e)
        history_seir['I'].append(i)
        history_seir['R'].append(r)
        history_seir['V'].append(v)

        for cid, members in communities.items():
            stats = {
                'S': sum(1 for mid in members if epidemic.agent_states.get(mid) == 'S'),
                'E': sum(1 for mid in members if epidemic.agent_states.get(mid) == 'E'),
                'I': sum(1 for mid in members if epidemic.agent_states.get(mid) == 'I'),
                'R': sum(1 for mid in members if epidemic.agent_states.get(mid) == 'R'),
                'V': sum(1 for mid in members if epidemic.agent_states.get(mid) == 'V'),
            }
            history_by_community[cid][day] = stats        
        
        # R0 dynamique
        r0 = compute_R0_dynamic(epidemic, day)
        history_r0.append(r0)
        
        print(f"  S={s}, E={e}, I={i}, R={r}, V={v} | R0={r0:.2f}")
        
        # Validation tous les 10 jours
        if day % 10 == 0:
            validator = EpidemicValidator(epidemic, agents)
            validator.validate_all()

        # Stats quotidiennes
        n_sain = sum(1 for a in agents if a.status == 'sain')
        n_infecte = sum(1 for a in agents if a.status == 'infecté')
        n_immunise = sum(1 for a in agents if a.status == 'immunisé')
        n_anxieux = sum(1 for a in agents if a.psychology == 'anxieux')
        n_contagieux = sum(1 for a in agents if a.status in ['infecté', 'incubation'])
        n_vaccines = sum(1 for a in agents if epidemic.agent_states.get(a.id) == 'V')
        
        history_sain.append(n_sain)
        history_infecte.append(n_infecte)
        history_immunise.append(n_immunise)
        history_anxieux.append(n_anxieux)
        history_contagieux.append(n_contagieux)
        history_vaccines.append(n_vaccines)
        
        print(f"Sains: {n_sain}, Infectés: {n_infecte}, Immunisés: {n_immunise}, Anxieux: {n_anxieux}")
        for cid, members in communities.items():
            infected_in_comm = [a for a in agents if a.id in members and a.status == 'infecté']
            if infected_in_comm:
                r0_comm = compute_R0_dynamic(epidemic, day)
                r0_by_community[cid][day] = r0_comm

    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(history_sain, label='Sains')
    plt.plot(history_infecte, label='Infectés')
    plt.plot(history_immunise, label='Immunisés')
    plt.plot(history_contagieux, label='Contagieux', linestyle='--')
    plt.plot(history_vaccines, label='Vaccinés', linestyle=':')
    plt.xlabel('Tour de simulation')
    plt.ylabel('Nombre d’agents')
    plt.title('Évolution de l’épidémie')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/epidemic_evolution.png')
    plt.show()

    # Répartition des profils psychologiques après simulation (état final)
    plot_psychology_distribution(agents, 'Répartition des profils psychologiques (après)', 'psychologie_apres.png', sample_size=20)

    transmission_tree = build_transmission_tree()

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(transmission_tree, k=2)
    nx.draw_networkx_nodes(transmission_tree, pos, node_size=300, node_color='red', alpha=0.6)
    nx.draw_networkx_edges(transmission_tree, pos, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(transmission_tree, pos, font_size=10)
    plt.title("Arbre de transmission (qui a contaminé qui)")
    plt.savefig('Projet_IA_Grégoire_Dunglas/outputs/transmission_tree.png')
    plt.close()

    # Dans main.py après simulation
    reinfections = [a for a in agents if a.status == 'sain' and a.id in epidemic.transmission_memory]
    print(f" {len(reinfections)} agents réinfectés")

    contact_matrix = Counter()
    for key, count in interaction_matrix.items():
        id1, id2 = key[:2]  # On ignore le lieu ici
        a1 = next(a for a in agents if a.id == id1)
        a2 = next(a for a in agents if a.id == id2)
        # Extraire lieu dominant
        contact_matrix[(a1.current_location, a2.current_location)] += count

    print("\nMatrice de contacts :")
    for (loc1, loc2), count in contact_matrix.most_common(10):
        print(f"  {loc1} ↔ {loc2} : {count} interactions")

    # Analyse des chaînes de transmission (Primer style)
    def analyze_transmission_chains():
        chains = {}
        for target, source in epidemic.transmission_memory.items():
            chains.setdefault(source, []).append(target)
        
        # Identifier superspreaders (agents ayant contaminé >5 personnes)
        superspreaders = [(agent_id, len(targets)) for agent_id, targets in chains.items() if len(targets) > 5]
        superspreaders.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTOP 5 SUPERSPREADERS :")
        for agent_id, count in superspreaders[:5]:
            agent = next(a for a in agents if a.id == agent_id)
            print(f"   Agent {agent_id} ({agent.psychology}, {agent.job}) : {count} contaminations")

    analyze_transmission_chains()
    plot_interaction_heatmap(interaction_matrix, agents, 'interaction_heatmap.png')

    # Comparaison vaccination ciblée vs aléatoire
    print("\nAnalyse de l'efficacité de la vaccination ciblée :")
    print(f"   - Nombre final d'infectés : {history_infecte[-1]}")
    print(f"   - Pic d'infection : {max(history_infecte)} (tour {history_infecte.index(max(history_infecte))})")
    print(f"   - Immunisés finaux : {history_immunise[-1]}")
    
    # Validation scientifique des réseaux
    validator = NetworkValidator(agents)
    validator.validate_small_world()
    validation_results = validator.run_all_validations()    
    export_social_network_to_gexf(agents, social_defense.social_bubbles, comm_id, 'reseau_social_final.gexf')

    # Appeler après les autres plots
    plot_degree_distribution(agents, 'degree_distribution.png')
    plot_degree_distribution_loglog(agents, 'degree_distribution_loglog.png')
        # Visualisations finales

    plot_seir_dynamics(history_seir)
    plot_r0_evolution(history_r0)

    # Distribution super-spreaders
    transmission_counts = [
        len([t for t, s in epidemic.transmission_memory.items() if s == a.id])
        for a in agents
    ]
    plot_superspreading_distribution(transmission_counts)
    plot_variant_timeline(epidemic, filename='variant_evolution.png', jours_max=jours_max+5)
    export_social_network_to_gexf_exhaustif(agents, social_defense.social_bubbles, comm_id=1, filename="reseau_social_exhaustif.gexf")
    plot_epidemic_community_timeline(agents, epidemic, communities, history_by_community, 'epidemic_community_timeline.png')

    print("BILAN VACCINATION")

    total_vaccinated = len(epidemic.vaccinated_agents)
    never_infected = sum(1 for aid in epidemic.vaccinated_agents 
                        if aid not in epidemic.transmission_memory)

    print(f"Total vaccinés : {total_vaccinated} ({total_vaccinated/len(agents)*100:.1f}%)")
    print(f"Jamais infectés (grâce vaccin) : {never_infected}")
    print(f"Efficacité réelle : {never_infected/total_vaccinated*100:.1f}%")

    # Calcul infections évitées (contre-factuel)
    expected_infections_without_vaccine = total_vaccinated * compute_attack_rate(agents)
    infections_avoided = int(expected_infections_without_vaccine)
    print(f"Infections évitées (estimé) : {infections_avoided}")

    # Graphique SEIR-V standard
    plot_seirv_dynamics(history_seir, vaccination_day=10-1)

    # Graphique impact détaillé
    plot_vaccination_impact(history_seir, vaccination_day=10-1)