"""
Module de visualisation scientifique
"""

# Import statements moved to the top
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from scipy.stats import linregress
import random
import matplotlib.cm as cm
from matplotlib.lines import Line2D

def plot_seir_dynamics(history, filename='seir_dynamics.png'):
    """
    Graphique SEIR classique avec zones colorées
    
    Inspired by : He et al. (2020), Nature Medicine
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    days = range(len(history['S']))
    
    # Aires empilées
    ax.fill_between(days, 0, history['S'], label='Susceptible', color='#3498db', alpha=0.7)
    ax.fill_between(days, history['S'], 
                     [s+e for s,e in zip(history['S'], history['E'])],
                     label='Exposed', color='#f39c12', alpha=0.7)
    ax.fill_between(days, [s+e for s,e in zip(history['S'], history['E'])],
                     [s+e+i for s,e,i in zip(history['S'], history['E'], history['I'])],
                     label='Infectious', color='#e74c3c', alpha=0.7)
    ax.fill_between(days, [s+e+i for s,e,i in zip(history['S'], history['E'], history['I'])],
                     [s+e+i+r for s,e,i,r in zip(history['S'], history['E'], history['I'], history['R'])],
                     label='Recovered', color='#2ecc71', alpha=0.7)
    
    ax.set_xlabel('Jours', fontsize=14)
    ax.set_ylabel('Nombre d\'agents', fontsize=14)
    ax.set_title('Dynamique SEIR de l\'épidémie', fontsize=16)
    ax.legend(loc='right', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    print(f" SEIR sauvegardé : Projet_IA_Grégoire_Dunglas/outputs/{filename}")

def plot_r0_heatmap(r0_by_community, day, filename='r0_heatmap.png'):
    """
    Heatmap R₀ par communauté (détection clusters actifs)
    
    Source : Pastor-Satorras & Vespignani (2001)
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Matrice communautés x temps
    communities = sorted(r0_by_community.keys())
    data = [[r0_by_community[c].get(d, 0) for d in range(day+1)] for c in communities]
    
    sns.heatmap(data, cmap='RdYlGn_r', vmin=0, vmax=3, 
                xticklabels=[str(i) for i in range(0, day+1, 5)],
                yticklabels=communities,
                cbar_kws={'label': 'R₀ effectif'},
                ax=ax)
    
    ax.set_xlabel('Jour', fontsize=12)
    ax.set_ylabel('Communauté', fontsize=12)
    ax.set_title(f'R₀ par communauté (jour {day})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()

def plot_r0_evolution(r0_history, filename='r0_evolution.png'):
    """
    Évolution du R0 effectif dans le temps
    
    Source : Fraser et al. (2009), Science
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = range(len(r0_history))
    ax.plot(days, r0_history, linewidth=2, color='#e74c3c', label='R₀ effectif')
    
    # Ligne R0=1 (seuil épidémique)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Seuil R₀=1')
    ax.fill_between(days, 0, 1, alpha=0.1, color='green', label='Contrôle')
    ax.fill_between(days, 1, max(r0_history), alpha=0.1, color='red', label='Croissance')
    
    ax.set_xlabel('Jours', fontsize=14)
    ax.set_ylabel('R₀ effectif', fontsize=14)
    ax.set_title('Évolution du nombre de reproduction', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()

def plot_superspreading_distribution(transmission_counts, filename='superspreaders.png'):
    """
    Distribution des transmissions (loi de Pareto)
    
    Source : Lloyd-Smith et al. (2005), Nature
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1 : Histogramme
    ax1.hist(transmission_counts, bins=range(0, max(transmission_counts)+2), 
             alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Nombre de transmissions secondaires', fontsize=12)
    ax1.set_ylabel('Nombre d\'agents', fontsize=12)
    ax1.set_title('Distribution des super-spreaders', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # Graphique 2 : Courbe de Lorenz (inégalité)
    sorted_counts = sorted(transmission_counts, reverse=True)
    cumsum = np.cumsum(sorted_counts) / np.sum(sorted_counts)
    
    ax2.plot(np.linspace(0, 1, len(cumsum)), cumsum, linewidth=2, color='#e74c3c')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Égalité parfaite')
    ax2.fill_between(np.linspace(0, 1, len(cumsum)), cumsum, np.linspace(0, 1, len(cumsum)),
    alpha=0.3, color='red')
    ax2.set_xlabel('Proportion d\'agents (cumulée)', fontsize=12)
    ax2.set_ylabel('Proportion de transmissions (cumulée)', fontsize=12)
    ax2.set_title('Courbe de Lorenz (inégalité de transmission)', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()

"""
Visualisation de la propagation des variants
"""

def plot_variant_timeline(epidemic_model, filename='variant_evolution.png', jours_max=20):
    """
    Évolution des variants dans le temps
    
    Source : Tegally et al. (2021) - Tracking SARS-CoV-2 variants
    """
    variant_history = {}  # {day: {variant: count}}
    
    for day in range(1, jours_max + 1):
        day_variants = {}
        for agent_id, infection_day in epidemic_model.agent_infection_day.items():
            if infection_day <= day:
                variant = epidemic_model.agent_variant.get(agent_id, 'alpha')
                day_variants[variant] = day_variants.get(variant, 0) + 1
        variant_history[day] = day_variants
    
    # Tracer
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'alpha': '#3498db', 'beta': '#e74c3c', 'gamma': '#f39c12'}
    
    for variant in ['alpha', 'beta', 'gamma']:
        counts = [variant_history.get(d, {}).get(variant, 0) 
                 for d in sorted(variant_history.keys())]
        ax.plot(sorted(variant_history.keys()), counts, 
               label=variant.capitalize(), color=colors[variant], linewidth=2)
    
    ax.set_xlabel('Jour', fontsize=14)
    ax.set_ylabel('Nombre d\'infectés', fontsize=14)
    ax.set_title('Évolution des variants viraux', fontsize=16)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    print(f" Variants sauvegardés : Projet_IA_Grégoire_Dunglas/outputs/{filename}")

def plot_social_network(agents, relation, title, filename, sample_size=20, color_by_psy=False):
    """
    Affiche et enregistre un graphique du réseau social (relation) pour un échantillon d'agents.
    Les couleurs représentent les profils psychologiques.
    """
    G3 = nx.Graph()
    sample_agents = random.sample(agents, min(sample_size, len(agents)))
    color_map = []
    psy_colors = {'calme': 'lightblue', 'anxieux': 'orange', 'leader': 'green', 'rebelle': 'red', 'suiveur': 'purple'}
    for agent in sample_agents:
        G3.add_node(agent.id, label=f"{agent.id}\n{agent.psychology}")
        if color_by_psy:
            color_map.append(psy_colors.get(agent.psychology, 'grey'))
        for other_id in getattr(agent, relation):
            if other_id in [a.id for a in sample_agents]:
                G3.add_edge(agent.id, other_id)
    pos = nx.spring_layout(G3)
    labels = nx.get_node_attributes(G3, 'label')
    plt.figure(figsize=(10, 8))
    if color_by_psy:
        nx.draw(G3, pos, with_labels=True, labels=labels, node_size=900, node_color=color_map, font_size=10, edge_color='#888', linewidths=2)
    else:
        nx.draw(G3, pos, with_labels=True, labels=labels, node_size=900, node_color='skyblue', font_size=10, edge_color='#888', linewidths=2)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()

def plot_degree_distribution(agents, filename):
    """
    Graphe de la distribution des degrés (nombre d'amis par agent)
    Permet de vérifier la propriété scale-free (loi de puissance)
    """
    degrees = [len(a.friends) for a in agents]
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(0, max(degrees)+2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Nombre d\'amis', fontsize=12)
    plt.ylabel('Nombre d\'agents', fontsize=12)
    plt.title(f'Distribution des degrés (moy={sum(degrees)/len(degrees):.1f})', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    print(f" Distribution des degrés sauvegardée : outputs/{filename}")

def plot_degree_distribution_loglog(agents, filename):
    """
    Graphique log-log pour vérifier loi de puissance (scale-free).
    Méthode Clauset et al. (2009) - robuste aux valeurs extrêmes
    """
    degrees = [len(a.friends) for a in agents]
    degree_counts = Counter(degrees)
    
    #  FILTRAGE STRICT : k_min=5 (Clauset recommande ≥3)
    valid_degrees = {k: v for k, v in degree_counts.items() if k >= 5 and v > 0}
    
    if len(valid_degrees) < 5:
        print("Données insuffisantes pour fit scale-free")
        return
    
    #  Calcul P(k) = fraction des nœuds avec degré k
    total_nodes = len(agents)
    x = np.array(sorted(valid_degrees.keys()), dtype=float)
    y = np.array([valid_degrees[k] / total_nodes for k in x], dtype=float)
    
    #  Régression log-log (MLE Clauset)
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept, r_value, p_value, std_err = map(float, linregress(log_x, log_y))
    gamma = -float(slope) + 1  #  γ = -α + 1 (convention)
    r2 = r_value**2
    
    #  Tracer
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(x, y, 'o', markersize=8, alpha=0.7, color='steelblue', label='Données')
    
    # Droite d'ajustement
    fit_y = np.exp(intercept) * x**slope
    ax.loglog(x, fit_y, '--', color='red', linewidth=2, 
             label=f'γ={gamma:.2f}, R²={r2:.3f}')
    
    ax.set_xlabel('Degré k (log)', fontsize=14)
    ax.set_ylabel('P(k) (log)', fontsize=14)
    ax.set_title('Distribution des degrés (échelle log-log)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    print(f" Distribution log-log : γ={gamma:.2f}, R²={r2:.3f}")

def plot_psychology_distribution(agents, title, filename, sample_size=20):
    """
    Affiche et enregistre la répartition des profils psychologiques pour un échantillon d'agents.
    """
    sample_agents = random.sample(agents, min(sample_size, len(agents)))
    psy_list = [a.psychology for a in sample_agents]
    psy_types = ['calme', 'anxieux', 'leader', 'rebelle', 'suiveur']
    counts = [psy_list.count(pt) for pt in psy_types]
    plt.figure(figsize=(7, 4))
    plt.bar(psy_types, counts, color=['lightblue', 'orange', 'green', 'red', 'purple'])
    plt.title(title)
    plt.ylabel('Nombre d’agents')
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()

def plot_bubble_network(agents, bubbles, filename):
    G = nx.Graph()
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(bubbles))]
    node_colors = {}
    # Attribution des couleurs par bulle
    for i, bubble in enumerate(bubbles):
        for agent_id in bubble:
            if agent_id not in node_colors:
                node_colors[agent_id] = colors[i]
    color_list = []
    for agent_id in sorted([a if isinstance(a, int) else a.id for a in node_colors.keys()]):
        G.add_node(agent_id, label=str(agent_id))
        color_list.append(node_colors[agent_id])
        # Liens internes à la bulle
        for bubble in bubbles:
            if agent_id in bubble:
                for other_id in bubble:
                    if other_id != agent_id and other_id in node_colors:
                        G.add_edge(agent_id, other_id)
                break
    if len(G.nodes()) == 0:
        print("Aucune bulle sociale à visualiser")
        return
    plt.figure(figsize=(18, 14))  # Taille augmentée
    pos = nx.spring_layout(G, k=1.5, iterations=120)  # Espacement augmenté
    nx.draw(G, pos, node_color=color_list, node_size=700,
            with_labels=True, labels={n: str(n) for n in G.nodes()}, font_size=14, edge_color='lightgray', alpha=0.8)
    plt.title(f'Bulles sociales de protection ({len(bubbles)} bulles, {len(G.nodes())} agents)', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    print(f"Graphique des bulles sauvegardé : outputs/{filename}")

def plot_optimized_social_network(agents, bubbles, filename):
    G = nx.Graph()

    # Ajouter les nœuds avec leurs attributs
    # Ajout des agents au graphe avec leurs caractéristiques
    for agent in agents:
        G.add_node(
            agent.id,
            status=agent.status,
            psychology=agent.psychology,
            size=1000 if agent.psychology == "leader" else 500
        )

    # Ajouter les arêtes (relations dans les bulles)
    # Ajout des connexions entre agents dans les groupes
    for bubble in bubbles:
        bubble_list = list(bubble)
        for i in range(len(bubble_list)):
            for j in range(i + 1, len(bubble_list)):
                agent_id = bubble_list[i]
                other_id = bubble_list[j]
                if G.has_node(agent_id) and G.has_node(other_id):
                    G.add_edge(agent_id, other_id)

    if len(G.nodes()) == 0:
        print("Aucune bulle sociale à visualiser")
        return

    # Calculer la centralité de degré
    degree_centrality = nx.degree_centrality(G)
    node_sizes = [3000 * degree_centrality[n] for n in G.nodes()]

    # Définir les couleurs en fonction du statut (sain/infecté)
    node_colors = []
    for n in G.nodes():
        if G.nodes[n]["status"] == "sain":
            node_colors.append("lightgreen")
        else:
            node_colors.append("salmon")

    # Définir les bordures en fonction de la psychologie (leader = bordure rouge)
    node_borders = []
    for n in G.nodes():
        if G.nodes[n]["psychology"] == "leader":
            node_borders.append("red")
        else:
            node_borders.append("gray")

    # Discurrent_location optimisée avec kamada_kawai_layout
    pos = nx.kamada_kawai_layout(G)

    # Dessiner le graphe
    fig, ax = plt.subplots(figsize=(20, 16))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_borders,
        linewidths=2,
        alpha=0.9,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.2, ax=ax)
    nx.draw_networkx_labels(
        G, pos,
        labels={n: str(n) for n in G.nodes() if degree_centrality[n] > 0.05},
        font_size=10,
        ax=ax
    )

    # Légende pour les couleurs et bordures
    ax.scatter([], [], c="lightgreen", label="Sain", s=100)
    ax.scatter([], [], c="salmon", label="Infecté", s=100)
    ax.scatter([], [], edgecolors="red", facecolors="none", label="Leader", s=100, linewidths=2)
    ax.legend(scatterpoints=1, frameon=False, labelspacing=1, loc="upper right")

    # Ajouter une colorbar pour la centralité de degré
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=min(node_sizes), vmax=max(node_sizes)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Centralité de degré')

    plt.title(f'Bulles sociales de protection ({len(bubbles)} bulles, {len(G.nodes())} agents)', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    print(f"Graphique optimisé sauvegardé : outputs/{filename}")

def export_social_network_to_gexf(agents, bubbles, comm_id, filename="reseau_social.gexf"):
    """
    Exporte réseau social en GEXF optimisé pour Gephi
    
    Améliorations :
    - Routine segmentée (6 attributs simples au lieu d'un dict)
    - Lieux favoris limités aux 6 types courants
    - Validation stricte des types
    """
    G = nx.Graph()

    # Métadonnées
    G.graph['creator'] = 'IA Epidemic Simulator v2.0'
    G.graph['description'] = 'Réseau social avec routines et lieux favoris'
    G.graph['date'] = str(__import__('datetime').date.today())

    # Ajouter nœuds avec attributs validés
    for agent in agents:
        # ========== ROUTINE SEGMENTÉE ==========
        routine = getattr(agent, 'weekly_routine', None)
        routine_attrs = {}
        
        if routine and isinstance(routine, dict):
            # Extraire segments clés (6 attributs au lieu de 42)
            for day_type in ['weekday', 'weekend']:
                for moment in ['morning', 'evening']:  # Seulement matin/soir
                    key = f"routine_{day_type}_{moment}"
                    value = None
                    if day_type in routine and isinstance(routine[day_type], dict):
                        value = routine[day_type].get(moment)
                    routine_attrs[key] = str(value) if value else ''
        
        # ========== LIEUX FAVORIS (6 TYPES MAX) ==========
        fav_places = getattr(agent, 'favorite_places', None)
        fav_attrs = {}
        
        if fav_places and isinstance(fav_places, dict):
            # Limiter aux 6 types courants
            core_types = ['restaurant', 'cafe', 'supermarche', 'gym', 'parc', 'cinema']
            for place_type in core_types:
                value = fav_places.get(place_type)
                if value and value != 'None':
                    fav_attrs[f"fav_{place_type}"] = str(value)
                else:
                    fav_attrs[f"fav_{place_type}"] = ''
        
        # ========== ATTRIBUTS PRINCIPAUX ==========
        node_attrs = {
            'prénom': str(getattr(agent, 'prenom', '')),
            'nom': str(getattr(agent, 'nom', '')),
            'label': str(agent.id),
            'status': str(getattr(agent, 'status', '')),
            'psychology': str(getattr(agent, 'psychology', '')),
            'age': str(getattr(agent, 'age', '')),
            'job': str(getattr(agent, 'job', '')),
            'passions': ", ".join(getattr(agent, 'passions', [])),
            'home_quarter': str(getattr(agent, 'home_quarter', '')),
            'work_quarter': str(getattr(agent, 'work_quarter', '')),
            'community_id': str(getattr(agent, 'comm_id', '')),
            'num_friends': str(len(getattr(agent, 'friends', []))),
            'num_colleagues': str(len(getattr(agent, 'colleagues', []))),
            'num_family': str(len(getattr(agent, 'family', []))),
            'is_leader': '1' if getattr(agent, 'psychology', '') == 'leader' else '0',
        }
        
        # Fusionner tous les attributs
        all_attrs = {**node_attrs, **routine_attrs, **fav_attrs}
        
        #  VALIDATION : Supprimer valeurs None/vides
        safe_attrs = {k: v for k, v in all_attrs.items() if v not in [None, '', 'None']}
        
        G.add_node(agent.id, **safe_attrs)

    # Ajouter arêtes
    for bubble_idx, bubble in enumerate(bubbles):
        bubble_list = list(bubble)
        for i in range(len(bubble_list)):
            for j in range(i + 1, len(bubble_list)):
                agent_id = bubble_list[i]
                other_id = bubble_list[j]
                
                if G.has_node(agent_id) and G.has_node(other_id):
                    edge_attrs = {
                        'weight': '1',
                        'relation': 'social_bubble',
                        'bubble_id': str(bubble_idx),
                        'community_id': str(comm_id),
                    }
                    G.add_edge(agent_id, other_id, **edge_attrs)

    # Exporter
    nx.write_gexf(G, f"Projet_IA_Grégoire_Dunglas/outputs/{filename}")
    print(f" GEXF exporté : Projet_IA_Grégoire_Dunglas/outputs/{filename}")
    print(f"   - {G.number_of_nodes()} nœuds")
    print(f"   - {G.number_of_edges()} arêtes")
    print(f"   - Attributs/nœud : {len(list(G.nodes(data=True))[0][1]) if G.number_of_nodes() > 0 else 0}")

def create_epidemic_animation(agents):
    """Animation du déplacement des agents et propagation"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Dessiner agents
        scatters = []
        for agent in agents:
            color = {'sain': 'blue', 'infecté': 'red', 'immunisé': 'green', 'incubation': 'orange'}[agent.status]
            scatter = ax.scatter(agent.current_location[0], agent.current_location[1], c=color, s=50, alpha=0.6)
            scatters.append(scatter)
        
        ax.set_title(f"Jour {frame // 5 + 1}, Moment {frame % 5}")
        return scatters
    
    anim = FuncAnimation(fig, update, frames=100, interval=200, blit=False)
    anim.save('Projet_IA_Grégoire_Dunglas/outputs/epidemic_animation.gif', writer='pillow')
    plt.close()

"""
Visualisation spatiale de la propagation épidémique
"""

def plot_epidemic_map(agents, epidemic_model, day, filename='epidemic_map.png'):
    """
    Carte choroplèthe animée de la propagation par quartier
    
    Source : Balcan & Vespignani (2011) - Spatial spread modeling
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Définir positions quartiers (grille 2x2)
    quarter_positions = {
        'Nord': (0, 1),
        'Sud': (0, 0),
        'Est': (1, 0),
        'Ouest': (1, 1)
    }
    
    # Compter infectés par quartier
    quarter_stats = defaultdict(lambda: {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'V': 0})
    
    for agent in agents:
        state = epidemic_model.agent_states.get(agent.id, 'S')
        quarter_stats[agent.home_quarter][state] += 1
    
    # Dessiner quartiers
    for quarter, (x, y) in quarter_positions.items():
        stats = quarter_stats[quarter]
        total = sum(stats.values())
        
        if total == 0:
            continue
        
        # Couleur selon prévalence
        prevalence = (stats['E'] + stats['I']) / total
        color = plt.get_cmap('Reds')(prevalence)
        
        # Rectangle quartier
        rect = patches.Rectangle(
            (x*10, y*10), 9, 9,
            linewidth=2,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Annotations
        ax.text(
            x*10 + 4.5, y*10 + 7,
            quarter,
            ha='center',
            fontsize=16,
            fontweight='bold'
        )
        
        ax.text(
            x*10 + 4.5, y*10 + 4.5,
            f"I: {stats['I']}\nE: {stats['E']}\nR: {stats['R']}, V: {stats['V']}",
            ha='center',
            fontsize=12
        )
        
        ax.text(
            x*10 + 4.5, y*10 + 2,
            f"Prévalence: {prevalence*100:.1f}%",
            ha='center',
            fontsize=10,
            style='italic'
        )
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title(f'Propagation spatiale - Jour {day}', fontsize=18)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Reds'), norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prévalence (I+E)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    
    print(f" Carte épidémique sauvegardée : outputs/{filename}")

def plot_epidemic_network_spread(agents, epidemic, day, filename='network_spread.png'):
    """
    Visualise propagation sur le réseau social (style Primer)
    
    - Nœuds colorés par état épidémique
    - Arêtes de transmission en rouge
    - Taille proportionnelle à la centralité
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    
    # Construire graphe
    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id, status=epidemic.agent_states.get(agent.id, 'S'))
        for friend_id in agent.friends:
            G.add_edge(agent.id, friend_id, transmission=False)
    
    # Marquer arêtes de transmission
    for target_id, source_id in epidemic.transmission_memory.items():
        if G.has_edge(source_id, target_id):
            G[source_id][target_id]['transmission'] = True
    
    # Layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Couleurs
    node_colors = []
    for node in G.nodes():
        state = G.nodes[node]['status']
        colors_map = {'S': '#3498db', 'E': '#f39c12', 'I': '#e74c3c', 'R': '#2ecc71', 'V': '#9b59b6'}
        node_colors.append(colors_map.get(state, 'gray'))
    
    # Taille (centralité)
    centrality = nx.betweenness_centrality(G)
    node_sizes = [500 + 2000 * centrality[n] for n in G.nodes()]
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Arêtes normales
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['transmission']]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, alpha=0.1, width=0.5, ax=ax)
    
    # Arêtes de transmission
    trans_edges = [(u, v) for u, v, d in G.edges(data=True) if d['transmission']]
    nx.draw_networkx_edges(G, pos, edgelist=trans_edges, edge_color='red', 
                           width=2, alpha=0.6, arrows=True, arrowsize=15, ax=ax)
    
    # Nœuds
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax)
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Susceptible'),
        Patch(facecolor='#f39c12', label='Exposé'),
        Patch(facecolor='#e74c3c', label='Infectieux'),
        Patch(facecolor='#2ecc71', label='Guéri'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_title(f'Propagation épidémique - Jour {day}', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}', dpi=150)
    plt.close()
    
    print(f" Réseau de propagation sauvegardé : outputs/{filename}")

import networkx as nx
from datetime import datetime

def export_social_network_to_gexf_exhaustif(agents, bubbles, comm_id, filename="reseau_social_exhaustif.gexf"):
    """
    Export GEXF EXHAUSTIF avec TOUS les attributs :
    - Routine complète (10 moments : 5 weekday + 5 weekend)
    - Lieux favoris complets (6 types)
    - Métadonnées agent
    - Réseaux sociaux (counts)
    
    Compatible Gephi avec attributs structurés
    """
    G = nx.Graph()

    # ========== MÉTADONNÉES GRAPHE ==========
    G.graph['creator'] = 'Epidemic Simulator v2.0 - Exhaustive Export'
    G.graph['description'] = 'Réseau social avec routine complète et lieux favoris'
    G.graph['date'] = str(datetime.now().date())
    G.graph['num_agents'] = str(len(agents))
    G.graph['num_bubbles'] = str(len(bubbles))
    G.graph['community_id'] = str(comm_id)

    # ========== AJOUT NŒUDS ==========
    for agent in agents:
        node_attrs = {}
        
        # --- IDENTITÉ ---
        node_attrs['label'] = str(agent.id)
        node_attrs['agent_id'] = str(agent.id)
        node_attrs['prénom'] = str(getattr(agent, 'prenom', ''))
        node_attrs['nom'] = str(getattr(agent, 'nom', ''))
        
        # --- DÉMOGRAPHIE ---
        node_attrs['age'] = str(getattr(agent, 'age', ''))
        node_attrs['job'] = str(getattr(agent, 'job', ''))
        node_attrs['passions'] = ", ".join(getattr(agent, 'passions', []))
        
        # --- LOCALISATION ---
        node_attrs['home_quarter'] = str(getattr(agent, 'home_quarter', ''))
        node_attrs['work_quarter'] = str(getattr(agent, 'work_quarter', ''))
        node_attrs['current_location'] = str(getattr(agent, 'current_location', ''))
        
        # --- PSYCHOLOGIE ---
        node_attrs['psychology'] = str(getattr(agent, 'psychology', ''))
        node_attrs['profile'] = str(getattr(agent, 'profile', ''))
        node_attrs['trust_level'] = str(round(getattr(agent, 'trust_level', 0), 2))
        node_attrs['anxiety_level'] = str(round(getattr(agent, 'anxiety_level', 0), 2))
        node_attrs['vaccine_hesitancy'] = str(round(getattr(agent, 'vaccine_hesitancy', 0), 2))
        node_attrs['compliance_to_rules'] = str(round(getattr(agent, 'compliance_to_rules', 0), 2))
        
        # --- STATUT ÉPIDÉMIQUE ---
        node_attrs['status'] = str(getattr(agent, 'status', ''))
        node_attrs['symptom_severity'] = str(getattr(agent, 'symptom_severity', ''))
        
        # AJOUT : STATUT VACCINAL
        node_attrs['is_vaccinated'] = '1' if getattr(agent, 'is_vaccinated', False) else '0'
        node_attrs['vaccination_day'] = str(getattr(agent, 'vaccination_day', ''))
        node_attrs['vaccine_type'] = str(getattr(agent, 'vaccine_type', ''))
        
        # --- RÉSEAUX SOCIAUX (COUNTS) ---
        node_attrs['num_friends'] = str(len(getattr(agent, 'friends', [])))
        node_attrs['num_colleagues'] = str(len(getattr(agent, 'colleagues', [])))
        node_attrs['num_family'] = str(len(getattr(agent, 'family', [])))
        node_attrs['num_neighbors'] = str(len(getattr(agent, 'neighbors', [])))
        node_attrs['num_connaissances'] = str(len(getattr(agent, 'connaissances', [])))
        
        # --- COMMUNAUTÉ ---
        node_attrs['community_id'] = str(getattr(agent, 'comm_id', ''))
        node_attrs['is_leader'] = '1' if getattr(agent, 'psychology', '') == 'leader' else '0'
        
        # ========== ROUTINE COMPLÈTE (10 ATTRIBUTS) ==========
        routine = getattr(agent, 'weekly_routine', None)
        
        if routine and isinstance(routine, dict):
            # WEEKDAY (5 moments)
            if 'weekday' in routine and isinstance(routine['weekday'], dict):
                node_attrs['routine_weekday_morning'] = str(routine['weekday'].get('morning', ''))
                node_attrs['routine_weekday_midday'] = str(routine['weekday'].get('midday', ''))
                node_attrs['routine_weekday_afternoon'] = str(routine['weekday'].get('afternoon', ''))
                node_attrs['routine_weekday_evening'] = str(routine['weekday'].get('evening', ''))
                node_attrs['routine_weekday_night'] = str(routine['weekday'].get('night', ''))
            else:
                # Fallback si structure invalide
                for moment in ['morning', 'midday', 'afternoon', 'evening', 'night']:
                    node_attrs[f'routine_weekday_{moment}'] = ''
            
            # WEEKEND (5 moments)
            if 'weekend' in routine and isinstance(routine['weekend'], dict):
                node_attrs['routine_weekend_morning'] = str(routine['weekend'].get('morning', ''))
                node_attrs['routine_weekend_midday'] = str(routine['weekend'].get('midday', ''))
                node_attrs['routine_weekend_afternoon'] = str(routine['weekend'].get('afternoon', ''))
                node_attrs['routine_weekend_evening'] = str(routine['weekend'].get('evening', ''))
                node_attrs['routine_weekend_night'] = str(routine['weekend'].get('night', ''))
            else:
                for moment in ['morning', 'midday', 'afternoon', 'evening', 'night']:
                    node_attrs[f'routine_weekend_{moment}'] = ''
        else:
            # Pas de routine → tous vides
            for day_type in ['weekday', 'weekend']:
                for moment in ['morning', 'midday', 'afternoon', 'evening', 'night']:
                    node_attrs[f'routine_{day_type}_{moment}'] = ''
        
        # ========== LIEUX FAVORIS (6 TYPES) ==========
        fav_places = getattr(agent, 'favorite_places', None)
        
        if fav_places and isinstance(fav_places, dict):
            core_types = ['restaurant', 'cafe', 'supermarche', 'gym', 'parc', 'cinema']
            
            for place_type in core_types:
                value = fav_places.get(place_type)
                # Convertir en string, remplacer None par chaîne vide
                if value and value != 'None':
                    node_attrs[f'fav_{place_type}'] = str(value)
                else:
                    node_attrs[f'fav_{place_type}'] = ''
        else:
            # Pas de favorite_places → tous vides
            for place_type in ['restaurant', 'cafe', 'supermarche', 'gym', 'parc', 'cinema']:
                node_attrs[f'fav_{place_type}'] = ''
        
        # ========== STATISTIQUES VISITES ==========
        visit_count = getattr(agent, 'place_visit_count', None)
        if visit_count and isinstance(visit_count, dict):
            total_visits = sum(visit_count.values())
            node_attrs['total_visits'] = str(total_visits)
            # Lieu le plus visité
            if total_visits > 0:
                most_visited = max(visit_count.items(), key=lambda x: x[1])
                node_attrs['most_visited_place'] = str(most_visited[0])
                node_attrs['most_visited_count'] = str(most_visited[1])
            else:
                node_attrs['most_visited_place'] = ''
                node_attrs['most_visited_count'] = '0'
        else:
            node_attrs['total_visits'] = '0'
            node_attrs['most_visited_place'] = ''
            node_attrs['most_visited_count'] = '0'
        
        # ========== VALIDATION : Supprimer None/vides ==========
        safe_attrs = {}
        for k, v in node_attrs.items():
            if v not in [None, 'None', '']:
                safe_attrs[k] = v
            else:
                safe_attrs[k] = ''  # Gephi préfère '' à None
        
        # Ajouter nœud
        G.add_node(agent.id, **safe_attrs)

    # ========== AJOUT ARÊTES ==========
    edge_count = 0
    
    for bubble_idx, bubble in enumerate(bubbles):
        bubble_list = list(bubble)
        
        for i in range(len(bubble_list)):
            for j in range(i + 1, len(bubble_list)):
                agent_id = bubble_list[i]
                other_id = bubble_list[j]
                
                if G.has_node(agent_id) and G.has_node(other_id):
                    # Calculer force du lien
                    agent = next((a for a in agents if a.id == agent_id), None)
                    other = next((a for a in agents if a.id == other_id), None)
                    
                    weight = 1.0
                    relation_type = 'bubble'
                    
                    if agent and other:
                        # Famille > Amis > Collègues > Voisins
                        if other_id in getattr(agent, 'family', set()):
                            weight = 3.0
                            relation_type = 'family'
                        elif other_id in getattr(agent, 'friends', set()):
                            weight = 2.0
                            relation_type = 'friend'
                        elif other_id in getattr(agent, 'colleagues', set()):
                            weight = 1.5
                            relation_type = 'colleague'
                        elif other_id in getattr(agent, 'neighbors', set()):
                            weight = 1.2
                            relation_type = 'neighbor'
                    
                    edge_attrs = {
                        'weight': str(weight),
                        'relation': relation_type,
                        'bubble_id': str(bubble_idx),
                        'community_id': str(comm_id),
                    }
                    
                    G.add_edge(agent_id, other_id, **edge_attrs)
                    edge_count += 1

    # ========== EXPORT ==========
    nx.write_gexf(G, f"Projet_IA_Grégoire_Dunglas/outputs/{filename}")
    
    # ========== STATISTIQUES ==========
    print(f"EXPORT GEXF EXHAUSTIF RÉUSSI")
    print(f"Fichier : Projet_IA_Grégoire_Dunglas/outputs/{filename}")
    print(f"Statistiques :")
    print(f"   - Nœuds : {G.number_of_nodes()}")
    print(f"   - Arêtes : {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        sample_node = list(G.nodes(data=True))[0]
        num_attrs = len(sample_node[1])
        print(f"   - Attributs/nœud : {num_attrs}")
        print(f"\n Liste attributs nœuds :")
        for i, attr in enumerate(sorted(sample_node[1].keys()), 1):
            print(f"      {i:2d}. {attr}")
    
    if G.number_of_edges() > 0:
        sample_edge = list(G.edges(data=True))[0]
        print(f"\n Attributs arêtes :")
        for attr in sorted(sample_edge[2].keys()):
            print(f"      - {attr}")
    
    return G

def plot_epidemic_community_spread(agents, epidemic_model, communities, day, filename='epidemic_community_spread.png'):
    """
    Visualise propagation épidémique PAR COMMUNAUTÉ
    
    Affichage :
    - 1 subplot par communauté (max 9)
    - Nœuds colorés par état SEIR
    - Taille proportionnelle à centralité
    - Arêtes de transmission en rouge
    
    Args:
        agents: Liste agents
        epidemic_model: Instance EpidemicModel
        communities: Dict {comm_id: [agent_ids]}
        day: Jour actuel
        filename: Nom fichier sortie
    """
    
    # Limiter à 9 premières communautés (3x3 grid)
    communities_to_plot = dict(list(communities.items())[:9])
    n_communities = len(communities_to_plot)
    
    # Calculer grille optimale
    n_cols = min(3, n_communities)
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    # Créer figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Forcer axes en array 2D
    if n_communities == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Dictionnaire agents
    agent_dict = {a.id: a for a in agents}
    
    # Couleurs états SEIR
    color_map = {
        'S': '#3498db',  # Bleu
        'E': '#f39c12',  # Orange
        'I': '#e74c3c',  # Rouge
        'R': '#2ecc71',   # Vert
        'V': '#9b59b6'   # Violet
    }
    
    # ========== TRACER CHAQUE COMMUNAUTÉ ==========
    for idx, (comm_id, members) in enumerate(communities_to_plot.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # --- Construire sous-graphe communauté ---
        G_comm = nx.Graph()
        
        for member_id in members:
            if member_id in agent_dict:
                agent = agent_dict[member_id]
                state = epidemic_model.agent_states.get(member_id, 'S')
                
                G_comm.add_node(
                    member_id,
                    state=state,
                    psychology=agent.psychology,
                    age=agent.age
                )
        
        # Arêtes intra-communauté
        for member_id in members:
            if member_id not in agent_dict:
                continue
            agent = agent_dict[member_id]
            
            # Ajouter liens amis/famille/collègues dans communauté
            for friend_id in agent.friends:
                if friend_id in members and G_comm.has_node(friend_id):
                    G_comm.add_edge(member_id, friend_id, relation='friend')
            
            for fam_id in getattr(agent, 'family', set()):
                if fam_id in members and G_comm.has_node(fam_id):
                    if not G_comm.has_edge(member_id, fam_id):
                        G_comm.add_edge(member_id, fam_id, relation='family')
        
        if G_comm.number_of_nodes() == 0:
            ax.axis('off')
            continue
        
        # --- Layout ---
        pos = nx.spring_layout(G_comm, k=0.5, iterations=50, seed=42)
        
        # --- Centralité pour tailles ---
        if G_comm.number_of_nodes() > 1:
            centrality = nx.degree_centrality(G_comm)
        else:
            centrality = {list(G_comm.nodes())[0]: 1.0}
        
        node_sizes = [300 + 1000 * centrality[n] for n in G_comm.nodes()]
        
        # --- Couleurs nœuds ---
        node_colors = [color_map[G_comm.nodes[n]['state']] for n in G_comm.nodes()]
        
        # --- Arêtes normales ---
        normal_edges = [(u, v) for u, v in G_comm.edges() 
                       if not _is_transmission_edge(u, v, epidemic_model)]
        
        nx.draw_networkx_edges(
            G_comm, pos, 
            edgelist=normal_edges,
            alpha=0.2, 
            width=1, 
            edge_color='gray',
            ax=ax
        )
        
        # --- Arêtes de transmission (rouge) ---
        transmission_edges = [(u, v) for u, v in G_comm.edges() 
                             if _is_transmission_edge(u, v, epidemic_model)]
        
        if transmission_edges:
            nx.draw_networkx_edges(
                G_comm, pos,
                edgelist=transmission_edges,
                edge_color='red',
                width=3,
                alpha=0.8,
                arrows=True,
                arrowsize=15,
                arrowstyle='-|>',
                ax=ax
            )
        
        # --- Nœuds ---
        nx.draw_networkx_nodes(
            G_comm, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            linewidths=2,
            edgecolors='black',
            ax=ax
        )
        
        # --- Labels (seulement hubs) ---
        hub_threshold = np.percentile(list(centrality.values()), 80)
        labels = {n: str(n) for n, c in centrality.items() if c >= hub_threshold}
        
        nx.draw_networkx_labels(
            G_comm, pos,
            labels=labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        # --- Statistiques communauté ---
        stats = _compute_community_stats(members, agent_dict, epidemic_model)
        
        title = f"Communauté {comm_id} (n={len(members)})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Texte statistiques
        stats_text = (
            f"S: {stats['S']} | E: {stats['E']} | "
            f"I: {stats['I']} | R: {stats['R']}\n"
            f"Transmissions: {stats['transmissions']}"
        )
        
        ax.text(
            0.5, -0.1,
            stats_text,
            transform=ax.transAxes,
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.axis('off')
    
    # Masquer axes vides
    for idx in range(n_communities, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # --- Légende globale ---
    legend_elements = [
        patches.Patch(facecolor=color_map['S'], label='Susceptible'),
        patches.Patch(facecolor=color_map['E'], label='Exposé'),
        patches.Patch(facecolor=color_map['I'], label='Infectieux'),
        patches.Patch(facecolor=color_map['R'], label='Guéri'),
        Line2D([0], [0], color='red', lw=3, label='Transmission')
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=5,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.98)
    )
    
    # Titre global
    fig.suptitle(
        f'Propagation Épidémique par Communauté - Jour {day}',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Visualisation communautés sauvegardée : outputs/{filename}")


def _is_transmission_edge(source_id, target_id, epidemic_model):
    """Vérifie si arête correspond à une transmission"""
    return (
        target_id in epidemic_model.transmission_memory and
        epidemic_model.transmission_memory[target_id] == source_id
    ) or (
        source_id in epidemic_model.transmission_memory and
        epidemic_model.transmission_memory[source_id] == target_id
    )


def _compute_community_stats(members, agent_dict, epidemic_model):
    """Calcule statistiques SEIR d'une communauté"""
    stats = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'V': 0, 'transmissions': 0}
    
    for member_id in members:
        if member_id in agent_dict:
            state = epidemic_model.agent_states.get(member_id, 'S')
            stats[state] += 1
    
    # Compter transmissions intra-communauté
    for target_id, source_id in epidemic_model.transmission_memory.items():
        if target_id in members and source_id in members:
            stats['transmissions'] += 1
    
    return stats


def plot_epidemic_community_timeline(agents, epidemic_model, communities, 
                                     history_by_community, filename='epidemic_community_timeline.png'):
    """
    Évolution temporelle SEIR par communauté
    
    Args:
        agents: Liste agents
        epidemic_model: Instance EpidemicModel
        communities: Dict {comm_id: [agent_ids]}
        history_by_community: Dict {comm_id: {day: {'S': ..., 'E': ..., 'I': ..., 'R': ...}}}
        filename: Nom fichier sortie
    """
    
    n_communities = min(9, len(communities))
    communities_to_plot = dict(list(communities.items())[:n_communities])
    
    # Calculer grille
    n_cols = 3
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    
    # Forcer array 2D
    if n_communities == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Couleurs
    colors = {
        'S': '#3498db',
        'E': '#f39c12',
        'I': '#e74c3c',
        'R': '#2ecc71'
    }
    
    for idx, (comm_id, members) in enumerate(communities_to_plot.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if comm_id not in history_by_community:
            ax.axis('off')
            continue
        
        history = history_by_community[comm_id]
        days = sorted(history.keys())
        
        # Extraire séries temporelles
        S_series = [history[d]['S'] for d in days]
        E_series = [history[d]['E'] for d in days]
        I_series = [history[d]['I'] for d in days]
        R_series = [history[d]['R'] for d in days]
        
        # Tracer aires empilées
        ax.fill_between(days, 0, S_series, label='Susceptible', color=colors['S'], alpha=0.7)
        ax.fill_between(days, S_series, 
                        [s+e for s,e in zip(S_series, E_series)],
                        label='Exposé', color=colors['E'], alpha=0.7)
        ax.fill_between(days, [s+e for s,e in zip(S_series, E_series)],
                        [s+e+i for s,e,i in zip(S_series, E_series, I_series)],
                        label='Infectieux', color=colors['I'], alpha=0.7)
        ax.fill_between(days, [s+e+i for s,e,i in zip(S_series, E_series, I_series)],
                        [s+e+i+r for s,e,i,r in zip(S_series, E_series, I_series, R_series)],
                        label='Guéri', color=colors['R'], alpha=0.7)
        
        ax.set_xlabel('Jour', fontsize=11)
        ax.set_ylabel('Nombre agents', fontsize=11)
        ax.set_title(f'Communauté {comm_id} (n={len(members)})', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='right', fontsize=9)
    
    # Masquer axes vides
    for idx in range(n_communities, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle('Évolution Épidémique par Communauté', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Timeline communautés sauvegardée : outputs/{filename}")

def plot_interaction_heatmap(interaction_matrix, agents, filename):
    """
    Heatmap des interactions par type de lieu.
    """        
    # Agréger par (lieu_agent1, lieu_agent2)
    location_matrix = defaultdict(int)
    agent_dict = {a.id: a for a in agents}
    
    for key, count in interaction_matrix.items():
        if len(key) == 3:
            id1, id2, location = key
            location_matrix[(location, location)] += count
        else:
            id1, id2 = key
            loc1 = getattr(agent_dict.get(id1), 'current_location', 'unknown')
            loc2 = getattr(agent_dict.get(id2), 'current_location', 'unknown')
            location_matrix[tuple(sorted([loc1 or "unknown", loc2 or "unknown"]))] += count
    
    # Créer matrice carrée
    locations = sorted(set(loc for pair in location_matrix.keys() for loc in pair))
    n = len(locations)
    matrix = np.zeros((n, n))
    
    for (loc1, loc2), count in location_matrix.items():
        i = locations.index(loc1)
        j = locations.index(loc2)
        matrix[i, j] += count
        matrix[j, i] += count  # Symétrie
    
    # Tracer heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(locations, rotation=45, ha='right')
    ax.set_yticklabels(locations)
    
    # Annotations
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, f'{int(matrix[i, j])}', 
                    ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title('Heatmap des interactions par lieu', fontsize=16)
    fig.colorbar(im, ax=ax, label='Nombre d\'interactions')
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs/{filename}')
    plt.close()
    print(f" Heatmap sauvegardée : Projet_IA_Grégoire_Dunglas/outputs/{filename}")

def plot_seirv_dynamics(history, vaccination_day=None, filename='seirv_dynamics.png'):
    """
    Graphique SEIR-V avec vaccinés et intervention visible
    
    Args:
        history: Dict {'S': [...], 'E': [...], 'I': [...], 'R': [...], 'V': [...]}
        vaccination_day: Jour de la campagne vaccinale (ligne verticale)
        filename: Nom fichier sortie
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    days = range(len(history['S']))
    
    # ========== AIRES EMPILÉES ==========
    
    # Susceptibles (bleu)
    ax.fill_between(days, 0, history['S'], 
                     label='Susceptible (S)', 
                     color='#3498db', alpha=0.8)
    
    # Exposés (orange)
    ax.fill_between(days, history['S'], 
                     [s+e for s,e in zip(history['S'], history['E'])],
                     label='Exposé (E)', 
                     color='#f39c12', alpha=0.8)
    
    # Infectieux (rouge vif)
    ax.fill_between(days, 
                     [s+e for s,e in zip(history['S'], history['E'])],
                     [s+e+i for s,e,i in zip(history['S'], history['E'], history['I'])],
                     label='Infectieux (I)', 
                     color='#e74c3c', alpha=0.9)
    
    # Guéris (vert)
    ax.fill_between(days, 
                     [s+e+i for s,e,i in zip(history['S'], history['E'], history['I'])],
                     [s+e+i+r for s,e,i,r in zip(history['S'], history['E'], history['I'], history['R'])],
                     label='Guéri (R)', 
                     color='#2ecc71', alpha=0.8)
    
    #  VACCINÉS (violet/mauve)
    ax.fill_between(days, 
                     [s+e+i+r for s,e,i,r in zip(history['S'], history['E'], history['I'], history['R'])],
                     [s+e+i+r+v for s,e,i,r,v in zip(history['S'], history['E'], history['I'], history['R'], 
                                                       history['V'])],
                     label='Vacciné (V) - Immunité parfaite', 
                     color='#9b59b6', alpha=0.9)
    
    # ========== LIGNE VERTICALE VACCINATION ==========
    if vaccination_day is not None:
        ax.axvline(x=vaccination_day, color='purple', linestyle='--', 
                   linewidth=3, alpha=0.8, label=f'Vaccination (J{vaccination_day})')
        
        # Annotation
        max_pop = len(history['S'])
        ax.annotate(' Campagne\nVaccinale', 
                    xy=(vaccination_day, max_pop * 0.8),
                    xytext=(vaccination_day + 2, max_pop * 0.85),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='purple', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    
    # ========== LIGNE R₀ = 1 (SEUIL CRITIQUE) ==========
    # Estimation visuelle du moment où I décroît
    i_peak_day = history['I'].index(max(history['I']))
    ax.axvline(x=i_peak_day, color='red', linestyle=':', 
               linewidth=2, alpha=0.6)
    ax.text(i_peak_day + 0.5, max(history['I']) * 1.05, 
            f'Pic infectieux\n(J{i_peak_day})',
            fontsize=10, color='red', fontweight='bold')
    
    # ========== STATISTIQUES EN BOX ==========
    total_pop = history['S'][0]
    final_infected = history['I'][-1] + history['R'][-1]
    final_vaccinated = history['V'][-1]
    attack_rate = (final_infected / total_pop) * 100
    vaccination_coverage = (final_vaccinated / total_pop) * 100
    
    stats_text = (
        f"Population : {total_pop}\n"
        f"Pic infectieux : {max(history['I'])} (J{i_peak_day})\n"
        f"Taux d'attaque final : {attack_rate:.1f}%\n"
        f"Couverture vaccinale : {vaccination_coverage:.1f}%\n"
        f"Protégés (V) : {final_vaccinated}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== STYLE ==========
    ax.set_xlabel('Jours', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nombre d\'agents', fontsize=14, fontweight='bold')
    ax.set_title('Dynamique SEIR-V : Impact de la Vaccination', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(days)-1)
    ax.set_ylim(0, total_pop * 1.05)
    
    # Annotations axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs//{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Graphique SEIR-V sauvegardé : outputs/{filename}")


def plot_vaccination_impact(history, vaccination_day, filename='vaccination_impact.png'):
    """
    Comparaison avant/après vaccination (2 subplots)
    
    Subplot 1 : Infectieux (I) avec ligne vaccination
    Subplot 2 : Cumul S+V (susceptibles protégés)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    days = range(len(history['S']))
    
    # ========== SUBPLOT 1 : INFECTIEUX ==========
    ax1.plot(days, history['I'], color='#e74c3c', linewidth=3, 
             label='Infectieux (I)')
    ax1.fill_between(days, 0, history['I'], color='#e74c3c', alpha=0.3)
    
    # Ligne vaccination
    if vaccination_day is not None:
        ax1.axvline(x=vaccination_day, color='purple', linestyle='--', 
                   linewidth=3, alpha=0.8, label='Vaccination')
    
    # Zones avant/après
    if vaccination_day:
        ax1.axvspan(0, vaccination_day, alpha=0.1, color='red', 
                   label='Avant vaccination')
        ax1.axvspan(vaccination_day, len(days)-1, alpha=0.1, color='green',
                   label='Après vaccination')
    
    ax1.set_ylabel('Nombre infectieux', fontsize=13, fontweight='bold')
    ax1.set_title('Impact Vaccination sur les Infections Actives', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # ========== SUBPLOT 2 : SUSCEPTIBLES VS VACCINÉS ==========
    ax2.plot(days, history['S'], color='#3498db', linewidth=3, 
             label='Susceptibles (S)', linestyle='-')
    ax2.plot(days, history['V'], color='#9b59b6', linewidth=3, 
             label='Vaccinés (V) - Protégés', linestyle='-')
    
    # Cumul protégés
    protected = [s+v for s,v in zip(history['S'], history['V'])]
    ax2.fill_between(days, 0, protected, color='green', alpha=0.2,
                    label='Total protégés (S+V)')
    
    # Ligne vaccination
    if vaccination_day is not None:
        ax2.axvline(x=vaccination_day, color='purple', linestyle='--', 
                   linewidth=3, alpha=0.8)
    
    ax2.set_xlabel('Jours', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Nombre agents', fontsize=13, fontweight='bold')
    ax2.set_title('Évolution Susceptibles vs Vaccinés', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs//{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Impact vaccination sauvegardé : outputs/{filename}")


def plot_vaccine_efficacy_comparison(history_no_vaccine, history_with_vaccine, 
                                     vaccination_day, filename='vaccine_efficacy.png'):
    """
    Comparaison scénario AVEC vs SANS vaccination (ligne pointillée)
    
    Args:
        history_no_vaccine: Historique simulation sans vaccin
        history_with_vaccine: Historique simulation avec vaccin
        vaccination_day: Jour vaccination
        filename: Nom fichier
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    days = range(len(history_with_vaccine['S']))
    
    # ========== SCÉNARIO AVEC VACCIN (LIGNE PLEINE) ==========
    total_infected_with = [e+i+r for e,i,r in zip(
        history_with_vaccine['E'],
        history_with_vaccine['I'],
        history_with_vaccine['R']
    )]
    
    ax.plot(days, total_infected_with, 
            color='#2ecc71', linewidth=3, 
            label='AVEC vaccination (E+I+R)', linestyle='-')
    ax.fill_between(days, 0, total_infected_with, 
                    color='#2ecc71', alpha=0.3)
    
    # ========== SCÉNARIO SANS VACCIN (LIGNE POINTILLÉE) ==========
    total_infected_without = None
    if history_no_vaccine:
        days_no_vac = range(len(history_no_vaccine['S']))
        total_infected_without = [e+i+r for e,i,r in zip(
            history_no_vaccine['E'],
            history_no_vaccine['I'],
            history_no_vaccine['R']
        )]
        
        ax.plot(days_no_vac, total_infected_without,
                color='#e74c3c', linewidth=3,
                label='SANS vaccination (contre-factuel)', 
                linestyle='--')
    
    # ========== LIGNE VACCINATION ==========
    if vaccination_day is not None:
        ax.axvline(x=vaccination_day, color='purple', linestyle='--',
                  linewidth=3, alpha=0.8, label='Vaccination')
    
    # ========== CALCUL INFECTIONS ÉVITÉES ==========
    if total_infected_without is not None:
        infections_avoided = total_infected_without[-1] - total_infected_with[-1]
        efficacy = (infections_avoided / total_infected_without[-1]) * 100
        
        ax.text(0.5, 0.95, 
                f'Infections évitées : {infections_avoided}\n'
                f'Efficacité globale : {efficacy:.1f}%',
                transform=ax.transAxes,
                fontsize=13, fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    ax.set_xlabel('Jours', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumul infectés (E+I+R)', fontsize=14, fontweight='bold')
    ax.set_title('Efficacité de la Vaccination : Comparaison Scénarios', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Projet_IA_Grégoire_Dunglas/outputs//{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Comparaison efficacité sauvegardée : outputs/{filename}")
