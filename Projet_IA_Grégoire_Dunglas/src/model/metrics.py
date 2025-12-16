"""
Métriques épidémiologiques avancées avec validation temporelle
Inspiré : Fraser et al. (2009) - Pandemic potential of influenza
"""

import numpy as np
from collections import defaultdict
from collections import Counter

def compute_attack_rate(agents):
    """Taux d'attaque = % population infectée à un moment"""
    infected = sum(1 for a in agents if a.status in ['infecté', 'immunisé'])
    return infected / len(agents)

def compute_R0_dynamic(epidemic_model, day, window=14):
    """
    R0 effectif par cohorte COMPLÈTE (Fraser et al. 2009)
    
    AMÉLIORATION MAJEURE :
    - Cohorte = infectés entre [day-window, day-window+7] (terminé leur période)
    - Exclusion des cas encore en incubation
    - Normalisation temporelle
    
    Args:
        epidemic_model: Instance EpidemicModel
        day: Jour actuel
        window: Fenêtre rétrospective (14 jours par défaut)
    
    Returns:
        float: R0 effectif (0 si données insuffisantes)
    """
    if not epidemic_model.transmission_memory or day < window:
        return 0.0
    
    # Fenêtre de cohorte : infectés il y a [14-7] jours
    cohort_start = day - window
    cohort_end = day - (window // 2)
    
    # Identifier infectés de cette cohorte
    cohort_infected = set()
    for agent_id, infection_day in epidemic_model.agent_infection_day.items():
        if cohort_start <= infection_day <= cohort_end:
            # Vérifier que l'agent a terminé sa période contagieuse
            if day - infection_day > 10:  # 10 jours = période contagieuse max
                cohort_infected.add(agent_id)
    
    if not cohort_infected:
        return 0.0
    
    # Compter transmissions causées par cette cohorte
    secondary_cases = 0
    for target_id, source_id in epidemic_model.transmission_memory.items():
        if source_id in cohort_infected:
            # Vérifier que la transmission a eu lieu APRÈS l'infection source
            target_day = epidemic_model.agent_infection_day.get(target_id)
            source_day = epidemic_model.agent_infection_day.get(source_id)
            if target_day and source_day and target_day > source_day:
                secondary_cases += 1
    
    r0 = secondary_cases / len(cohort_infected)
    
    return r0

def compute_generation_time(epidemic_model):
    """
    Temps de génération = délai moyen entre infections successives
    
    Métrique clé épidémiologique (Fraser 2009)
    """
    generation_times = []
    
    for target_id, source_id in epidemic_model.transmission_memory.items():
        source_day = epidemic_model.agent_infection_day.get(source_id)
        target_day = epidemic_model.agent_infection_day.get(target_id)
        
        if source_day is not None and target_day is not None:
            generation_times.append(target_day - source_day)
    
    if not generation_times:
        return None
    
    return np.mean(generation_times), np.std(generation_times)

def compute_dispersion_k(epidemic_model):
    """
    Paramètre de dispersion k (Lloyd-Smith 2005)
    
    k < 1 → super-spreading (20% infectent 80%)
    k > 1 → transmission homogène
    """    
    # Compter contaminations par source
    contaminations = Counter(epidemic_model.transmission_memory.values())
    secondary_cases = list(contaminations.values())
    
    if not secondary_cases:
        return None
    
    # Estimation k via méthode des moments
    mean_R = np.mean(secondary_cases)
    var_R = np.var(secondary_cases)
    
    if var_R <= mean_R:
        return float('inf')  # Distribution poisson (pas de super-spreading)
    
    k = mean_R**2 / (var_R - mean_R)
    return k

def compute_effective_reproduction_number(epidemic_model, day, susceptibles_fraction):
    """
    Nombre de reproduction effectif Rt = R0 × S(t)/N
    
    Tient compte de la déplétion des susceptibles.
    """
    r0 = compute_R0_dynamic(epidemic_model, day)
    return r0 * susceptibles_fraction

def validate_epidemic_realism(epidemic_model, day):
    """
    Validation scientifique de la dynamique épidémique
    
    Critères :
    - R0 ∈ [1.5, 4] (plausible pour pathogènes respiratoires)
    - Temps génération ∈ [3, 7] jours
    - k < 1 (super-spreading)
    """
    R0 = compute_R0_dynamic(epidemic_model, day)
    gen_time = compute_generation_time(epidemic_model)
    k = compute_dispersion_k(epidemic_model)
    
    print(f"\n Validation épidémique (jour {day}) :")
    print(f"   - R0 effectif : {R0:.2f} (attendu : 1.5-4)")
    
    if gen_time:
        print(f"   - Temps génération : {gen_time[0]:.1f} ± {gen_time[1]:.1f} jours (attendu : 3-7)")
    
    if k is not None:
        print(f"   - Dispersion k : {k:.2f} (k<1 = super-spreading)")
    
    # Alertes
    warnings = []
    
    if R0 < 1.2:
        warnings.append("R0 trop faible → extinction prématurée")
    elif R0 > 5:
        warnings.append("R0 trop élevé → dynamique irréaliste")
    
    if gen_time and gen_time[0] < 2:
        warnings.append("Temps génération trop court")
    
    if k and k > 2:
        warnings.append("Pas de super-spreading (k>1)")
    
    if warnings:
        print("\nALERTES :")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\nDynamique épidémique réaliste")
    
    return len(warnings) == 0