# Prédiction du Risque de Crises d'Asthme en Île-de-France

**Projet IA - Algorithmique**  
**François DESLANDES, Agathe LESCHIERA, Solène VON MULLEM**

---

## Présentation du Projet

### Objectif
Développer un modèle de machine learning capable de **prédire les semaines à risque élevé pour les crises d'asthme** en Île-de-France en se basant sur :
- 🌡️ **Données météorologiques** (températures, précipitations)
- 💨 **Qualité de l'air** (NO2, NO, NOX)
- 📅 **Temporalité** (mois, saison)

### Contexte
L'asthme est une maladie chronique touchant des millions de personnes. Les conditions environnementales jouent un rôle crucial dans le déclenchement des crises. Ce projet utilise des données réelles françaises pour créer un outil prédictif permettant d'anticiper les périodes à risque.

---

## Sources de Données

### 1. Airparif (Qualité de l'Air)
- **Période** : 2018-2025 (8 ans)
- **Source** : https://data.airparif.asso.fr/
- **Station** : Saint-Denis (Seine-Saint-Denis)
- **Mesures** : NO2, NO, NOX (µg/m³) - Données horaires agrégées par semaine

### 2. Santé Publique France (Hospitalisations)
- **Période** : 2020-2025
- **Source** : https://www.data.gouv.fr/fr/organizations/sante-publique-france/
- **Région** : Île-de-France uniquement
- **Données** : Taux hebdomadaire de passages aux urgences pour asthme (pour 100k habitants)

### 3. Météo France (Météorologie)
- **Période** : 2020-2025
- **Source** : https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes/
- **Station** : Aubervilliers (Seine-Saint-Denis)
- **Mesures** : Températures (min, max, moyenne), précipitations, vent - Données quotidiennes agrégées par semaine

### Dataset Final
- **310 semaines** de données complètes (2020-2025)
- **17 variables** incluant météo, pollution, santé et temporalité
- **Agrégation hebdomadaire** (lundi = début de semaine)

---

## Méthodologie

### 1. Acquisition et Préparation des Données Réelles
- Téléchargement de 8 ans de données Airparif
- Fusion intelligente des hospitalisations (agrégation par classes d'âge)
- Calcul de températures moyennes manquantes : `(T_min + T_max) / 2`
- Nettoyage : Suppression de 15 semaines sans données de pollution

### 2. Analyse Exploratoire
**Découvertes clés** :
- 🌡️ **Température** : Corrélation négative (-0.30) → Plus froid = plus d'urgences
- 💨 **Pollution NO2/NOX** : Corrélation positive (+0.14 à +0.17)
- 📅 **Effet saisonnier fort** : Automne (novembre) = pire période
- Variables de vent : Toutes NaN → Exclues du modèle

### 3. Modélisation Machine Learning

#### Cible (Variable à Prédire)
- **Classe binaire** : Risque NORMAL (0) vs Risque ÉLEVÉ (1)
- **Seuil** : 75ème percentile du taux d'urgences (1773 passages/100k hab)
- **Répartition** : 75% Normal / 25% Élevé

#### Features (Variables Prédictives)
**AVEC temporalité (9 features)** :
- Météo : `temp_min_C`, `temp_max_C`, `temp_moy_C`, `precipitations_mm`
- Pollution : `NO2_ugm3`, `NO_ugm3`, `NOX_ugm3`
- Temporalité : `mois`, `trimestre`

**SANS temporalité (7 features)** :
- Uniquement météo + pollution (pas de mois/trimestre)

#### Split & Normalisation
- **Train/Test** : 80/20 avec stratification (236 train / 59 test)
- **Normalisation** : StandardScaler (important pour SVM, KNN, Logistic Regression)

---

## Résultats des Modèles

### Comparaison des Algorithmes (AVEC temporalité)

| Modèle | Accuracy | CV Score (5-fold) | AUC-ROC |
|--------|----------|-------------------|---------|
| **Decision Tree** ⭐ | **86.4%** | 80.1% | 0.763 |
| **Gradient Boosting** ⭐ | **86.4%** | 80.5% | **0.905** |
| **SVM** ⭐ | **86.4%** | 80.9% | 0.852 |
| Random Forest | 84.7% | 81.7% | 0.903 |
| Logistic Regression | 81.4% | 78.4% | 0.833 |
| KNN | 79.7% | 78.8% | 0.792 |

**Meilleur modèle sélectionné** : **Decision Tree** (86.4% accuracy)
- Précision "Risque normal" : 89% (recall 93%)
- Précision "Risque élevé" : 77% (recall 67%)
- F1-Score global : 0.86

### Impact de la Temporalité

Test **SANS** les variables `mois` et `trimestre` :

| Modèle | AVEC temporalité | SANS temporalité | Différence |
|--------|------------------|------------------|------------|
| Decision Tree | 86.4% | 62.7% | 📈 **+23.7%** |
| Gradient Boosting | 86.4% | 71.2% | 📈 +15.3% |
| SVM | 86.4% | 74.6% | 📈 +11.9% |
| Random Forest | 84.7% | **74.6%** | 📈 +10.2% |
| Logistic Regression | 81.4% | 72.9% | 📈 +8.5% |
| KNN | 79.7% | 74.6% | 📈 +5.1% |

### 💡 Découverte Majeure
**La temporalité (saison/mois) est CRUCIALE** : Impact de **+11.9 points** en moyenne !
- Sans elle, le meilleur modèle plafonne à **74.6%** (Random Forest)
- Avec elle, on atteint **86.4%** (+36.4 points vs hasard)

**Importance des Variables** (Decision Tree) :
1. 🗓️ **Mois** : ~50% de l'importance → Effet saisonnier dominant
2. 🌡️ **Temp. minimale** : ~15%
3. 💨 **NOX** : ~12%
4. 🌡️ **Temp. moyenne** : ~10%

---

## Fonction de Prédiction

### Utilisation

```python
# Prédiction AVEC temporalité (recommandé - 86.4% accuracy)
resultat = predire_risque_asthme(
    temp_min=8, temp_max=14, temp_moy=11,
    precipitations=2,
    no2=55, no=40, nox=120,
    mois=11, trimestre=4,
    avec_temporalite=True
)

print(resultat)
# {'risque': 'ÉLEVÉ ⚠️', 
#  'probabilite_risque_eleve': 0.944,
#  'modele_utilise': 'Decision Tree',
#  'accuracy_modele': 0.864}

# Prédiction SANS temporalité (74.6% accuracy)
resultat = predire_risque_asthme(
    temp_min=10, temp_max=15, temp_moy=12.5,
    precipitations=20,
    no2=30, no=10, nox=50,
    avec_temporalite=False  # Pas besoin de mois/trimestre
)
```

### Exemples de Prédictions

**✅ Risque NORMAL** :
- Été chaud (juillet) : 18-28°C, NO2=15 → Probabilité 0%
- Hiver très froid (décembre) mais pollution modérée → Probabilité 33%

**⚠️ Risque ÉLEVÉ** :
- Automne froid + pic de pollution (novembre) : 8-14°C, NO2=55 → **Probabilité 94%**
- Conditions intermédiaires sans info temporelle → Probabilité 55%

### Insights Climatiques
Le modèle a appris que :
- **Novembre (automne)** est la pire période pour l'asthme en Île-de-France
- **Janvier/février (hiver plein)** est moins risqué que prévu
- **Juillet/août (été)** : Risque minimal

→ Cohérent avec les données réelles : transition automne-hiver + reprise d'activité + chauffage

---

## Fichiers du Projet

```
├── prediction_notebook.ipynb    # Notebook principal (analyse complète)
├── best_asthma_model.pkl        # Modèle Decision Tree sauvegardé
├── scaler.pkl                   # StandardScaler pour normalisation
├── data/
│   ├── airparif/               # 8 CSV Airparif (2018-2025)
│   ├── hospitalisations/       # CSV Santé Publique France
│   └── meteo/                  # 2 CSV Météo France
└── README.md                   # Ce fichier
```

---

## Installation et Exécution

### Prérequis
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Lancer le Notebook
```bash
jupyter notebook prediction_notebook.ipynb
```

### Sections du Notebook
1. **Chargement des données** : Airparif, Santé Publique France, Météo France
2. **Préparation** : Agrégation hebdomadaire, fusion, nettoyage
3. **Visualisations** : Corrélations, séries temporelles, analyses saisonnières
4. **Modélisation ML** : 6 algorithmes testés avec/sans temporalité
5. **Fonction de prédiction** : Tests avec données fictives

---

## Résultats Clés

### ✅ Points Forts
- **86.4% d'accuracy** avec données réelles
- **Temporalité cruciale** : +11.9 points en moyenne
- **Modèle interprétable** : Decision Tree montre importance des variables
- **Données françaises réelles** : 5 ans de données Île-de-France
- **Fonction prédictive opérationnelle** : 2 modes (avec/sans temporalité)

### 🎯 Découvertes Scientifiques
1. **Automne (novembre) = période la plus à risque** (94% probabilité avec pollution)
2. **Température** : Facteur #2 après la temporalité (-0.30 corrélation)
3. **Pollution** : Impact modéré mais significatif (+0.14 à +0.17)
4. **Hiver plein** : Moins risqué que la transition automne-hiver

### ⚠️ Limitations
- Données limitées à l'Île-de-France (1 station)
- Variables de vent non disponibles (toutes NaN)
- Période 2020-2025 incluant COVID (possible biais)
- Classe déséquilibrée (75/25) par construction

---

## Perspectives d'Amélioration

### À Court Terme
- [ ] Tester d'autres régions françaises
- [ ] Intégrer données de pollens (allergènes)
- [ ] Ajouter humidité (non disponible actuellement)
- [ ] Implémenter API temps réel (Airparif, Météo France)

### À Moyen Terme
- [ ] Modèle de régression pour prédire le taux exact (pas juste binaire)
- [ ] Prédictions à J+7 (forecasting)
- [ ] Application web interactive
- [ ] Alertes SMS/mail pour semaines à risque

### Recherche
- [ ] Comparaison multi-régions (Paris vs Lyon vs Marseille)
- [ ] Analyse de l'impact COVID sur l'asthme (2020-2021)
- [ ] Étude des pollens + météo combinés
- [ ] Deep Learning (LSTM) pour séries temporelles

---

## Équipe

- **François DESLANDES**
- **Agathe LESCHIERA**
- **Solène VON MULLEM**

**EPF - 2025 - MSMIS5IN11 - Probas & ML**

---

## Licence

Projet pédagogique - EPF 2025
