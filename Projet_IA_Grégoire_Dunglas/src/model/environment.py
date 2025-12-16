import random

# Modélisation de l'environnement urbain

class CityEnvironment:
    def __init__(self, size=(100, 100)):
        self.size = size
        self.quarters = self._create_quarters()
        self.hospitals = self._create_hospitals()
        self.transports = self._create_transports()
        self.gathering_points = self._create_gathering_points()
        # Génération de lieux par quartier (diversité et répartition non uniforme)
        self.places_by_quarter = self._create_places_by_quarter()
        # Dictionnaire des lieux et indices de risque (sources : Le Monde, SPF, Sciences&Avenir)
        self.place_risk = {
            'domicile': 0.2,
            'ecole': 0.4,
            'bureau': 0.5,
            'transports': 0.7,
            'restaurant': 0.8,
            'cafe': 0.6,
            'supermarche': 0.5,
            'gym': 0.5,
            'parc': 0.2,
            'cinema': 0.7,
            'hopital': 0.6,
            'public_exterieur': 0.1,
            'public_interieur': 0.6,
            'rassemblement': 0.9
        }
        self.all_agents = agents = []

    def get_risk_index(self, place_type):
        """Retourne l'indice de risque pour un type de lieu."""
        return self.place_risk.get(place_type, 0.2)

    def _create_quarters(self):
        # Création de 4 quartiers
        return {
            'Nord': 'Nord',
            'Sud': 'Sud',
            'Est': 'Est',
            'Ouest': 'Ouest'
        }

    def _create_places_by_quarter(self):
        # Pour chaque quartier, on génère un nombre variable de lieux de chaque type
        # Certains quartiers peuvent ne pas avoir certains types de lieux
        place_types = [
            'restaurant', 'cafe', 'supermarche', 'gym', 'parc', 'cinema',
            'theatre', 'piscine', 'bibliotheque', 'musee', 'stadium', 'bowling', 'salle_jeux', 'dojo', 'salle_danse', 'salle_concert', 'patinoire', 'salle_escalade', 'salle_yoga', 'salle_musculation', 'salle_boxe', 'salle_multisport', 'club_sportif', 'terrain_foot', 'terrain_basket', 'terrain_tennis', 'terrain_rugby', 'terrain_hand', 'terrain_volley', 'skatepark', 'centre_equestre', 'salle_chess', 'escape_game', 'laser_game', 'karting', 'paintball', 'parc_aquatique', 'zoo', 'aquarium', 'planetarium', 'salle_cinema_art', 'salle_cinema_blockbuster', 'salle_cinema_independant', 'salle_cinema_plein_air'
        ]
        quarters = list(self._create_quarters().keys())
        places_by_quarter = {q: {} for q in quarters}
        random.seed(42)  # Pour reproductibilité
        for q in quarters:
            for place_type in place_types:
                # Certains quartiers n'ont pas tous les types de lieux (20% de chance d'absence)
                if random.random() < 0.2:
                    continue
                n_places = random.randint(1, 4)  # 1 à 4 lieux par type/quartier
                places = [f"{place_type}_{q}_{i+1}" for i in range(n_places)]
                places_by_quarter[q][place_type] = places
        return places_by_quarter

    def _create_public_places(self):
        return [(50, 50), (20, 60), (70, 30)]

    def _create_hospitals(self):
        return [(15, 85), (85, 15)]

    def _create_transports(self):
        return [(50, 10), (10, 50), (90, 50), (50, 90)]

    def _create_gathering_points(self):
        return [(50, 50), (20, 60), (70, 30), (15, 85), (85, 15)]

    def get_random_location(self, type_, quarter=None):
        if type_ == 'quarter':
            return random.choice(list(self.quarters.values()))
        elif type_ in ['restaurant', 'cafe', 'supermarche', 'gym', 'parc', 'cinema']:
            # Prendre dans le quartier si possible, sinon dans toute la ville
            if quarter and quarter in self.places_by_quarter:
                places = self.places_by_quarter[quarter].get(type_, [])
                if places:
                    return random.choice(places)
            # Sinon, choisir dans tous les quartiers
            all_places = []
            for q in self.places_by_quarter:
                all_places.extend(self.places_by_quarter[q].get(type_, []))
            if all_places:
                return random.choice(all_places)
            return None
        elif type_ == 'public':
            # Pour compatibilité, choisir un lieu public aléatoire (tous types confondus)
            all_places = []
            for q in self.places_by_quarter:
                for t in self.places_by_quarter[q]:
                    all_places.extend(self.places_by_quarter[q][t])
            if all_places:
                return random.choice(all_places)
            return None
        elif type_ == 'hospital':
            return random.choice(self.hospitals)
        elif type_ == 'transport':
            return random.choice(self.transports)
        elif type_ == 'gathering':
            return random.choice(self.gathering_points)
        else:
            return (0, 0)
