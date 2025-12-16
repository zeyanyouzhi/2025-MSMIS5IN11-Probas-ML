import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# --- 1. CHARGEMENT DES DONNÉES ---
# (Même méthode manuelle que cancer_trainer.py pour la cohérence)
print("Chargement et préparation des données...")

X_train = np.array([[float(j) for j in i.rstrip().split(",")] 
                    for i in open("train.csv").readlines()])
Y_train = X_train[:,-1]
X_train = X_train[:,0:-1]

X_test = np.array([[float(j) for j in i.rstrip().split(",")] 
                   for i in open("test.csv").readlines()])
Y_test = X_test[:,-1]
X_test = X_test[:,0:-1]

# --- 2. DÉFINITION DES 3 CHALLENGERS ---

# A. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# B. SVM (Le favori pour la sécurité)
svm = SVC(kernel='linear', random_state=42)

# C. Deep Learning (Ton réseau de neurones - Version Scikit-Learn)
# hidden_layer_sizes=(64, 64) correspond à tes 2 couches de 64 neurones
dl = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)

# --- 3. BOUCLE DE TEST UNIQUE ---
# On met les modèles dans une liste pour les tester un par un proprement
modeles = [
    ("Random Forest", rf),
    ("SVM (Classique)", svm),
    ("Deep Learning (MLP)", dl)
]

for nom, modele in modeles:
    print(f"\n⏳ Entraînement de : {nom}...")
    modele.fit(X_train, Y_train)
    y_pred = modele.predict(X_test)
    
    # Calcul des scores
    acc = accuracy_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    cm = confusion_matrix(Y_test, y_pred)
    faux_negatifs = cm[1][0] # Le chiffre le plus important !
    
    # Affichage unique
    print(f"📊 RÉSULTATS : {nom}")
    print(f"   > Précision (Accuracy) : {acc*100:.2f}%")
    print(f"   > Rappel (Sécurité)    : {rec*100:.2f}%")
    print(f"   > Faux Négatifs        : {faux_negatifs} (Malades non détectés)")
    print("-" * 30)