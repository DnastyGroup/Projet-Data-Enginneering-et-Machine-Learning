# MBAESG_MBDIA-2025-2026_Classe1_EVALUATION_DATAENGINEER_MLOPS

## Projet : Data Engineering et Machine Learning avec Snowflake

---

## Contexte

Ce projet a été réalisé dans le cadre du workshop **Data Engineering et Machine Learning avec Snowflake**.  
L'objectif est de construire un pipeline ML complet directement dans l'environnement Snowflake, sans exporter les données vers un système externe.

Les données utilisées proviennent d'un dataset immobilier stocké sur S3 (`s3://logbrain-datalake/datasets/house_price/`), contenant des caractéristiques de maisons ainsi que leur prix de vente.

---

## Dataset

| Colonne | Description |
|---|---|
| `price` | Prix de vente de la maison |
| `area` | Surface totale (m²) |
| `bedrooms` | Nombre de chambres |
| `bathrooms` | Nombre de salles de bain |
| `stories` | Nombre d'étages |
| `mainroad` | Accès à une route principale (oui/non) |
| `guestroom` | Présence d'une chambre d'amis (oui/non) |
| `basement` | Présence d'un sous-sol (oui/non) |
| `hotwaterheating` | Chauffage à eau chaude (oui/non) |
| `airconditioning` | Climatisation (oui/non) |
| `parking` | Nombre de places de stationnement |
| `prefarea` | Zone privilégiée (oui/non) |
| `furnishingstatus` | État d'ameublement (meublé / semi-meublé / non meublé) |

**Variable cible :** `price`

---

## Architecture du pipeline

```
S3 (JSON)
   └─► Stage Snowflake
         └─► Table RAW (VARIANT)
               └─► Table HOUSE_PRICE (structurée)
                     └─► Préparation features (Snowpark + pandas)
                           ├─► Modèle Linear Regression
                           ├─► Modèle XGBoost (base)
                           └─► Modèle XGBoost optimisé (RandomizedSearchCV)
                                 └─► Snowflake Model Registry
                                       └─► Inférence → Table HOUSE_PRICE_PREDICTIONS
                                             └─► Application Streamlit
```

---

## Étapes réalisées

### 1. Configuration de l'environnement
- Création de la base de données `HOUSE_DB`
- Configuration d'un stage S3 `house_price_stage`
- Initialisation d'une session Snowpark

### 2. Ingestion des données
- Chargement du fichier JSON depuis S3 vers une table `house_price_raw` (format VARIANT)
- Aplatissement (`LATERAL FLATTEN`) et insertion dans la table structurée `house_price`

### 3. Exploration et préparation des données
- Chargement via `session.table("HOUSE_PRICE").to_pandas()`
- Séparation features (`X`) et variable cible (`y = PRICE`)
- Encodage des variables catégorielles avec `pd.get_dummies(drop_first=True)`
- Standardisation avec `StandardScaler`
- Split train/test : **80% / 20%** avec `random_state=42`

### 4. Entraînement des modèles

#### Régression Linéaire
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

#### XGBoost (base)
```python
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    random_state=42
)
```

#### XGBoost optimisé (RandomizedSearchCV)
```python
param_dist = {
    "n_estimators":     [50, 100, 200, 300],
    "max_depth":        [3, 5, 7, 9],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
random_search = RandomizedSearchCV(xgb_base, param_distributions=param_dist,
                                   n_iter=20, scoring="r2", cv=3, random_state=42)
```

### 5. Évaluation des modèles

#### Métriques de régression (MSE, R²)

| Modèle | MSE | R² |
|---|---|---|
| Régression Linéaire | 2.903025e+09 | ~0.67 |
| XGBoost (base) | 7.105386e+08 | ~0.92 |
| XGBoost optimisé | - | meilleur R² |

> Les métriques exactes sont disponibles dans les sorties du notebook.

#### Métriques de classification (analyse par classes de prix)

Le prix a été discrétisé en 3 classes (`bas`, `moyen`, `élevé`) via `pd.qcut` pour une analyse de classification complémentaire.

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| bas | 0.72 | 0.73 | 0.72 | 66 |
| moyen | 0.74 | 0.60 | 0.66 | 84 |
| élevé | — | — | — | — |
| **Accuracy globale** | | | **0.734** | |
| **Weighted avg** | **0.733** | **0.734** | — | |

#### Interprétation des performances

- **Accuracy = 73,4 %** → le modèle prédit correctement la classe de prix dans 73 % des cas.
- **Precision (weighted) = 0.733** → lorsqu'une classe est prédite, elle est correcte à ~73 %.
- **Recall (weighted) = 0.734** → le modèle capture environ 73 % de toutes les instances réelles.
- La classe `moyen` présente un recall plus faible (0.60), indiquant une confusion avec les classes adjacentes, ce qui est attendu pour une classe intermédiaire.
- L'XGBoost optimisé surpasse la régression linéaire sur toutes les métriques de régression.

### 6. Stockage dans le Model Registry

```python
reg = Registry(session=session, database_name="HOUSE_DB", schema_name="PUBLIC")
reg.log_model(
    model=best_model,
    model_name="HOUSE_PRICE_PREDICTOR",
    version_name="v1",
    metrics={"MSE": 0 , "R2": 0.88}
)
```

### 7. Inférence

- Chargement du modèle depuis le registry via `reg.get_model("HOUSE_PRICE_PREDICTOR").version("v1")`
- Génération de prédictions sur de nouvelles données
- Stockage des résultats dans la table `HOUSE_DB.PUBLIC.HOUSE_PRICE_PREDICTIONS`

### 8. Application Streamlit

Une application Streamlit déployée directement dans Snowflake permet aux utilisateurs métier de :
- Saisir les caractéristiques d'un bien via une interface graphique
- Obtenir une estimation de prix en temps réel
- Interagir avec le modèle sans connaissance technique

---

## Stack technique

| Outil | Usage |
|---|---|
| Snowflake | Plateforme de données, stockage, compute |
| Snowpark | Manipulation des données en Python |
| Snowflake ML Registry | Versionnage et déploiement des modèles |
| scikit-learn | Préparation des données, régression linéaire, optimisation |
| XGBoost | Modèle de régression principal |
| Streamlit in Snowflake | Interface utilisateur métier |
| S3 (AWS) | Source des données brutes |

---

## Livrables

- `NB_1.ipynb` : Notebook Snowflake contenant le pipeline ML complet
- Modèle entraîné enregistré dans le Snowflake Model Registry (`HOUSE_PRICE_PREDICTOR v1`)
- Application Streamlit déployée dans Snowflake
- Ce fichier `README.md`

---

## Contact

Envoi à : **axel@logbrain.fr**  
Objet : `MBAESG_MBDIA-2025-2026_Classe1_EVALUATION_DATAENGINEER_MLOPS`
