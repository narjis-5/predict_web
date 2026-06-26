# LOGI-PREDICT

**Application web d'aide à la décision logistique** : recalibrage de l'ETA (date de
livraison estimée) et classification de la durée de transit, à partir d'un modèle
d'apprentissage automatique entraîné sur des données e-commerce nigérianes.

> Le score de risque produit par l'application est un **indicateur opérationnel de
> priorisation**, pas une probabilité calibrée de retard individuel. C'est un parti pris
> scientifique assumé : sur ce jeu de données, l'écart à l'ETA est un bruit largement
> irréductible (voir la section « Lecture scientifique » plus bas).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20892821.svg)](https://doi.org/10.5281/zenodo.20892821)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Streamlit App](https://img.shields.io/badge/Streamlit-app%20en%20ligne-ff4b4b)](https://predictweb-d4jlpwxvkuw9njztmtjcrz.streamlit.app/)

🔗 **Application en ligne (déployée sur Streamlit Community Cloud) :**
<https://predictweb-d4jlpwxvkuw9njztmtjcrz.streamlit.app/>

📌 **Pour citer ce travail :** OURIZ, Narjisse (2026). *LOGI-PREDICT: Machine-Learning
Decision Support for E-Commerce Delivery-Time Performance*. Zenodo.
DOI : [10.5281/zenodo.20892821](https://doi.org/10.5281/zenodo.20892821)

---

## Aperçu

<!-- TODO : ajouter une capture d'écran de la page d'accueil dans docs/screenshot.png -->
![Page d'accueil de LOGI-PREDICT](docs/screenshot.png)

*(Si l'image ne s'affiche pas, lancez l'application localement — voir « Installation » — et
ajoutez votre propre capture sous `docs/screenshot.png`.)*

---

## Fonctionnalités

- **Upload** de fichiers tabulaires : CSV, TXT/TSV, Excel (`.xlsx`), Parquet, JSON.
- **Interface multilingue** : français, anglais, arabe (avec mise en page RTL).
- **Prédiction de la durée de livraison** via le modèle de régression entraîné
  (`delivery_duration_model_bundle.joblib` — `HistGradientBoostingRegressor`).
- **Classification de la durée** (standard vs longue) via le modèle de classification
  (`delivery_duration_class_model_bundle.joblib` — `LogisticRegression`).
- **Recalibrage de l'ETA** et tampon recommandé selon le niveau de service visé.
- **Table de décision** par axe (origine → destination) et par transporteur.
- **Validation experte** : split temporel passé → futur et évaluation diagnostique de la
  règle de dépassement d'ETA.
- **Graphiques décisionnels** : axes à risque, transporteurs sous tension, matrice
  risque/tampon, distribution de la marge ETA.
- **Assistant local multilingue** : explique les résultats du fichier déposé et les termes
  techniques (ETA, tampon, score de risque), sans API externe.

---

## Schéma de données attendu

Fournissez un fichier tabulaire contenant **au minimum** ces colonnes :

| Colonne                  | Type      | Description                                  |
|--------------------------|-----------|----------------------------------------------|
| `ship_date`              | date      | Date d'expédition                            |
| `expected_delivery_date` | date      | Date de livraison promise (ETA)              |
| `origin_city`            | texte     | Ville d'origine                              |
| `destination_city`       | texte     | Ville de destination                         |
| `logistics_company`      | texte     | Transporteur                                 |
| `supplier_name`          | texte     | Fournisseur                                  |
| `quantity`               | entier    | Quantité expédiée                            |
| `shipping_cost_ngn`      | nombre    | Coût d'expédition (Naira nigérian)           |

Colonnes **optionnelles** :

| Colonne                | Rôle                                                              |
|------------------------|------------------------------------------------------------------|
| `actual_delivery_date` | Sert seulement à comparer le prédit au réel ; **jamais** en entrée du modèle (anti-fuite). |
| `delivery_status`      | Statut de livraison, pour l'analyse descriptive uniquement.      |

Un échantillon prêt à l'emploi est fourni : [`data_example.csv`](data_example.csv) (30 lignes
au schéma exact). Déposez-le dans l'application pour la tester sans aucune donnée lourde.

---

## Installation

Prérequis : Python 3.10 ou plus récent.

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-compte>/logi-predict.git
cd logi-predict

# 2. (recommandé) Créer un environnement virtuel
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`. Aucune base de données ni clé d'API n'est
nécessaire : les deux modèles sont versionnés dans le dépôt et chargés au démarrage.

> **Compatibilité des modèles** : les bundles `.joblib` ont été sérialisés avec
> **scikit-learn 1.7.2**. Cette version est épinglée dans `requirements.txt` ; ne pas la
> changer sans ré-entraîner les modèles, sous peine d'échec de chargement.

---

## Source des données

Le modèle a été entraîné sur un **jeu de données synthétique** de chaîne logistique
e-commerce nigériane publié par **Electric Sheep Africa** sur **Hugging Face** :

- Dataset : *Nigerian Retail and E-commerce Supply Chain Logistics Data* (synthétique).
- Éditeur : Electric Sheep Africa.
- Licence du dataset : **GPL** — d'où le choix d'une licence GPL v3 pour ce projet.
- Lien : <https://huggingface.co/datasets?search=nigerian%20retail%20ecommerce%20supply%20chain>
  *(remplacer par l'URL exacte de la fiche du dataset).*

Le **dataset complet n'est pas publié** dans ce dépôt (il n'est pas nécessaire au
fonctionnement de l'application : les modèles entraînés suffisent). Pour reproduire
l'entraînement, téléchargez le dataset depuis Hugging Face et lancez `train_model.py`.

---

## Lecture scientifique honnête

Sur ces données, la part prévisible de la durée vient essentiellement de l'ETA promise ; la
part imprévisible (l'écart à l'ETA) est un bruit largement irréductible. En conséquence :

- la prédiction du **retard relatif** (réel > promesse) est, ici, **non discriminante** ;
- la **durée absolue** se prédit bien (recalibrage utile de l'ETA) ;
- l'apport réel de l'outil est donc le **recalibrage de l'ETA et la priorisation**, pas la
  détection fiable d'un retard individuel.

---

## Structure du dépôt

```text
app.py                                         Application Streamlit (autonome)
pipeline_utils.py                              Fonctions partagées (feature engineering, météo)
train_model.py                                 Script de ré-entraînement (nécessite le dataset)
delivery_duration_model_bundle.joblib          Modèle de régression (durée)
delivery_duration_class_model_bundle.joblib    Modèle de classification (durée standard/longue)
model_report.json                              Métriques et interprétation du modèle
data_example.csv                               Échantillon de test (schéma exact)
requirements.txt                               Dépendances épinglées
ml_academique/                                 Figures et tables de l'étude
```

---

## Déploiement sur Streamlit Community Cloud

1. Pousser ce dépôt sur GitHub (voir instructions Git ci-dessous).
2. Aller sur <https://share.streamlit.io> et se connecter avec GitHub.
3. **New app** → choisir le dépôt, la branche `main` et le fichier `app.py`.
4. **Deploy**. Streamlit installe `requirements.txt` et publie une URL publique.

> **ℹ️ Note Python 3.14 / pyarrow (résolu).** Streamlit Community Cloud déploie désormais en
> **Python 3.14**. Pour que tout s'installe en *wheels* (sans compilation `cmake` qui échoue
> sur l'image), `requirements.txt` épingle **`pyarrow==22.0.0`** (1ʳᵉ version avec wheel cp314)
> et **`streamlit==1.52.2`** (1ʳᵉ ligne qui relâche le plafond `pyarrow<22` imposé par 1.51).
> Aucun réglage manuel de version Python n'est donc nécessaire. Le fichier
> [`.python-version`](.python-version) (`3.13`) reste présent comme repli, mais Streamlit Cloud
> l'ignore : le sélecteur **Advanced settings → Python version** reste l'unique moyen de forcer
> une version si besoin.

---

## Licence

Ce projet est distribué sous licence **GNU General Public License v3.0** — voir le fichier
[`LICENSE`](LICENSE). Ce choix est compatible avec la licence GPL du jeu de données source.
