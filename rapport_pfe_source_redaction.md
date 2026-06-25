# Mémoire de fin d'études

## Sujet recommandé

**Entraînement d'un modèle de Machine Learning pour la prédiction et le recalibrage des délais de livraison e-commerce**

## Informations de page de garde

- Étudiante : OURIZ NARJISSE
- Encadrant : Pr. BENBERAHIM HOUSSAM
- Filière : Master en ingénierie de la décision
- Diplôme : Mémoire de fin d'études en vue d'obtention du diplôme de master en ingénierie de la décision
- Année universitaire : 2026-2027

## Plan final retenu

### Introduction générale

Problématique générale des délais de livraison e-commerce, fragilité de l'ETA, rôle des données historiques et justification d'une approche supervisée. L'introduction doit installer la tension centrale : le système logistique promet une date avant de disposer de toute l'information opérationnelle, alors que le client juge la qualité du service au moment où cette promesse est tenue ou rompue.

### Chapitre 1 - Logistique e-commerce, dernier kilomètre et problématique des délais

Ce chapitre constitue le premier bloc théorique. Il explique les transformations du commerce électronique, la fragmentation des flux B2C, le coût du dernier kilomètre, la congestion, la dépendance aux transporteurs et l'effet des retards sur la satisfaction client. Le chapitre doit intégrer un schéma d'Ishikawa des causes de retard.

Sections recommandées :

1. Mutation du commerce électronique et fragmentation des flux.
2. Dernier kilomètre et coûts de distribution.
3. ETA, promesse client et asymétrie d'information.
4. Causes systémiques des retards de livraison.
5. Synthèse du chapitre.

### Chapitre 2 - Machine Learning appliqué à la prédiction logistique

Ce chapitre constitue le second bloc théorique. Il relie la Logistique 4.0, la donnée opérationnelle, la prédiction supervisée et l'aide à la décision. Il ne doit pas seulement définir le Machine Learning, mais expliquer pourquoi la formulation de la cible détermine la validité scientifique du projet.

Sections recommandées :

1. Logistique 4.0 et exploitation des traces numériques.
2. Apprentissage supervisé : classification et régression.
3. Prévention des fuites de données et validation hors échantillon.
4. Interprétabilité, métriques et limites des modèles.
5. Synthèse du chapitre.

### Chapitre 3 - Méthodologie expérimentale, entraînement et évaluation du modèle

Ce chapitre porte le coeur scientifique du mémoire. Il décrit la source des données, les variables, la construction de la cible, la contre-épreuve sur le retard, la reformulation vers la durée réelle, le pipeline de prétraitement, l'entraînement, la comparaison aux baselines et la validation temporelle.

Sections recommandées :

1. Source des données et structure du fichier Parquet.
2. Audit, nettoyage et construction des variables.
3. Formulation initiale du retard et résultat nul.
4. Reformulation en prédiction de durée réelle.
5. Prétraitement, encodage et prévention des fuites.
6. Modèles candidats, validation croisée et optimisation.
7. Résultats, interprétabilité et validation temporelle.
8. Limites scientifiques.

### Chapitre 4 - Déploiement Streamlit et exploitation opérationnelle des résultats

Ce chapitre présente le site comme une phase de déploiement du modèle entraîné. Il ne devient pas le sujet principal du mémoire ; il sert à montrer comment la prédiction est transformée en indicateurs utilisables par un responsable logistique.

Sections recommandées :

1. Objectif du déploiement.
2. Architecture fonctionnelle de l'application Streamlit.
3. Import du fichier Parquet nigérian.
4. Calcul de la durée prédite, de la marge ETA et du score de priorité.
5. Table d'aide à la décision par axe et transporteur.
6. Analyse des résultats obtenus sur la donnée complète.
7. Assistant décisionnel local.
8. Limites du déploiement et conditions d'usage.

### Conclusion générale

La conclusion doit assumer le résultat principal : les variables disponibles permettent de prédire correctement la durée de transit, mais elles ne permettent pas de discriminer finement les retards individuels. La contribution réside dans l'entraînement d'un modèle robuste de recalibrage de l'ETA et dans son déploiement sous forme d'aide à la décision.

## Liste des abréviations

| Abréviation | Signification |
|---|---|
| IA | Intelligence Artificielle |
| ML | Machine Learning |
| ETA | Estimated Time of Arrival |
| EDA | Exploratory Data Analysis |
| KPI | Key Performance Indicator |
| API | Application Programming Interface |
| IoT | Internet of Things |
| BI | Business Intelligence |
| ETL | Extract, Transform, Load |
| CSV | Comma-Separated Values |
| JSON | JavaScript Object Notation |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient de détermination |
| AUC | Area Under the Curve |
| ROC | Receiver Operating Characteristic |
| PR-AUC | Precision-Recall Area Under Curve |
| SHAP | SHapley Additive exPlanations |
| HGBR | HistGradientBoostingRegressor |
| CV | Cross-Validation |
| 3PL | Third-Party Logistics |
| 4PL | Fourth-Party Logistics |

## Figures et tableaux à intégrer

### Figures issues du notebook et des calculs expérimentaux

- Figure 1 : `report_assets/figures/01_distribution_durees.png` - distribution des durées planifiées et réelles.
- Figure 4 : `report_assets/figures/02_planifie_vs_reel.png` - relation entre ETA planifiée et durée réelle.
- Figure 5 : `report_assets/figures/03_reel_vs_predit.png` - alignement entre prédiction et durée observée.
- Figure 6 : `report_assets/figures/04_comparaison_r2.png` - performance du modèle face au baseline métier.
- Figure 7 : `report_assets/figures/05_importance_permutation.png` - importance par permutation.
- Figure 8 : `report_assets/figures/06_matrice_confusion_regle_eta.png` - matrice de confusion de la règle ETA.
- Figure 9 : `report_assets/figures/07_distribution_marge_eta.png` - distribution de la marge ETA prédite.
- Figure 10 : `report_assets/figures/08_groupes_prioritaires.png` - groupes axe-transporteur prioritaires.

### Captures issues du site Streamlit

- Capture 1 : accueil, upload et indicateurs globaux.
- Capture 2 : aperçu ligne par ligne des prédictions.
- Capture 3 : aide à la décision opérationnelle.
- Capture 4 : table des tampons ETA.
- Capture 5 : axes à risque et risque par transporteur.
- Capture 8 : journal de préparation.
- Capture 9 ou 10 : modèle, validation experte et importance des variables.
- Capture 11 : assistant décisionnel local.

### Tableaux

- Tableau 1 : dictionnaire des variables du dataset.
- Tableau 2 : synthèse des métriques du modèle.
- Tableau 3 : métriques de la règle de dépassement ETA.
- Tableau 4 : synthèse des résultats du site.
- Tableau 5 : top 20 des groupes axe-transporteur prioritaires.

## Valeurs centrales à conserver

Le modèle final est un HistGradientBoostingRegressor entraîné avec un pipeline complet de prétraitement. Sur le test aléatoire, il atteint une MAE de 1,987 jour, une RMSE de 2,279 jours et un R² de 0,751. Le baseline métier fondé sur la durée planifiée atteint un R² de 0,639, ce qui montre que le modèle apporte un gain réel, mais non spectaculaire.

La validation temporelle donne une MAE de 1,979 jour, une RMSE de 2,274 jours et un R² de 0,751. Cette stabilité montre que le modèle ne dépend pas uniquement d'un découpage aléatoire favorable.

La règle de dépassement ETA obtient une AUC-ROC de 0,510 et un taux de fausses alertes de 100 %. Ce résultat confirme que la règle ne discrimine pas les retards individuels. La lecture scientifique doit donc rester prudente : le modèle recalcule mieux la durée probable qu'il ne détecte chaque retard.

Sur le site Streamlit, l'import du fichier nigérian donne 400 000 lignes brutes, 400 000 lignes exploitables dans l'interface, 100 % d'expéditions signalées, un score moyen de 78,7 %, une durée prédite moyenne de 9,04 jours, une durée estimée moyenne de 7,51 jours et une marge ETA moyenne de 1,54 jour.

## Début de l'introduction générale

La livraison e-commerce condense une difficulté que les systèmes logistiques traditionnels masquaient derrière la massification des flux : le délai n'est plus seulement une grandeur interne de planification, il devient une promesse visible, datée et évaluée par le client. La plateforme annonce une date avant que toutes les incertitudes du transport soient résolues, tandis que le consommateur juge l'expérience au moment où l'écart entre la promesse et la réception devient tangible. Cette tension transforme l'ETA en objet décisionnel, car elle relie la planification, le transporteur, la satisfaction client et le coût de traitement des réclamations.

L'enjeu scientifique du présent travail ne consiste pas à produire une alerte artificiellement flatteuse sur les retards. Il consiste à examiner ce que les données disponibles permettent réellement d'apprendre. Cette nuance oriente toute la méthodologie. Une première formulation du problème sous forme de classification du retard est testée, puis abandonnée lorsque les métriques montrent une absence de pouvoir discriminant. Le problème est alors reformulé en régression de la durée réelle de livraison, cible plus informative parce qu'elle conserve la structure temporelle du transit et permet de recalibrer l'ETA.

La démarche suit une chaîne expérimentale complète : audit du fichier Parquet, construction des variables disponibles avant la livraison, prévention des fuites, comparaison avec des baselines, validation croisée, optimisation des hyperparamètres, interprétabilité, validation temporelle et déploiement dans une interface Streamlit. Le site web n'est pas traité comme un élément décoratif du projet. Il matérialise la dernière étape du pipeline, celle où une prédiction numérique devient une décision opérationnelle : ajouter un tampon ETA, prioriser un axe ou communiquer plus tôt avec le client.

Le résultat principal impose une lecture honnête. Le modèle prédit la durée de transit avec une stabilité acceptable et dépasse le baseline métier, mais la règle de dépassement ETA reste faiblement discriminante. Cette limite n'affaiblit pas le projet ; elle en précise la portée. Les données disponibles permettent surtout de recalibrer l'ETA et de hiérarchiser les actions, non de désigner avec certitude les colis qui seront en retard.

## Références vérifiées à utiliser

Ces références sont réelles, retrouvables sur Google Scholar et disponibles via DOI, PDF éditeur ou page institutionnelle.

| N° | Référence | Lien utile |
|---|---|---|
| [1] | Lim, S. F. W. T., Jin, X., & Srai, J. S. (2018). Consumer-driven e-commerce: A literature review, design framework, and research agenda on last-mile logistics models. *International Journal of Physical Distribution & Logistics Management*. | https://doi.org/10.1108/IJPDLM-02-2017-0081 |
| [2] | Ranieri, L., Digiesi, S., Silvestri, B., & Roccotelli, M. (2018). A review of last mile logistics innovations in an externalities cost reduction vision. *Sustainability*, 10(3), 782. | https://doi.org/10.3390/su10030782 |
| [3] | Boysen, N., Fedtke, S., & Schwerdfeger, S. (2021). Last-mile delivery concepts: a survey from an operational research perspective. *OR Spectrum*, 43, 1-58. | https://doi.org/10.1007/s00291-020-00607-8 |
| [4] | World Economic Forum. (2020). *The Future of the Last-Mile Ecosystem: Transition Roadmaps for Public- and Private-Sector Players*. | https://www.weforum.org/reports/the-future-of-the-last-mile-ecosystem/ |
| [5] | Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4). | https://doi.org/10.1145/2382577.2382579 |
| [6] | Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27-32. | https://doi.org/10.1145/507533.507538 |
| [7] | Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182. | https://www.jmlr.org/papers/v3/guyon03a.html |
| [8] | Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*. | https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf |
| [9] | Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *The Annals of Statistics*, 29(5), 1189-1232. | https://doi.org/10.1214/aos/1013203451 |
| [10] | Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32. | https://doi.org/10.1023/A:1010933404324 |
| [11] | Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281-305. | https://www.jmlr.org/papers/v13/bergstra12a.html |
| [12] | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*. | https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html |





## Chapitre 1 - Logistique e-commerce, dernier kilomètre et problématique des délais

### Introduction

Le commerce électronique déplace la difficulté logistique vers l'aval du réseau. Dans un modèle de distribution traditionnel, les flux sont massifiés, planifiés par lots et absorbés par des points de vente capables de stocker temporairement l'incertitude. Dans le modèle e-commerce, la commande devient un colis individualisé, adressé à un client final, soumis à une promesse de livraison explicite et suivi à travers une chaîne de statuts numériques. Ce glissement transforme le délai en indicateur exposé, car la durée de transport n'est plus seulement une variable interne de planification ; elle devient une composante de l'expérience client [1].

La logistique du dernier kilomètre concentre cette tension. Elle intervient à la fin du parcours physique, mais elle supporte une part disproportionnée du coût, de l'incertitude et de la variabilité opérationnelle. Les travaux sur les modèles de livraison e-commerce décrivent ce segment comme un espace où la densité urbaine, la fragmentation des arrêts, la disponibilité du client, le choix du transporteur et la qualité des informations se combinent pour produire des délais difficilement stabilisables [2]. La prédiction des délais ne peut donc pas être réduite à une simple extrapolation de moyennes historiques ; elle suppose de comprendre les mécanismes qui rendent la promesse ETA fragile.

### 1.1 Mutation du commerce électronique et fragmentation des flux

La croissance du commerce électronique modifie la nature du flux logistique. Le réseau ne transporte plus seulement des volumes consolidés vers des magasins, mais une multitude de colis hétérogènes vers des lieux dispersés. Cette fragmentation augmente le nombre d'arrêts, réduit les possibilités de mutualisation et rend l'exécution plus sensible aux incidents locaux. Elle crée aussi une tension entre la promesse commerciale, souvent formulée en jours fixes, et la réalité d'un transport soumis à des variations territoriales, temporelles et organisationnelles [1].

La promesse de livraison occupe une place particulière dans cette configuration. Elle est formulée avant l'exécution complète du transport, mais elle est évaluée après coup par le client. Cette dissociation temporelle produit un risque structurel : le vendeur annonce une date à partir d'informations partielles, tandis que la livraison subit ensuite la congestion, le tri, la disponibilité du transporteur et les aléas du dernier kilomètre. Le délai devient alors un objet de coordination entre la plateforme, l'entrepôt, le transporteur et le destinataire.

Les exigences du cyberconsommateur accentuent ce mécanisme. La rapidité attendue n'agit pas seule ; elle s'articule avec la gratuité apparente, la transparence du suivi et la fiabilité de l'ETA. Une livraison rapide mais imprévisible peut dégrader la perception du service, tandis qu'une livraison plus lente mais correctement annoncée peut préserver la confiance. Dans cette logique, le problème n'est pas seulement d'aller vite, mais de formuler une promesse temporelle tenable.

### 1.2 Dernier kilomètre et coût opérationnel de l'incertitude

Le dernier kilomètre représente le segment où le réseau logistique cesse d'être principalement massifié. Chaque colis rejoint une adresse, une fenêtre horaire, un destinataire et un contexte local. Cette individualisation multiplie les micro-décisions : affectation du transporteur, ordonnancement de tournée, regroupement des colis, arbitrage entre rapidité et coût, traitement des échecs de première présentation. Les revues opérationnelles du dernier kilomètre montrent que ces décisions forment un système dense, où l'optimisation locale d'un critère peut déplacer la contrainte vers un autre [3].

Le coût du dernier kilomètre ne provient donc pas uniquement de la distance. Il naît de la faible densité utile des arrêts, des temps improductifs, des retours, des reprogrammations et de la difficulté à prédire l'accessibilité réelle du client. Les innovations observées dans la littérature, telles que les points relais, les consignes, les micro-hubs ou le crowdshipping, visent moins à supprimer l'incertitude qu'à la déplacer vers des configurations plus contrôlables [2].

La figure 1 représente le diagramme d'Ishikawa des facteurs de dépassement de l'ETA. Elle synthétise les familles de causes qui peuvent perturber la livraison : planification, transporteur, infrastructure, données, client, météo, produit et coût. Ce schéma n'a pas pour rôle de multiplier les causes possibles, mais de montrer pourquoi un retard individuel est difficile à prédire lorsque plusieurs familles agissent simultanément.

**Figure à insérer :** `report_assets/figures/09_ishikawa_retards_eta.png`

La figure 4 illustre la chaîne de formation de la promesse ETA. Elle montre que l'ETA n'est pas une simple date affichée au client, mais le résultat d'un enchaînement entre commande, promesse commerciale, préparation, transport, dernier kilomètre et réception. Les perturbations peuvent apparaître à plusieurs niveaux, ce qui explique pourquoi le modèle doit traiter la durée comme une grandeur recalibrable.

**Figure à insérer :** `report_assets/figures/21_theorie_chaine_eta.png`

Le diagramme éclaire directement la suite méthodologique du mémoire. Si la base de données ne contient qu'une partie de ces familles, le modèle ne peut apprendre qu'une partie du phénomène. Une variable comme la durée planifiée peut résumer implicitement certaines décisions amont, mais elle ne capture pas les perturbations en temps réel, les incidents de tournée ou l'absence du destinataire. Cette limite explique pourquoi la classification directe du retard peut échouer, même lorsque la prédiction de la durée moyenne reste acceptable.

### 1.3 ETA, promesse client et asymétrie d'information

L'ETA fonctionne comme une promesse contractuelle faible et comme un signal de qualité. Elle n'est pas toujours juridiquement contraignante, mais elle structure l'attente du client, la charge du service après-vente et la perception de fiabilité. Lorsqu'elle est trop optimiste, elle crée un écart entre l'information affichée et l'expérience vécue. Lorsqu'elle est trop prudente, elle peut réduire l'attractivité commerciale de l'offre. La décision pertinente consiste donc à calibrer l'ETA assez finement pour concilier satisfaction client et capacité opérationnelle.

Cette calibration dépend de la qualité des informations circulant entre les acteurs. Le e-commerçant connaît la commande, le fournisseur, la date d'expédition et l'ETA annoncée. Le transporteur connaît davantage l'état réel du réseau, mais cette information n'est pas toujours transmise de manière complète ou instantanée. La littérature sur les modèles de livraison souligne que la performance du dernier kilomètre repose autant sur la coordination informationnelle que sur l'exécution physique [1].

La boîte noire logistique apparaît lorsque le colis quitte l'espace contrôlé par le vendeur sans que les informations de transport soient suffisamment précises pour anticiper les anomalies. Dans ce cas, le retard n'est découvert qu'après sa matérialisation, souvent au moment où le client réclame une explication. La donnée historique permet alors de reconstruire des régularités, mais elle ne remplace pas l'observation en temps réel. Cette distinction sera déterminante dans l'interprétation des résultats expérimentaux.

Le tableau 1.1 représente les principales familles d'incertitude du dernier kilomètre et leur effet sur la promesse ETA. Il relie directement le cadrage théorique au choix méthodologique : lorsque les causes sont multiples et partiellement invisibles, la prédiction individuelle du retard devient fragile, tandis que le recalibrage de la durée reste plus réaliste.

| Famille d'incertitude | Mécanisme logistique | Effet sur l'ETA | Lecture pour le modèle |
|---|---|---|---|
| Demande | Pics de commandes et saisonnalité | Surcharge temporaire des capacités | Signal partiellement capté par les dates |
| Transporteur | Disponibilité de flotte et qualité de service | Dispersion des temps de transit | Signal faible à interpréter prudemment |
| Territoire | Distance, congestion et accessibilité | Allongement du dernier kilomètre | Effet résumé par les axes origine-destination |
| Client | Absence, report ou adresse imprécise | Échec de première livraison | Cause souvent non observée dans les données |
| Météo | Pluie, chaleur, vent ou conditions extrêmes | Ralentissement ponctuel des flux | Enrichissement utile mais insuffisant seul |

### 1.4 Causes systémiques des retards de livraison

Les retards peuvent être classés en deux familles. Les retards structurels sont récurrents, liés à une planification trop optimiste, à un axe territorial instable, à un transporteur moins régulier ou à une capacité insuffisante. Les retards conjoncturels sont plus dispersés : pluie intense, incident routier, panne, absence du client, surcharge ponctuelle d'un centre de tri. La première famille peut être partiellement apprise par un modèle historique ; la seconde exige des données dynamiques rarement présentes dans les fichiers transactionnels.

Le dataset étudié dans ce mémoire contient des variables utiles avant livraison : villes d'origine et de destination, transporteur, fournisseur, coût d'expédition, quantité, date d'expédition et ETA planifiée. Il ne contient pas les statuts de tournée en temps réel, les positions GPS, les incidents routiers, les scans intermédiaires ou les confirmations client. Cette composition rend scientifiquement plausible une prédiction de durée, mais elle rend beaucoup plus fragile une prédiction binaire du retard. L'expérience menée au Chapitre 3 confirmera cette différence.

La distinction entre durée et retard est centrale. La durée réelle de livraison conserve une structure continue : un colis peut mettre 5, 8 ou 13 jours, et cette grandeur reste fortement liée à la durée planifiée. Le retard binaire, lui, dépend d'un seuil : il suffit que la durée réelle dépasse l'ETA pour passer de 0 à 1. Ce changement de nature peut effacer une partie du signal, surtout lorsque l'ETA elle-même résume déjà des décisions de planification imparfaites.





<!-- OLD_MEMOIRE_EXPANSION_START -->
### 1.5 Approfondissement théorique intégré depuis le mémoire initial

Le développement suivant reprend et réorganise le cadre théorique déjà rédigé dans le mémoire initial. Il renforce le chapitre sans modifier son axe : comprendre pourquoi le dernier kilomètre e-commerce rend la promesse ETA instable et difficile à contrôler.

L'Émergence de l'E-Commerce et la Complexité Logistique du Dernier Kilomètre Ce chapitre examine comment la numérisation et la croissance du commerce en ligne ont transformé la logistique finale. Nous analysons l'évolution macroéconomique des flux physiques B2C, les défis d'encombrement urbain et les nouvelles configurations des réseaux de transport [1] , [2] , [10]. Propos liminaire du chapitre La bascule du commerce physique vers le flux numérique ne se limite pas à un transfert de canal de vente ; elle redistribue la charge logistique vers l'aval et concentre la difficulté sur le segment terminal du transport [1], [2]. Chaque commande unitaire expédiée vers un domicile transforme un réseau pensé pour des volumes massifiés en une trame dispersée de points de livraison, et cette dispersion agit comme un multiplicateur de coût et d'incertitude [2], [3]. Nous abordons cette mutation non comme une simple croissance de la demande, mais comme un changement de régime où la variabilité des arrivées de commandes devient la contrainte qui gouverne le travail des planificateurs [1], [4].

Le dernier kilomètre se laisse alors décrire comme une interface de couplage entre un flux physique et un flux d'information, et non comme un maillon isolé du réseau [7], [10]. La performance perçue par le client se construit moins dans l'entrepôt que dans la capacité du transporteur à honorer une promesse temporelle formulée en amont, ce qui déplace le centre de gravité de la valeur vers la coordination des données [10], [11]. Cette grille de lecture irrigue notre étude de cas, puisque les variables exploitées par le modèle prédictif, du transporteur retenu à l'axe emprunté et au délai estimé, dérivent directement de cette interface de couplage [13], [16]. L'intégration omnicanale ajoute une couche de complexité, car la promesse de visibilité en temps réel exige une synchronisation des systèmes d'information que peu de réseaux maîtrisent de bout en bout [1], [10]. Lorsque cette synchronisation se rompt, l'écart entre l'information affichée au client et l'état réel du colis nourrit la défiance et prépare le terrain au litige [5], [10].

Le fil conducteur retenu relie ainsi trois mécanismes qui se renforcent mutuellement : la fragmentation de la demande qui sature les réseaux, la structure de coût qui rend le segment terminal disproportionnellement onéreux, et l'asymétrie informationnelle qui prive l'expéditeur de visibilité dès la sortie d'entrepôt [3], [7], [12]. 1.1. La dynamique du commerce électronique et les nouvelles exigences omnicanales Cette section pose les bases de l'évolution du e-commerce et de la psychologie des cyberconsommateurs. Nous analysons la fragmentation des flux de livraison et la volatilité de la demande en ligne [1], [4]. 1.1.1.

Évolution macroéconomique et structurelle du e-commerce Nous abordons ici les mutations structurelles de la distribution et l'atomisation des colis. Ce passage du commerce physique au flux unitaire redéfinit la capillarité urbaine [2], [3]. 1.1.1.1. Historique, indicateurs de croissance globale et transition du commerce physique vers le flux unitaire B2C La transition des circuits de distribution traditionnels vers le commerce électronique a profondément restructuré l'organisation globale des flux physiques. La croissance exponentielle des transactions en ligne, intensifiée par les mesures sanitaires et les confinements, a provoqué une mutation des chaînes logistiques vers des modèles plus agiles et orientés client [1], [3].

Historiquement conçues pour le transport de marchandises massifiées (palettes et conteneurs) vers des points de vente physiques, les infrastructures logistiques doivent désormais soutenir une atomisation extrême des flux [2]. Cette transition vers le modèle Business-to-Consumer (B2C) se caractérise par l'expédition de colis individuels à destination d'une multitude de domiciles privés, transformant chaque foyer en un point terminal de livraison [2], [6]. Il en résulte une hausse majeure de la capillarité des transports urbains et une complexification des opérations de groupage et de consolidation [3]. 1.1.1.2. L'impact de la numérisation des parcours d'achat sur la volatilité et l'imprévisibilité de la demande logistique La dématérialisation des parcours d'achat a introduit une variabilité structurelle difficilement gérable pour les planificateurs de flottes.

Les commandes en ligne se caractérisent par une taille unitaire restreinte, une dispersion géographique des clients et une irrégularité marquée des arrivées de commandes [1]. De plus, les pics saisonniers de demande (comme les périodes de fêtes ou le Black Friday ) saturent temporairement les réseaux [1], [3]. La transition des détaillants vers des stratégies omnicanales ( Omni-Channel ), où les clients naviguent sans couture entre canaux physiques et numériques, impose une intégration complète des systèmes informatiques et logistiques afin de maintenir une visibilité en temps réel sur les stocks [1]. Cette numérisation accroît la sensibilité des acheteurs aux conditions de service, les rendant plus volatils : un parcours d'achat insatisfaisant ou des options de livraison limitées incitent immédiatement le cyberconsommateur à se tourner vers des plateformes concurrentes [4], [5]. Finalement, cette évolution vers le B2C génère des flux fragmentés complexes qui surchargent temporairement les réseaux logistiques [1], [3].

1.1.2. Les exigences temporelles et psychologiques du "cyberconsommateur" Cette partie traite de l'évolution des attentes des clients concernant le temps de trajet. L'immédiateté perçue pousse les vendeurs à compresser les délais [5], [6]. 1.1.2.1. Le concept d'immédiateté perçue et l'évolution sociologique de la valeur du temps de transport La psychologie de l'acheteur en ligne s'organise désormais autour de l'immédiateté perçue, réduisant la tolérance psychologique face aux délais de transit.

L'expérience d'achat ne se limite plus à l'acte d'acquisition sur la plateforme web, mais englobe la phase de livraison physique comme un jalon critique de la satisfaction globale [5]. Cette évolution comportementale se traduit par une demande croissante pour des délais compressés, tels que la livraison le lendemain ( next-day ) ou le jour même ( same-day ) [5], [6]. Sociologiquement, le temps d'attente est perçu comme un coût d'opportunité par le consommateur, ce qui pousse les e-commerçants à réviser leurs schémas de planification. Néanmoins, cette exigence de rapidité extrême engendre des externalités environnementales négatives substantielles, en raison de l'utilisation inefficace de la capacité des véhicules de transport [2], [6]. 1.1.2.2.

La rapidité, la gratuité apparente et la fiabilité de la livraison comme critères majeurs de différenciation concurrentielle Les attributs logistiques du dernier kilomètre constituent aujourd'hui le principal levier de différenciation stratégique pour les détaillants en ligne. Le cyberconsommateur opère un arbitrage complexe entre le coût de la livraison et sa rapidité. Bien qu'exigeant une exécution rapide, il manifeste une forte sensibilité aux frais de transport, dont la tolérance varie selon son profil socio-démographique, son revenu ou son genre [4]. Les retards de livraison et les échecs lors de la première présentation du colis altèrent gravement la qualité de service perçue et augmentent le taux de réclamation [5], [6]. Face à un incident de livraison, le client insatisfait a tendance à migrer silencieusement vers la concurrence sans notifier le vendeur [5].

Dès lors, l'établissement de créneaux horaires précis, l'accès à un suivi en temps réel et la diversification des modes de réception (points relais, consignes automatiques) s'avèrent indispensables pour fidéliser les acheteurs [4], [6]. En résumé, l'exigence de rapidité et la sensibilité aux frais de port forcent les e-commerçants à diversifier leurs options de réception [4], [6]. Pour conclure, l'analyse montre que la satisfaction et la fidélisation des acheteurs dépendent directement de la vitesse et de la fiabilité des livraisons [4], [5]. Synthèse conceptuelle des variables d'influence Le tableau ci-dessous synthétise les principales variables caractérisant la dynamique du e-commerce et leurs impacts opérationnels identifiés dans la littérature analysée : Variable d'influence Nature opérationnelle / Impact logistique Conséquences sur la Supply Chain Références Structure des commandes Fragmentée (colis unitaires, faible volume par commande). Nécessité de consolidation des flux, augmentation de la capillarité urbaine.

[1], [2] Volatilité de la demande Saisonalité accentuée, pics imprévisibles, irrégularité des transactions. Surcharge temporaire des réseaux de transport et des entrepôts de tri. [1], [3] Valeur du temps (ETA) Attente minimale tolérée (exigence de livraison J+1 ou jour même). Compression des fenêtres de livraison, hausse des coûts de distribution. [5], [6] Arbitrage tarifaire Forte sensibilité aux frais de port, attente de gratuité apparente.

Risque d'abandon de panier, pressions sur les marges des transporteurs. [4] Modes de réception Multiplicité (domicile, travail, consignes automatiques, points relais). Complexification du routage des véhicules (VRP multi-options). [4], [6] Schéma conceptuel : La boucle de satisfaction du cyberconsommateur Ce schéma DOT (Graphviz) modélise la dynamique d'interaction entre la numérisation des achats, la planification logistique et la satisfaction client : 1.2. La logistique du dernier kilomètre (Last-Mile Delivery) Nous analysons ici la structure du dernier kilomètre, son coût élevé et son impact environnemental.

L'objectif est de comprendre pourquoi cette étape est à la fois la plus courte et la plus difficile [7], [9]. 1.2.1. Délimitation opérationnelle et positionnement stratégique dans la Supply Chain Nous étudions le positionnement du dernier kilomètre et la répartition de ses charges financières. Les inefficacités opérationnelles y provoquent un paradoxe des coûts [7], [8]. 1.2.1.1.

Délimitation opérationnelle de la livraison finale : du dernier hub de tri régional au point de livraison final La logistique du dernier kilomètre, communément désignée sous le vocable anglophone de Last-Mile Logistics (LML), constitue la phase terminale et la plus critique de la chaîne logistique globale [7], [8]. D'un point de vue fonctionnel, cette étape s'amorce au niveau du goulot d'étranglement que représente le dernier centre de tri ou hub de distribution régional, pour se clore par la remise physique du colis au destinataire final [2], [7]. Cette livraison peut s'opérer directement au domicile de l'acheteur ou via des points de consolidation alternatifs tels que des casiers automatisés ou des commerces partenaires faisant office de points relais [6], [8]. La transition d'une logistique traditionnelle de gros volumes vers une atomisation extrême des flux e-commerce a engendré une reconfiguration radicale des réseaux de distribution, modifiant les exigences d'agilité et de réactivité des transporteurs [1], [7]. Ainsi, cette segmentation terminale ne se limite pas à un simple déplacement physique, mais constitue une interface dynamique déterminant la satisfaction globale de l'expérience d'achat [5], [8].

1.2.1.2. Analyse de la structure des coûts : le paradoxe économique du dernier kilomètre La gestion financière de cette phase terminale met en exergue une asymétrie de coût saisissante, désignée sous le terme de paradoxe pécuniaire du dernier kilomètre [2], [3]. Bien que cette étape représente la distance physique la plus succincte du parcours d'acheminement, elle engendre à elle seule la portion la plus substantielle des charges logistiques totales, oscillant fréquemment entre 50 % et 53 % des coûts d'expédition totaux [7], [9]. Cette inefficience économique est exacerbée par la dispersion spatio-temporelle des destinataires, qui contraint les véhicules à effectuer des tournées hautement fragmentées caractérisées par de fréquents arrêts [8], [9]. De surcroît, le taux d'échec de livraison lors de la première présentation du colis constitue un inducteur de coût majeur, obligeant les prestataires à planifier des passages itératifs ou à supporter les frais logistiques de retour [6], [7].

Face à cette compression des marges financières, les détaillants en ligne sont contraints de concevoir des solutions opérationnelles viables pour concilier les attentes de gratuité des consommateurs et la viabilité économique de leur réseau [4], [8]. Au final, la fragmentation des livraisons explique pourquoi la livraison finale représente la part la plus élevée du coût de transport global [7], [9]. 1.2.2. Les barrières structurelles et les défis de la distribution urbaine Nous examinons comment la densité urbaine et la congestion routière freinent les tournées. Les livraisons doivent aussi s'adapter aux zones à faibles émissions (ZFE) [8], [9].

1.2.2.1. Densité démographique, saturation des infrastructures routières et phénomènes de congestion L'essor exponentiel du commerce en ligne a intensifié le flux de véhicules de transport au sein des métropoles, se heurtant à la saturation structurelle des infrastructures routières [2], [7]. La densité démographique et l'essor des livraisons unitaires à domicile induisent une congestion anthropique croissante, ralentissant la vélocité des tournées de livraison [7], [9]. Cette densification se traduit par une hausse des temps de parcours et une dégradation notable de la fiabilité des prévisions de délais de livraison, perturbant les promesses clients d'ETA [5], [6]. Les prévisions estiment d'ailleurs que la demande de livraison urbaine va accroître le volume global de véhicules de livraison de plus de 36 % dans les grandes agglomérations mondiales à l'horizon 2030, entraînant une hausse mécanique des embouteillages de plus de 21 % [9].

La gestion de cette saturation routière impose aux planificateurs logistiques d'ajuster dynamiquement leurs itinéraires de distribution pour éviter les zones d'engorgement [8], [9]. 1.2.2.2. Nouvelles contraintes environnementales, zones à faibles émissions (ZFE) et réglementations municipales d'accès Parallèlement aux entraves physiques de la circulation, les municipalités imposent un nouveau paradigme réglementaire pour atténuer les externalités environnementales délétères du transport (émissions de CO2, pollution sonore, gaz à effet de serre) [3], [9]. L'instauration progressive de zones à faibles émissions (ZFE) et de restrictions d'accès temporelles ou dimensionnelles pour les véhicules thermiques contraint les opérateurs logistiques à réviser leur flotte de véhicules [8], [9]. Pour se conformer à ces prescriptions légales, les prestataires recourent de plus en plus à des vecteurs de transport décarbonés, tels que les véhicules électriques légers ou les vélos-cargos, qui s'avèrent particulièrement adaptés aux centres-villes denses [7], [8].

De surcroît, cette décarbonation urbaine exige l'implantation de micro-hubs de proximité pour faciliter le transbordement et la consolidation des flux de marchandises avant la distribution finale [3], [8]. Pour finir, la congestion et les réglementations environnementales imposent l'adoption de véhicules propres et de micro-hubs [7], [8]. En synthèse, le dernier kilomètre représente un défi économique et écologique majeur qui nécessite des régulations municipales adaptées [2], [7]. Synthèse conceptuelle des contraintes du dernier kilomètre Le tableau ci-dessous synthétise les principaux défis, impacts économiques et opérationnels liés à la logistique du dernier kilomètre : Facteur de complexité Nature opérationnelle / Défi identifié Conséquences sur la structure de coût Solutions de remédiation Références Atomisation des flux Dispersion extrême des livraisons unitaires à domicile. Hausse des distances parcourues, inefficacité des tournées.

Implantation de micro-hubs urbains de tri. [2], [7], [8] Paradoxe financier Représente 50 % à 53 % du coût total de transport de la Supply Chain. Compression drastique des marges opérationnelles des e-commerçants. Consolidation via points relais ou consignes autonomes. [7], [9] Échecs de livraison Absence du destinataire final lors du premier passage du transporteur.

Surcharge logistique, gestion des retours et seconds passages coûteux. Planification de créneaux horaires interactifs (ETA). [6], [7] Congestion urbaine Saturation routière provoquée par l'augmentation de 36 % des véhicules d'ici 2030. Augmentation des délais de transit, retards réguliers de livraison. Optimisation algorithmique et routage dynamique des véhicules.

[7], [9] Réglementation environnementale Restriction d'accès aux centres-villes denses via l'instauration des ZFE. Coûts d'investissement pour la mise à niveau des flottes de livraison. Déploiement de flottes décarbonées (vélos-cargos, véhicules électriques). [8], [9] Schéma conceptuel : Flux physiques, obstacles et alternatives de réception du dernier kilomètre Ce schéma DOT (Graphviz) modélise la transition physique des flux du dernier kilomètre, en 1.3. La gouvernance du transport: Modèles d'externalisation et flux d'information Cette section étudie les différentes manières de structurer le réseau de transport final.

Nous présentons les réseaux directs, multi-échelons et les alternatives innovantes de livraison [10], [11]. 1.3.1. Les prestataires logistiques spécialisés (3PL et 4PL) Nous décrivons ici les architectures physiques des réseaux, des hubs régionaux jusqu'aux points de livraison. Chaque schéma de réseau présente ses propres contraintes d'optimisation [10], [11]. 1.3.1.1.

Rôle opérationnel, exécution physique et gestion des flottes par les acteurs 3PL (Third-Party Logistics) La gestion opérationnelle des flux e-commerce repose historiquement sur l'externalisation stratégique des activités de stockage et de transport à des prestataires de services logistiques de type Third-Party Logistics (3PL) [10], [12]. Ces intermédiaires assument l'exécution physique des opérations logistiques, englobant la manutention des colis, le groupage au sein des terminaux et la gestion directe de leurs flottes de transport [10], [12]. Apparus lors des vagues de dérégulation du transport dans les années 1980, les 3PL jouent un rôle central dans la mutualisation des capacités physiques et la réduction des coûts unitaires d'expédition [10], [12]. Néanmoins, face à l'essor des parcours omnicanaux caractérisés par une atomisation des commandes et des exigences de livraison compressées, les 3PL font face à un défi d'adaptation technologique majeur pour s'intégrer au paradigme de l'Industrie 4.0 [1], [10]. La simple mise à disposition d'infrastructures physiques ne suffit plus, contraignant les 3PL à intégrer des services numériques avancés pour demeurer compétitifs [10], [12].

1.3.1.2. Émergence, intermédiation et fonctions d'intégration technologique des acteurs 4PL (Fourth-Party Logistics) Pour surmonter les limites de l'externalisation classique et coordonner des chaînes de distribution devenues hautement complexes, de nouveaux modèles d'orchestration ont émergé, portés par les acteurs de type Fourth-Party Logistics (4PL) [11], [12]. Contrairement aux 3PL, les prestataires 4PL opèrent généralement comme des intégrateurs neutres et sans actifs physiques ( asset-light ou asset-free ), dont le cœur de métier réside dans la gouvernance informationnelle et l'assemblage de solutions logistiques sur mesure [11], [12]. Ils font office de point de contact unique, coordonnant plusieurs prestataires de transport (3PL) pour le compte des donneurs d'ordres du e-commerce [11], [12]. L'apport principal des 4PL réside dans leur capacité d'intégration technologique, s'appuyant sur des plateformes numériques pour optimiser l'affectation dynamique des flux en temps réel et harmoniser les performances opérationnelles tout au long de la chaîne d'approvisionnement [11], [12].

Cette intermédiation numérique redéfinit la valeur ajoutée dans le secteur logistique, déplaçant le centre de gravité de la possession d'actifs physiques vers l'exploitation intelligente des données [10], [11]. En conclusion, les modèles à multi-échelons permettent de réduire les distances parcourues mais augmentent les coûts de manutention [10], [11]. 1.3.2. L'asymétrie informationnelle entre e-commerçants et transporteurs Nous analysons les consignes automatiques, les points relais et le crowdshipping. Ces solutions visent à réduire les échecs lors du premier passage du livreur [11], [12].

1.3.2.1. Les protocoles d'échange de données de traçabilité (EDI, API, statuts de livraison) La gouvernance efficace de la distribution e-commerce requiert une synchronisation continue des flux physiques et des flux d'information, s'appuyant sur des protocoles d'interopérabilité numérique [10], [12]. Historiquement, l'échange de données informatisé (EDI) constituait le standard industriel pour le transfert structuré de documents commerciaux et d'informations logistiques à intervalles prédéfinis [10], [12]. Toutefois, la rigidité de l'EDI s'avère insuffisante face aux exigences de réactivité de la distribution terminale, favorisant l'émergence des interfaces de programmation d'applications (API) [10], [12]. Les API permettent un échange bidirectionnel et instantané des données de traçabilité, connectant directement les systèmes d'information des e-commerçants avec ceux des prestataires de transport pour mettre à jour les statuts de livraison en temps réel [11], [12].

Cette intégration logicielle fluide constitue un facteur clé de performance, permettant d'automatiser les alertes d'ETA et de fluidifier le suivi pour le destinataire final [6], [10]. 1.3.2.2. Le phénomène de la "boîte noire" logistique et la perte de visibilité sur le flux après la sortie d'entrepôt Malgré la prolifération des technologies d'intégration, les chaînes logistiques souffrent fréquemment d'une dissymétrie cognitive, matérialisée par le phénomène de la « boîte noire » logistique [10], [12]. Ce dysfonctionnement se caractérise par une perte de visibilité et d'opacité informationnelle dès lors que le colis franchit les portes de l'entrepôt d'origine pour être pris en charge par le transporteur final [11], [12]. L'asymétrie informationnelle qui en résulte limite la capacité du e-commerçant à anticiper les anomalies logistiques en cours de route et à avertir de manière proactive le consommateur [10], [12].

Cette absence de traçabilité continue engendre une détérioration de la satisfaction client et surcharge les services après-vente, les e-commerçants étant souvent incapables de localiser précisément les envois en transit [5], [10]. La prédiction algorithmique des anomalies logistiques vise précisément à briser cette boîte noire en estimant la ponctualité des livraisons à partir de variables exogènes avant le démarrage physique de la tournée [7], [10]. Finalement, ces alternatives flexibles diminuent les kilomètres parcourus et améliorent l'empreinte carbone des livraisons [6], [12]. En résumé, le choix d'une configuration réseau dépend de la densité de clientèle et des coûts de stockage associés [10], [12]. Synthèse comparative des modèles d'externalisation et de gouvernance de l'information Le tableau ci-dessous confronte les attributs et limites opérationnelles des différents modèles de gouvernance du transport en e-commerce : Modèle / Technologie Rôle opérationnel Niveau d'intégration technologique Avantages logistiques Limites et points de vigilance Références Prestataires 3PL Exécution physique du transport, gestion des entrepôts et flottes.

Faible à modéré (dépendance des outils traditionnels WMS/TMS). Mutualisation des capacités physiques, réduction des coûts opérationnels. Manque de réactivité face aux demandes unitaires dynamiques. [10], [12] Prestataires 4PL Intégration neutre, orchestration globale des chaînes logistiques. Très élevé (plateformes numériques de coordination).

Visibilité de bout en bout, optimisation dynamique des ressources. Dépendance totale à l'interopérabilité des systèmes d'information. [11], [12] Protocoles EDI Transfert standardisé de fichiers structurés par lots ( batch ). Modéré (systèmes rigides et synchronisations asynchrones). Standard industriel hautement sécurisé et éprouvé.

Latence dans l'actualisation des données de traçabilité. [10], [12] Interfaces API Échange de flux de données instantané et dynamique en temps réel. Élevé (requiert une infrastructure de services web moderne). Visibilité immédiate des statuts, intégration logicielle fluide. Risques liés à la sécurité des données et à la stabilité des serveurs.

[10], [11], [12] Boîte noire logistique Perte d'information durant la phase de transit terminal. Nul (rupture du flux informationnel post-entrepôt). Aucun avantage (subi par le donneur d'ordres). Clients insatisfaits, hausse du churn silencieux et litiges. [5], [10], [12] Schéma conceptuel : Flux informationnels et exécution logistique dans le transport e-commerce Ce schéma DOT (Graphviz) modélise la gouvernance de l'information, l'interopérabilité API et la lutte contre l'asymétrie informationnelle : En conclusion, ce premier chapitre a montré que le commerce en ligne impose des contraintes physiques et écologiques majeures sur le dernier kilomètre.

Face à ces contraintes, les entreprises doivent réorganiser leurs réseaux et proposer des alternatives de livraison plus flexibles [3], [8], [12]. Bilan analytique du chapitre Trois enchaînements structurent cette première lecture. La fragmentation des flux B2C démultiplie les points de contact et fait du segment terminal le poste où se concentrent les charges et les aléas [2], [7]. La structure de coût, où la livraison finale absorbe une part majoritaire des dépenses de transport, convertit toute inefficacité de tournée en perte directe de marge [7], [9]. L'asymétrie informationnelle prive enfin l'expéditeur de toute capacité d'anticipation, puisque le colis quitte le champ de visibilité dès qu'il franchit les portes de l'entrepôt [10], [12].

Ces enchaînements convergent vers une même conséquence : le retard cesse d'être un incident isolé pour devenir une propriété émergente du réseau lui-même [2], [10]. Tant que la coordination des données demeure lacunaire, l'expéditeur subit les déviations au lieu de les corriger en amont, ce qui motive le déplacement vers une logique d'anticipation que la suite du mémoire formalise [12], [13]. La visibilité dégradée et la saturation des réseaux s'alimentent réciproquement, de sorte qu'un pic de demande non absorbé amplifie les surcoûts du dernier kilomètre au lieu de les diluer [9], [12]. L'examen des causes profondes du retard, de leur typologie et de leurs répercussions financières prolonge cette analyse, car identifier les inducteurs précis du dépassement de délai conditionne la sélection des variables explicatives du modèle [16], [19]. C'est cette cartographie causale, et non une description générale des flux, qui alimente la phase expérimentale conduite sur les quatre cent mille expéditions du jeu de données [13], [16].

La Problématique des Délais de Livraison et l'Analyse Systémique des Retards Ce deuxième chapitre se concentre sur les retards de livraison dans le commerce en ligne. Nous modélisons les indicateurs de performance clés (OTD, AD) et analysons les causes structurelles et financières des anomalies [13], [14], [19]. Propos liminaire du chapitre Mesurer le respect des délais revient à instrumenter une promesse : l'écart entre l'instant réel d'arrivée et l'instant promis fonctionne comme un signal contractuel dont le franchissement enclenche une cascade comportementale chez le client [5], [13]. Nous traitons donc l'indicateur de ponctualité non comme une statistique descriptive, mais comme le point de bascule à partir duquel se forment la réclamation, le refus de colis et l'attrition [13], [15]. Cette lecture conditionne la manière dont la variable cible est construite dans l'étude de cas, puisqu'elle sépare l'anomalie tolérée de l'anomalie destructrice de valeur [15], [19].

La distinction entre retard structurel et retard conjoncturel ne relève pas d'une simple nomenclature ; elle oppose deux processus générateurs distincts, l'un récurrent et imputable au réseau, l'autre aléatoire et exogène [7], [14]. Un modèle prédictif n'apprend pas de la même façon ces deux régimes, car le premier laisse une trace stable dans les données historiques tandis que le second se manifeste comme un bruit à longue traîne [14], [16]. Isoler le signal structurel, qui est apprenable, du bruit conjoncturel, qui ne l'est qu'imparfaitement, constitue l'enjeu méthodologique qui oriente la cartographie développée ensuite [13], [14]. Le chapitre relie trois plans complémentaires : la formalisation des indicateurs de performance, l'identification des inducteurs géospatiaux, organisationnels et temporels du retard, puis la chaîne de coûts qu'un dépassement déclenche [16], [19], [21]. Chaque inducteur retenu correspond à une variable mobilisable par l'algorithme, ce qui ancre la théorie dans le dispositif expérimental plutôt que de la laisser à l'état purement descriptif [13], [16].

La valeur d'une cause, dans cette optique, se mesure à son pouvoir de discrimination une fois traduite en variable d'entrée du modèle [13], [17]. 2.1. Le respect des délais (On-Time Delivery) comme indicateur de performance pivot Cette section formalise le concept de respect des délais et de promesse de livraison. Nous introduisons les indicateurs OTD et AD pour évaluer le réseau [13], [15]. 2.1.1.

Définition et mesure de la performance temporelle Nous définissons ici l'ETA (temps estimé d'arrivée) et le taux de service de livraison. Des équations mathématiques sont utilisées pour formaliser ces notions [13], [14]. 2.1.1.1. Le concept de délai estimé de livraison (Estimated Time of Arrival - ETA) et le mécanisme de la promesse client La promesse de livraison formulée par le e-commerçant représente le point d'ancrage de la confiance client et détermine la satisfaction du parcours d'achat numérique [5], [15]. Cette promesse temporelle est matérialisée par le délai estimé de livraison ( Estimated Time of Arrival - ETA), qui correspond à l'horizon temporel auquel le transporteur s'engage à achever la remise physique de la marchandise [6], [13].

D'un point de vue managérial, l'ETA n'est pas une simple estimation statistique, mais un mécanisme contractuel implicite dont le non-respect déclenche immédiatement une dégradation de l'image de marque et une attrition de la clientèle [13], [15]. La prédiction précise de cet indicateur requiert l'analyse conjointe de multiples variables stochastiques (conditions de circulation, préparation de commande, disponibilité des transporteurs) afin de substituer aux fenêtres de livraison génériques des estimations dynamiques personnalisées [7], [13]. 2.1.1.2. Formalisation mathématique du taux de service d'un réseau logistique et indicateurs clés de performance (KPI) La performance temporelle d'un réseau de distribution s'évalue principalement à travers des indicateurs clés de performance (KPI) formalisés de manière rigoureuse [13], [14]. Le taux de service de livraison à l'heure, ou On-Time Delivery (OTD), constitue la métrique pivot et se définit mathématiquement comme le rapport entre le nombre de livraisons effectuées dans les délais prescrits et le volume total d'expéditions réalisées [7], [13] : OTD = (Nombre de commandes livrées dans les délais / Nombre total de commandes) × 100 où : ATAᵢ (Actual Time of Arrival) représente la date ou l'heure réelle d'arrivée de la livraison i ; ETAᵢ (Estimated Time of Arrival) correspond à la date ou l'heure de livraison promise au client pour la commande i ; N désigne le nombre total de commandes considérées dans l'échantillon ; La fonction indicatrice I(ATAᵢ ≤ ETAᵢ) prend la valeur 1 lorsque la commande i est livrée dans les délais prévus et la valeur 0 dans le cas contraire.

Pour caractériser la sévérité des retards constatés, les planificateurs associent à l'OTD le délai moyen de retard (AD - Average Delay ), calculé uniquement sur le sous-ensemble des livraisons en retard [11], [14] : AD = (∑_(i=1)^N max(0, ATA_i - ETA_i)) / (∑_(i=1)^N I(ATA_i > ETA_i)) Ces indicateurs macroscopiques guident la prise de décision opérationnelle et permettent de détecter les défaillances structurelles des flottes de transport [10], [13]. Pour résumer, les KPI temporels comme l'OTD permettent de surveiller la qualité opérationnelle de la flotte de transport [10], [13]. 2.1.2. Typologie et nature des anomalies de livraison Nous étudions la différence entre les retards systématiques (structurels) et imprévisibles (conjoncturels). La gravité de ces retards détermine la réaction du client [14], [15].

2.1.2.1. Retards structurels (systématiques et récurrents) vs retards conjoncturels (aléatoires et imprévisibles) L'analyse scientifique des défaillances de livraison impose d'opérer une distinction fine entre les anomalies de nature structurelle et celles de nature conjoncturelle [7], [14]. Les retards structurels, systématiques et récurrents, découlent d'inefficacités intrinsèques au réseau logistique, telles qu'une mauvaise sectorisation des zones de livraison, une sous-capacité chronique des centres de tri régionaux ou une inadaptation chronique des flottes des prestataires 3PL [2], [10]. À l'inverse, les retards conjoncturels résultent d'aléas exogènes et temporaires, caractérisés par une forte incertitude statistique, à l'instar d'accidents routiers majeurs, d'intempéries météorologiques exceptionnelles ou de pics soudains et imprévisibles de commande [7], [14]. La modélisation de ces événements de longue traîne stochastique s'appuie souvent sur la Théorie des Valeurs Extrêmes (EVT), qui permet d'ajuster une loi de probabilité généralisée (GEV) sur les retards critiques afin d'évaluer le niveau de retour et la probabilité d'occurrence d'anomalies majeures [14].

2.1.2.2. Segmentation de la gravité des anomalies (retards mineurs acceptables vs retards critiques générateurs de litiges) Toutes les anomalies temporelles ne partagent pas le même impact sur le comportement de rachat des e-consommateurs, imposant une segmentation de la gravité des retards [4], [15]. La littérature identifie un seuil de tolérance psychologique au-delà duquel un retard mineur (souvent accepté sans réclamation s'il est notifié de manière transparente) se transforme en retard critique, catalyseur de réclamations officielles [6], [15]. Une déviation critique par rapport à l'ETA convenu détruit le capital confiance et pousse le client vers des comportements de rejet (refus du colis à la livraison, demandes de remboursement) [13], [15]. Ces frictions opérationnelles provoquent une saturation des centres de support client et, si elles ne sont pas résolues, déclenchent une escalade relationnelle menant à la diffusion d'avis négatifs en ligne ou à la perte définitive du client au profit de plateformes concurrentes [5], [15].

En synthèse, un dépassement critique du seuil de tolérance client provoque une hausse des litiges et sature le service après-vente [5], [15]. En conclusion, le respect des délais est un indicateur de performance pivot qui sépare les retards acceptables des anomalies critiques [13], [15]. Synthèse conceptuelle de l'évaluation de la performance temporelle Le tableau ci-dessous synthétise les indicateurs logistiques, la typologie des retards et leurs répercussions sur la relation client : Dimension d'évaluation Indicateurs / Variables clés Typologie opérationnelle Conséquences sur la relation client Références Performance contractuelle Écart temporel (ATA − ETA), taux de service OTD. Respect strict vs déviation modérée/critique de la promesse. Déterminant majeur de la satisfaction et du taux de fidélisation.

[13], [15] Défaillances structurelles Inducteurs de réseau (ex. goulots de tri, flottes 3PL inadaptées). Retards systématiques, récurrents et prévisibles. Détérioration lente de l'image de marque et attrition passive. [2], [10], [14] Aléas conjoncturels Événements de longue traîne (intempéries, accidents routiers).

Retards sporadiques, modélisés par la théorie des valeurs extrêmes (GEV). Frustration ponctuelle, tolérée si l'information est partagée en temps réel. [7], [14] Gravité comportementale Dépassement du seuil de tolérance (ex. retard > 24 heures). Retards critiques générateurs de litiges et demandes de remboursement.

Saturation du SAV, réclamations formelles et churn silencieux. [5], [13], [15] Schéma conceptuel : Modélisation systémique des impacts du retard de livraison Ce schéma DOT (Graphviz) illustre la dynamique de transition entre l'apparition des anomalies temporelles et les conséquences sur le comportement client : 2.2. Étiologie du retard : Cartographie des facteurs d'influence (Variables de l'étude) Cette section propose une classification des causes des retards de livraison. Nous analysons l'influence de l'environnement urbain, de l'organisation interne et de la saisonnalité [16], [17]. 2.2.1.

L'impact des variables géospatiales et territoriales Nous analysons ici l'impact de la distance, de la densité urbaine et de la géographie des villes sur les délais. La congestion locale perturbe fortement les plannings [16], [17]. 2.2.1.1. Rôle des infrastructures de la ville de départ et performance des centres de tri d'origine La performance de la livraison e-commerce dépend en grande partie de la localisation et de la qualité des infrastructures de départ [2], [17]. Le centre de tri d'origine sert de point de départ pour la préparation et l'expédition des colis.

Si ce centre rencontre des problèmes techniques, manque de personnel ou gère mal les flux de camions, cela crée un retard dès le début de la chaîne d'approvisionnement [2], [17]. Les grandes villes de départ disposent souvent d'infrastructures routières saturées, ce qui ralentit la sortie des marchandises vers les régions de destination [7], [17]. Ainsi, les retards au niveau des centres de tri d'origine se répercutent directement sur la suite du transport, rendant les prévisions de livraison plus incertaines dès la première étape de la chaîne logistique [13], [17]. 2.2.1.2. Influence de la topographie, de l'éloignement et des spécificités d'accessibilité de la ville de destination Les caractéristiques géographiques de la ville de destination jouent également un rôle déterminant dans le respect des délais promis aux clients [16], [17].

Plus la distance physique entre le centre de tri régional et le client final est grande, plus le risque d'incident pendant le transport augmente [2], [17]. De plus, les zones urbaines denses posent des difficultés d'accès majeures pour les véhicules de livraison, notamment en raison du trafic routier complexe, du manque de places de stationnement et des rues étroites [16], [17]. Ces barrières géographiques et routières forcent les planificateurs à diviser les zones de livraison en secteurs spécifiques afin de limiter les retards provoqués par les embouteillages urbains [7], [16]. En résumé, l'optimisation des tournées doit prendre en compte les contraintes géographiques locales pour rester réaliste [8], [17]. 2.2.2.

L'impact des variables organisationnelles, temporelles et techniques Nous étudions l'influence de la préparation des colis, du choix du transporteur 3PL et des pics de saisonnalité. Ces facteurs internes et temporels modifient les temps de transit [16], [18]. 2.2.2.1. Corrélation entre la fiabilité intrinsèque des prestataires de transport choisis et la ponctualité constatée Le choix du transporteur par le e-commerçant influe directement sur la ponctualité de la livraison finale [13], [16]. Chaque prestataire possède des méthodes de gestion de flotte, des technologies de suivi de colis et des niveaux de formation des chauffeurs différents [10], [16].

Ces différences opérationnelles expliquent pourquoi certains transporteurs affichent des taux de retard plus élevés que d'autres sur des trajets pourtant identiques [13], [16]. Une coordination étroite et un partage fluide des données entre le e-commerçant et le prestataire permettent d'anticiper les anomalies et d'améliorer globalement la qualité du service client [10], [13]. 2.2.2.2. Influence de la nature et des attributs de la marchandise (poids, volume, catégorie de produit) Les caractéristiques physiques du colis, comme son poids, ses dimensions et sa catégorie, compliquent la planification des tournées de livraison [2], [16]. Les colis volumineux ou lourds nécessitent des véhicules de transport adaptés et ralentissent les opérations de chargement et de déchargement [16], [17].

De plus, certains produits fragiles ou de valeur imposent des précautions particulières, ce qui augmente le temps de manutention à chaque point d'arrêt [2], [16]. Ces contraintes techniques réduisent la vitesse de distribution et augmentent le risque de ne pas respecter la promesse horaire de livraison faite au client final [6], [16]. 2.2.2.3. Facteurs temporels : impact des cycles hebdomadaires et des pics de saisonnalité (Black Friday, fêtes de fin d'année) La demande en e-commerce varie de manière importante selon des cycles temporels réguliers et des périodes de pics d'activité dans l'année [1], [18]. Durant les jours de forte activité de la semaine ou lors d'événements commerciaux comme le Black Friday et les fêtes de fin d'année, les volumes de commandes augmentent brutalement [1], [18].

Cette surcharge temporaire engorge les centres de tri et sature les capacités de transport disponibles sur le marché [18]. Sans outils de prévision adaptés permettant de planifier les ressources à l'avance, cette saisonnalité marquée conduit inévitablement à une hausse des retards et à une dégradation de la satisfaction client [13], [18]. Au final, les pics saisonniers (comme le Black Friday) exigent une flexibilité accrue des ressources de transport pour éviter les retards en cascade [3], [18]. Finalement, cette taxonomie montre que les retards résultent d'une combinaison complexe de facteurs externes et internes [17], [18]. Synthèse des variables influençant les délais de livraison Le tableau ci-dessous regroupe les différentes variables analysées, leur type et leur impact sur la ponctualité des livraisons : Catégorie de variable Variables spécifiques Mode d'impact sur la livraison Niveau d'influence Références Géospatiale Infrastructures de départ, centres de tri.

Temps d'attente initial et préparation des colis. Modéré à élevé [2], [17] Territoriale Distance, accessibilité, trafic urbain. Augmentation du temps de trajet et des incidents de route. Élevé [16], [17] Organisationnelle Fiabilité du prestataire de transport (3PL). Qualité de service, gestion de flotte et formation des chauffeurs.

Élevé [13], [16] Technique Poids, volume et nature de la marchandise. Contraintes de chargement, véhicules spécifiques requis. Modéré [2], [16] Temporelle Cycles hebdomadaires, pics de fin d'année (Black Friday). Saturation des entrepôts et des capacités de transport. Élevé [1], [18] Schéma conceptuel : Relation entre les variables d'étude et les délais de livraison Ce schéma DOT (Graphviz) montre comment les variables géospatiales, organisationnelles et temporelles influencent la performance de livraison : 2.3.

Les conséquences économiques et stratégiques de la non-performance temporelle Nous mesurons l'impact financier et stratégique des retards de livraison sur l'e-commerçant. Le but est de chiffrer les pertes directes et indirectes [19], [21]. 2.3.1. Les coûts opérationnels directs pour le e-commerçant Nous détaillons les coûts liés à la logistique des retours, aux colis refusés et aux pénalités contractuelles. Ces coûts pèsent directement sur la rentabilité [19], [20].

2.3.1.1. Saturation des centres d'appels, surcharge du service après-vente (SAV) et coût de traitement des réclamations Les retards de livraison provoquent une augmentation immédiate des appels et des messages vers le service après-vente (SAV) [15], [19]. Lorsque les clients ne reçoivent pas leur colis à la date prévue, ils contactent les centres de support pour obtenir des informations et exprimer leur mécontentement [15], [19]. Cette surcharge de travail oblige les e-commerçants à embaucher plus d'agents de support ou à surcharger leurs équipes existantes, ce qui augmente fortement les coûts de gestion des réclamations [5], [19]. De plus, le manque d'outils de suivi transparents aggrave cette situation, car le SAV passe beaucoup de temps à essayer de localiser le colis auprès des transporteurs partenaires [10], [15].

2.3.1.2. Coûts financiers de la logistique des retours (Reverse Logistics), des réexpéditions et des pénalités commerciales La mauvaise gestion des délais de livraison entraîne également des coûts logistiques importants liés aux retours de marchandises, appelés logistique inverse ou reverse logistics [20], [21]. En effet, les achats en ligne sont retournés en moyenne trois fois plus souvent que les achats effectués dans les magasins physiques [21]. Si un colis arrive trop tard, le client peut refuser la livraison ou demander un remboursement, ce qui oblige le vendeur à payer pour le retour du produit vers l'entrepôt [20], [21]. Ces retours, combinés aux frais de réexpédition des colis perdus et aux pénalités commerciales dues aux retards, pèsent lourdement sur les dépenses et réduisent la rentabilité des entreprises de commerce en ligne [6], [21].

En résumé, les échecs de livraison augmentent les coûts de reverse logistics et réduisent la marge nette des opérations [19], [20]. 2.3.2. La destruction du capital client et le risque réputationnel Nous analysons le départ silencieux des clients (churn) et les avis négatifs sur internet. Ces impacts dématérialisés affectent l'image de marque [15], [21]. 2.3.2.1.

Impact sur la satisfaction globale et mécanisme d'attrition des clients déçus (customer churn) Au-delà des coûts financiers immédiats, les retards de livraison nuisent gravement à la satisfaction des clients et provoquent leur départ vers la concurrence, un phénomène appelé attrition ou customer churn [19], [21]. Acquérir un nouveau client coûte beaucoup plus cher que de fidéliser un client existant, ce qui rend la perte de clientèle particulièrement dommageable [19]. Les études montrent que près de 58 % des consommateurs arrêtent d'utiliser une plateforme de vente après seulement quelques mauvaises expériences d'achat, souvent liées aux livraisons [19]. Les retards à répétition détruisent la relation de confiance et incitent les acheteurs déçus à quitter définitivement le site sans même avertir le vendeur [5], [19]. 2.3.2.2.

Amplification numérique des expériences négatives (avis en ligne, réseaux sociaux) et dégradation à long terme de l'image de marque À l'ère d'Internet, la frustration d'un client déçu par une livraison se diffuse rapidement sur les réseaux sociaux et les sites d'avis en ligne [5], [15]. Les consommateurs écrivent de nombreux commentaires négatifs pour décrire leurs problèmes logistiques et le manque de réactivité du support client [15], [19]. Ces avis publics découragent les nouveaux acheteurs potentiels et nuisent à la réputation du e-commerçant à long terme [5], [15]. Cette dégradation de l'image de marque est difficile à corriger et peut impacter les ventes futures de l'entreprise sur plusieurs années, montrant que la qualité de la livraison est un élément clé de la réputation commerciale [15], [21]. Pour finir, la dégradation de la confiance pousse les consommateurs vers les concurrents, ce qui augmente le coût d'acquisition client [15], [21].

En conclusion, les retards dégradent la rentabilité financière et la réputation en ligne, ce qui nuit à la compétitivité à long terme [19], [21]. Synthèse des impacts de la non-performance temporelle Le tableau ci-dessous résume les conséquences opérationnelles et réputationnelles des retards de livraison pour les entreprises de e-commerce : Type de conséquence Impact spécifique Effet sur l'entreprise Gravité pour le business Références Opérationnelle Surcharge du service après-vente (SAV). Augmentation des coûts de traitement des appels et messages. Modérée à élevée [15], [19] Logistique Gestion des retours ( reverse logistics ). Frais de transport retour, réexpéditions et gestion des stocks.

Élevée [20], [21] Financière Pénalités commerciales et remboursements. Pertes d'argent directes et baisse de la marge bénéficiaire. Élevée [6], [21] Relationnelle Attrition des clients ( customer churn ). Perte de revenus réguliers, coût élevé d'acquisition de nouveaux clients. Très élevée [19], [21] Réputationnelle Avis négatifs sur internet et réseaux sociaux.

Perte de confiance des prospects, dégradation de l'image de marque. Élevée [5], [15] Schéma conceptuel : Boucle de perte de performance liée aux retards de livraison Ce schéma DOT (Graphviz) illustre comment les retards de livraison créent une boucle négative affectant à la fois les finances et la réputation du e-commerçant : En résumé, les retards de livraison ont des conséquences économiques directes et dégradent durablement la relation de confiance avec le client [15], [19], [20]. Bilan analytique du chapitre La performance temporelle se laisse réduire à un écart mesurable, mais ses effets ne se propagent pas de façon linéaire : au-delà d'un seuil de tolérance, un retard mineur bascule en litige et déclenche une perte de capital client difficilement réversible [13], [15]. Les inducteurs cartographiés, du centre de tri d'origine à la fiabilité du prestataire et aux pics de saisonnalité, agissent moins isolément qu'en combinaison, ce qui explique l'échec des moyennes historiques à les anticiper [16], [17], [18]. La chaîne de coûts referme la démonstration : surcharge du service après-vente, logistique inverse, pénalités, puis attrition silencieuse et dégradation réputationnelle amplifiée par les canaux numériques [19], [20], [21].

Ces postes ne se cumulent pas seulement, ils s'entretiennent, car un client mécontent renchérit simultanément le coût de service et le coût d'acquisition de son remplaçant [19], [21]. La non-performance temporelle se comporte ainsi comme un coût composé, dont la base s'élargit à chaque épisode non maîtrisé [15], [21]. L'imputation de ces déviations à des facteurs identifiables, plutôt qu'au hasard, ouvre la voie à leur prédiction, à condition de disposer d'un appareillage analytique capable de croiser des variables hétérogènes [13], [24]. Les approches descriptives traditionnelles atteignant ici leur limite, le passage à un cadre algorithmique outillé devient la suite logique, et ce basculement vers la Logistique 4.0 constitue l'objet du chapitre suivant [22], [24].
<!-- OLD_MEMOIRE_EXPANSION_END -->

### Synthèse du chapitre

La logistique e-commerce transforme le délai de livraison en promesse visible. Le dernier kilomètre concentre les facteurs de variabilité parce qu'il combine fragmentation des flux, contraintes territoriales, disponibilité du client, dépendance au transporteur et information incomplète. Cette configuration justifie l'usage du Machine Learning, mais elle impose une prudence méthodologique : toutes les cibles ne sont pas également prédictibles.

Le chapitre suivant déplace l'analyse vers le cadre algorithmique. Il ne s'agit pas d'introduire le Machine Learning comme une solution automatique, mais comme une méthode expérimentale permettant de tester ce que les données savent réellement expliquer. Cette posture prépare la décision principale du projet : abandonner la classification directe du retard lorsque les métriques montrent son instabilité, puis reformuler le problème en prédiction de durée et en recalibrage de l'ETA.

## Chapitre 2 - Machine Learning appliqué à la prédiction logistique

### Introduction

La prédiction logistique ne devient pertinente que lorsque la donnée cesse d'être un simple enregistrement administratif. Une date d'expédition, une ville d'origine, un transporteur ou une promesse ETA ne sont pas seulement des colonnes stockées dans un fichier ; ce sont les traces compactes de décisions déjà prises dans le réseau. Le Machine Learning exploite ces traces pour apprendre une relation entre un état connu avant ou pendant l'expédition et une grandeur observée après la livraison. Cette relation n'est pas une causalité pure, mais un mécanisme statistique de généralisation.

La Logistique 4.0 repose précisément sur cette transformation. Les systèmes de transport, les plateformes e-commerce, les API de suivi, les capteurs et les historiques d'expédition produisent des flux numériques qui rendent le pilotage moins réactif et plus anticipatif. Le passage de la donnée brute à l'aide à la décision suppose cependant une discipline expérimentale : choisir une cible valide, isoler les informations disponibles au moment de la prédiction, construire un jeu de test crédible et interpréter les métriques sans les détacher du problème métier.

### 2.1 Logistique 4.0 et exploitation des traces numériques

La figure 5 représente l'architecture méthodologique du pipeline prédictif. Elle relie la donnée brute, la préparation anti-fuite, l'encodage, la comparaison des modèles, la validation temporelle et le déploiement Streamlit. Cette représentation synthétique montre que le Machine Learning n'est pas une cellule isolée du projet, mais une chaîne complète qui commence par la qualité de la donnée et se termine par une décision opérationnelle.

**Figure à insérer :** `report_assets/figures/22_theorie_pipeline_ml.png`

La Logistique 4.0 peut être comprise comme l'application des principes de connectivité, d'automatisation et d'analyse prédictive aux flux physiques. Elle ne remplace pas le transport par l'information ; elle synchronise les deux. Un colis continue de circuler dans un réseau matériel, mais chaque événement de son parcours peut produire une trace : création de commande, prise en charge, tri, départ hub, arrivée hub, tentative de livraison, réception. Lorsque ces traces sont reliées à des variables contextuelles, elles deviennent un matériau exploitable pour l'apprentissage.

Cette exploitation reste incomplète lorsque les données disponibles ne couvrent qu'une partie de la chaîne. Dans notre cas, les variables historiques décrivent la commande et la promesse initiale, mais elles ne décrivent pas toute l'exécution réelle. Le transporteur est présent, les villes sont présentes, le coût et la quantité sont présents, mais les scans intermédiaires, les incidents, la position GPS ou les contraintes de tournée ne le sont pas. Cette limite ne rend pas le modèle inutile ; elle définit ce qu'il peut raisonnablement apprendre.

L'intérêt d'une approche supervisée apparaît lorsque la variable à prédire est observée dans l'historique. Pour chaque expédition ancienne, la durée réelle peut être calculée à partir des dates, puis rapprochée des informations connues au moment de l'expédition. Le modèle apprend alors à associer des profils d'expédition à une durée probable. Le point méthodologique décisif consiste à ne jamais utiliser, dans les variables d'entrée, une information qui n'aurait pas été disponible avant la livraison [5].

### 2.2 Apprentissage supervisé : classification et régression

L'apprentissage supervisé consiste à estimer une fonction entre un ensemble de variables explicatives et une cible observée. Lorsque la cible est une classe, le problème relève de la classification ; lorsque la cible est numérique et continue, il relève de la régression. Dans un contexte de livraison, ces deux formulations semblent proches, car un retard peut être déduit d'une durée. Elles ne portent pourtant pas le même signal.

La classification du retard transforme la livraison en événement binaire : le colis dépasse ou ne dépasse pas l'ETA. Cette formulation paraît naturellement alignée avec la décision opérationnelle, mais elle perd une partie de l'information. Deux colis livrés avec un jour et cinq jours de dépassement sont classés dans la même catégorie, tandis qu'un colis livré exactement au seuil peut basculer d'une classe à l'autre pour une variation minime. Le seuil rend le problème plus abrupt.

La régression de la durée réelle conserve au contraire la granularité temporelle. Elle permet au modèle d'apprendre que certaines expéditions se situent autour de quatre jours, d'autres autour de dix jours, et d'autres encore autour de quinze jours. La décision de risque peut ensuite être dérivée en comparant la durée prédite à la durée planifiée. Cette formulation est plus stable lorsque le retard pur dépend d'aléas non observés, car elle exploite la structure continue du transit avant de produire une lecture opérationnelle.

Le choix entre classification et régression ne doit donc pas être arrêté par préférence théorique. Il doit être testé. Une classification qui obtient une AUC proche du hasard indique que les variables disponibles ne discriminent pas correctement les colis retardés. Une régression qui dépasse un baseline métier montre, au contraire, que la durée contient encore un signal exploitable. Cette distinction gouverne l'architecture méthodologique du Chapitre 3.

### 2.3 Prévention des fuites de données

La fuite de données apparaît lorsqu'une information utilisée pendant l'entraînement ne serait pas disponible au moment réel de la prédiction. Elle produit des performances artificiellement élevées, car le modèle apprend une partie de la réponse au lieu d'apprendre une relation généralisable. Dans un projet logistique, la fuite la plus évidente consiste à utiliser la date de livraison réelle pour prédire la durée de livraison. Elle peut aussi apparaître plus subtilement à travers un statut final, une variable postérieure ou un encodage calculé avant la séparation train-test [5].

Le pipeline doit donc respecter une chronologie stricte. Les dates brutes servent à construire la cible et les variables disponibles avant l'exécution, puis les informations postérieures sont exclues des entrées. Le découpage train-test précède l'ajustement des transformations, afin que le test reste réellement non vu. Les imputations, les standardisations et les encodages doivent être appris sur l'entraînement, puis appliqués au test. Cette discipline est moins spectaculaire que le choix du modèle, mais elle conditionne la crédibilité de toutes les métriques.

Le target encoding des villes illustre cette exigence. Une ville d'origine ou de destination peut posséder une cardinalité plus élevée qu'un transporteur ou un fournisseur. L'encodage par la cible transforme cette modalité en valeur numérique liée à la moyenne observée de la cible, mais il peut surapprendre si la ligne utilise sa propre cible dans le calcul. L'usage d'un encodage régularisé ou croisé permet de réduire cette fuite, ce qui rend la transformation plus compatible avec un déploiement réel [6].

### 2.4 Validation, métriques et baselines

Une métrique isolée ne suffit pas à juger un modèle. Une MAE faible peut paraître satisfaisante jusqu'à ce qu'on la compare à une règle simple. Dans ce projet, le baseline métier le plus important consiste à recopier la durée planifiée. Si le modèle ne bat pas cette règle, il n'ajoute pas de valeur opérationnelle. Si le modèle la dépasse, le gain doit être quantifié et interprété sans excès.

La validation croisée permet de réduire la dépendance à une partition unique. Elle entraîne et évalue le modèle sur plusieurs plis, ce qui donne une estimation plus stable de sa capacité de généralisation [8]. Cette méthode reste néanmoins une simulation statistique. Pour un problème logistique, un split temporel complète utilement l'évaluation, car l'exploitation réelle apprend sur le passé et prédit le futur. Un modèle qui conserve une performance proche entre split aléatoire et split temporel inspire davantage confiance.

Les métriques de régression répondent à des lectures différentes. La MAE exprime l'erreur moyenne en jours, donc elle parle directement au métier. La RMSE pénalise davantage les fortes erreurs, ce qui signale les situations où le modèle rate plus lourdement. Le R² mesure la part de variance expliquée par rapport à une prédiction moyenne. Aucune de ces métriques ne suffit seule ; leur combinaison permet de lire à la fois l'utilité opérationnelle et la qualité statistique.

Les métriques de classification doivent être interprétées avec encore plus de prudence. Lorsque le taux de base du retard est élevé, une règle qui alerte presque tout peut obtenir un rappel très élevé et un F1 apparemment honorable, tout en ne discriminant presque rien. L'AUC-ROC devient alors essentielle, car elle mesure la capacité du score à classer les vrais dépassements avant les non-dépassements. Une AUC proche de 0,5 indique que la règle se comporte presque comme un classement aléatoire.

### 2.5 Modèles tabulaires et choix algorithmique

Le tableau 2.1 représente la logique de sélection des modèles candidats. Il explicite pourquoi le modèle final ne doit pas être choisi uniquement sur la complexité algorithmique, mais sur la combinaison entre performance, robustesse, interprétabilité et capacité à être déployé sans divergence entre notebook et application.

| Famille de modèle | Rôle dans l'expérience | Force principale | Limite à surveiller |
|---|---|---|---|
| Baseline moyenne | Plancher statistique | Mesure la difficulté minimale | Ne tient pas compte de l'ETA |
| Baseline ETA | Référence métier | Compare le modèle à la pratique existante | Reproduit les biais de la promesse actuelle |
| Ridge | Modèle simple régularisé | Lisible et stable | Faible capture des interactions |
| Random Forest | Ensemble non linéaire | Capture des ruptures locales | Plus lourd à interpréter et déployer |
| HistGradientBoosting | Modèle final candidat | Bon compromis performance-déploiement | Nécessite une validation stricte |

Les données de livraison utilisées dans ce mémoire sont tabulaires. Elles combinent des variables numériques, des variables catégorielles, des dates transformées et des variables externes éventuelles. Ce type de données se prête bien à plusieurs familles de modèles : modèles linéaires régularisés, forêts aléatoires et méthodes de gradient boosting. Le choix ne doit pas être dicté par la sophistication apparente, mais par la performance, la stabilité et l'interprétabilité.

Le modèle Ridge constitue un repère linéaire robuste. Il estime une relation additive en limitant l'amplitude des coefficients par régularisation. Il est utile parce qu'il révèle si la structure du problème est presque linéaire autour de quelques variables fortes. La Random Forest introduit une logique d'arbres multiples, capable de capter des interactions non linéaires sans imposer une forme fonctionnelle simple [10]. Le HistGradientBoostingRegressor, enfin, construit séquentiellement des arbres correctifs et se montre efficace sur les grands jeux tabulaires [9].

Dans une démarche expérimentale rigoureuse, le modèle final n'est pas choisi parce qu'il paraît plus avancé. Il est choisi parce qu'il améliore légèrement les résultats, reste stable en validation et s'intègre dans un pipeline déployable. Si un modèle plus simple obtient une performance presque identique, cette proximité doit être mentionnée. Elle peut signaler que la donnée contient un signal dominant, concentré dans une variable comme la durée planifiée, plutôt qu'un ensemble riche d'interactions faibles.

L'optimisation des hyperparamètres complète cette comparaison. Une grille exhaustive devient vite coûteuse lorsque plusieurs paramètres interagissent. La recherche aléatoire explore un espace plus large à budget limité et se justifie lorsque toutes les dimensions n'ont pas la même importance [11]. Dans ce mémoire, l'optimisation est effectuée sur un sous-échantillon pour maîtriser le temps de calcul, puis la meilleure configuration est réajustée sur l'ensemble d'entraînement.

### 2.6 Interprétabilité et portée décisionnelle

L'interprétabilité répond à une exigence scientifique et opérationnelle. Un modèle qui fournit une prédiction sans expliquer les variables qui portent le signal peut difficilement être discuté avec un décideur logistique. L'importance par permutation mesure la baisse de performance lorsque chaque variable est brouillée. Elle ne dit pas seulement qu'une variable est corrélée à la cible ; elle mesure la perte réelle de pouvoir prédictif lorsque cette variable devient inutilisable.

Les valeurs SHAP proposent une autre lecture, plus locale, en attribuant à chaque variable une contribution dans la prédiction. Elles sont utiles lorsque les variables interagissent et lorsque le décideur veut comprendre pourquoi une expédition particulière reçoit une durée prédite plus élevée [12]. Dans le cas présent, l'interprétabilité sert surtout à vérifier une hypothèse : si la durée planifiée domine toutes les autres variables, le modèle fonctionne principalement comme un recalibreur d'ETA.

Cette lecture protège le projet contre une interprétation excessive. Une application déployée peut afficher un score de risque, des axes prioritaires et des transporteurs à surveiller. Ces éléments sont utiles pour organiser l'action, mais ils ne doivent pas être présentés comme une vérité causale. Lorsque les importances montrent que les autres variables contribuent faiblement, la conclusion doit rester mesurée : le système aide à corriger une promesse temporelle, non à expliquer exhaustivement les causes individuelles du retard.





<!-- OLD_MEMOIRE_EXPANSION_START -->
### 2.7 Approfondissement théorique intégré depuis le mémoire initial

Le développement suivant prolonge le cadrage technologique et méthodologique. Il relie la Logistique 4.0, l'exploitation des traces numériques et les exigences de validation qui justifient le pipeline retenu dans la partie pratique.

Le Paradigme de la Logistique 4.0 et le Cadre Théorique du Machine Learning Ce troisième chapitre introduit le cadre technologique de la Logistique 4.0 et les algorithmes de Machine Learning. Nous analysons l'apport des capteurs IoT, de l'apprentissage supervisé, et détaillons les modèles candidats et de validation [22], [25], [30]. Propos liminaire du chapitre Le passage à la Logistique 4.0 ne se résume pas à une superposition de capteurs sur des processus existants ; il modifie la temporalité de la décision en rendant disponible, avant le départ du véhicule, une information jusque-là constatée a posteriori [22], [23]. Nous analysons ce basculement comme un déplacement du point d'intervention : la donnée captée en continu par l'Internet des objets alimente un modèle qui transforme une variable d'état en variable prédictive, et déplace ainsi la gestion du retard de l'aval vers l'amont [13], [22]. C'est cette inversion du sens de l'information, et non la seule abondance des données, qui fonde l'intérêt opérationnel de l'apprentissage supervisé [24], [25].

L'apprentissage supervisé procède par induction d'une relation entre des attributs d'expédition et un résultat observé, puis projette cette relation sur des livraisons inédites [25], [27]. Le choix entre une cible discrète et une cible continue n'est pas neutre : il détermine la fonction de perte optimisée, la métrique d'évaluation pertinente et l'action métier déclenchée en sortie [13], [26]. Notre étude de cas privilégie la formulation binaire, car la décision recherchée, surveiller ou laisser passer une expédition, épouse la structure d'une classification plutôt que celle d'une estimation chiffrée [5], [27]. L'écosystème algorithmique mobilisé s'étage des modèles linéaires interprétables aux méthodes d'ensemble par agrégation et par renforcement séquentiel, chacune capturant différemment les interactions non linéaires entre variables logistiques [28], [29], [30]. La validation de ces modèles, par partitionnement puis validation croisée, ne constitue pas un appendice technique mais la garantie que la relation apprise se généralise au-delà de l'échantillon, condition sans laquelle aucune prédiction de retard ne tiendrait en exploitation [29], [30].

Le déséquilibre marqué des classes, propre à un phénomène de retard minoritaire, impose en outre des métriques et des pondérations spécifiques que ce chapitre prépare avant la mise en œuvre [13], [30]. 3.1. L'avènement de la Logistique 4.0 et la connectivité des flux Cette section étudie les limites des ERP et TMS traditionnels face aux flux modernes. Nous décrivons comment les capteurs connectés (IoT, RFID) permettent d'obtenir des données en temps réel [22], [23]. 3.1.1.

Les limites des approches de gestion et d'analyse traditionnelles Nous analysons la rigidité des anciens systèmes d'information et l'insuffisance des moyennes historiques simples. Ces outils manquent de réactivité face aux variations de transport [12], [23]. 3.1.1.1. Rigidité des systèmes d'information classiques (ERP et TMS) et traitement de l'information a posteriori Les entreprises de commerce en ligne ont longtemps utilisé des systèmes informatiques traditionnels, comme les logiciels de gestion d'entreprise (ERP) et les systèmes de gestion du transport (TMS), pour organiser leurs livraisons [10], [12]. Cependant, ces outils classiques montrent aujourd'hui leurs limites face aux nouvelles exigences de rapidité et d'agilité de la distribution urbaine [23], [24].

Le principal défaut de ces systèmes est qu'ils traitent l'information après coup (a posteriori), c'est-à-dire une fois que la livraison est déjà effectuée ou que le retard est déjà survenu [12], [23]. De plus, ces outils fonctionnent souvent de manière isolée et manquent de flexibilité pour échanger des données en temps réel entre les e-commerçants et les transporteurs partenaires, ce qui empêche d'anticiper les problèmes de transport [10], [24]. 3.1.1.2. Incapacité des approches statistiques descriptives simples (moyennes historiques) à capturer la complexité des interactions de variables Pour prévoir les délais de livraison, les méthodes traditionnelles se limitent généralement à calculer des statistiques simples, comme la moyenne des temps de trajet passés [1], [13]. Ces calculs simples s'avèrent incapables de saisir comment les différentes variables (comme la météo, la taille du colis, le trafic routier ou le choix du transporteur) s'influencent mutuellement à un instant précis [13], [23].

Par exemple, une moyenne historique ne peut pas anticiper l'effet combiné d'un colis lourd livré un vendredi pluvieux de Black Friday [1], [18]. Cette incapacité à croiser les données rend les prévisions de livraison peu fiables et oblige les gestionnaires à subir les retards plutôt qu'à les prévoir [13], [24]. En conclusion, les méthodes statistiques simples ne parviennent pas à capturer les interactions complexes entre les variables logistiques [13], [24]. 3.1.2. La connectivité et la massification des données (Big Data Logistique) Nous présentons le rôle de l'IoT et décrivons les 4V du Big Data appliqués au transport (Volume, Variété, Vélocité, Véracité).

Ces données constituent le socle de l'analyse moderne [22], [24]. 3.1.2.1. Le rôle de l'Internet des Objets (IoT), de la RFID et des capteurs embarqués dans la captation de données en temps réel Pour moderniser la gestion des flux, la Logistique 4.0 s'appuie désormais sur l'Internet des Objets (IoT) et les capteurs intelligents comme les puces RFID et les traceurs GPS [22], [23]. Ces technologies permettent de suivre chaque colis en temps réel, depuis sa sortie de l'entrepôt jusqu'au domicile du client [2], [22]. Les capteurs enregistrent automatiquement des informations clés (position géographique, vitesse du véhicule, étapes de tri), éliminant ainsi les ruptures d'information entre les différents acteurs de la chaîne logistique [3], [23].

Grâce à ce suivi continu, les e-commerçants peuvent enfin voir ce qui se passe durant le transport final et réagir rapidement en cas d'anomalie [22], [23]. 3.1.2.2. Caractéristiques de la donnée de transport (Volume, Variété, Vélocité, Véracité) appliquée aux variables du e-commerce L'usage des capteurs et des objets connectés génère une grande quantité d'informations, caractérisée par les 4V du Big Data appliqués au transport [13], [24] : Volume : Les systèmes enregistrent des milliers de scans de colis et de positions GPS chaque seconde [13], [22]. Variété : Les données récoltées proviennent de sources diverses (adresses de livraison, caractéristiques physiques des produits, historiques des transporteurs) [22], [24]. Vélocité : Les informations sont transmises et actualisées de façon instantanée [13], [24].

Véracité : Les traceurs GPS et les scans automatiques garantissent la précision et la fiabilité des coordonnées géographiques récoltées [22], [24]. Ces données massives et variées servent de matière première pour alimenter les modèles de Machine Learning, qui vont analyser ces flux complexes pour prédire précisément les risques de retard avant que le véhicule ne prenne la route [13], [24]. Finalement, la collecte automatisée en temps réel élimine les zones d'ombre durant le transport du dernier kilomètre [2], [22]. En résumé, l'Internet des Objets fournit les données massives (Big Data) nécessaires pour alimenter les modèles de prédiction logistique [22], [24]. Synthèse de la transition vers la Logistique 4.0 Le tableau ci-dessous compare l'organisation logistique traditionnelle et le nouveau modèle de la Logistique 4.0 : Dimension d'analyse Logistique traditionnelle Logistique 4.0 Avantages de la transition Références Outils de gestion Logiciels classiques (ERP, TMS rigides).

Plateformes connectées, capteurs IoT et RFID. Suivi en temps réel et partage fluide des données. [10], [22], [23] Type d'analyse Statistiques simples (moyennes historiques). Modèles de Machine Learning et Big Data. Détection des interactions complexes de variables.

[13], [23] Prise de décision Réactive (après constatation du retard). Proactive (prédiction du risque avant départ). Réduction des coûts et meilleure satisfaction client. [12], [24] Visibilité des flux Faible (phénomène de boîte noire post-entrepôt). Totale (suivi GPS et scans automatisés en continu).

Suppression des zones d'ombre durant le transport. [2], [22] Schéma conceptuel : De la logistique réactive à la logistique proactive 4.0 Ce schéma DOT (Graphviz) illustre le passage d'une gestion traditionnelle des retours et retards à une planification intelligente basée sur le Big Data et l'IoT : 3.2. L'Intelligence Artificielle comme moteur de la "Smart Supply Chain" Cette section présente les principes du Machine Learning supervisé appliqué à la logistique. Nous comparons les approches de classification discrète et de régression continue [25], [26]. 3.2.1.

Du Big Data à l'aide à la décision : Le rôle du Machine Learning Nous abordons les fondements de l'apprentissage supervisé à partir des données historiques de livraison. Cette transition permet d'anticiper le risque de retard avant expédition [25], [27]. 3.2.1.1. Définition et principes fondamentaux du paradigme de l'apprentissage supervisé L'apprentissage supervisé est une méthode de Machine Learning où la machine apprend à l'aide d'exemples historiques déjà connus et étiquetés [25], [27]. Dans le domaine de la logistique, ces exemples sont des données de livraisons passées (comme la distance, le transporteur choisi ou le jour d'expédition) associées au résultat réel (comme le temps de trajet final ou le fait que le colis soit arrivé en retard) [25], [26].

L'algorithme analyse ces exemples historiques pour repérer des règles et des relations logiques entre les variables d'entrée et le résultat final [13], [27]. Une fois entraîné, ce modèle devient capable de prédire le résultat pour de nouvelles livraisons dont on ne connaît pas encore l'issue, aidant ainsi les entreprises à prendre des décisions basées sur les données [25], [27]. 3.2.1.2. La transition managériale : passer d'une gestion réactive (constater le retard) à une anticipation proactive (prédire le risque avant expédition) L'intégration du Machine Learning permet aux gestionnaires de transport de changer radicalement de méthode de travail [12], [25]. Traditionnellement, la gestion était réactive : l'entreprise constatait le retard une fois le délai dépassé et devait gérer la frustration du client et la surcharge du service après-vente [5], [15].

Avec l'apprentissage supervisé, la gestion devient proactive : le modèle estime le risque de retard avant même que le camion ne quitte l'entrepôt [25], [27]. Cette anticipation permet d'ajuster les tournées en temps réel, de changer de transporteur si le risque est trop élevé ou d'avertir le client à l'avance pour maintenir sa confiance [6], [25]. Au final, l'analyse prédictive transforme les processus décisionnels en permettant d'ajuster les tournées en temps réel [6], [25]. 3.2.2. L'arbitrage méthodologique : Classification Discrète vs Régression Continue Nous formalisons mathématiquement la classification binaire ( Y ∈ {0, 1} ) et la régression continue (Y ∈ ℝ⁺).

Nous évaluons l'intérêt de chaque approche pour le gestionnaire [13], [26]. 3.2.2.1. L'approche par Classification : Formulation mathématique pour la prédiction d'un statut binaire Y ∈ {0, 1} (En retard / À l'heure) ou multiclasse L'approche par classification consiste à prédire une catégorie ou un statut à partir des données de livraison [13], [27]. Dans sa forme la plus simple (binaire), la variable à prédire $Y$ prend deux valeurs possibles [13], [27] : Y ∈ {0, 1}, où Y est la variable cible telle que : - Y = 1 si la commande est livrée en retard ; - Y = 0 si la commande est livrée à l'heure. Le modèle de classification cherche à estimer la probabilité que le colis soit en retard en fonction des caractéristiques de la livraison X (comme le poids du colis, la ville de destination ou le jour de la semaine) [5], [27] : P(Y = 1 | X) = f(X) Cette approche est particulièrement utile pour identifier rapidement les colis à risque et déclencher des alertes automatiques [13], [25].

3.2.2.2. L'approche par Régression: Formulation mathématique pour l'estimation d'une variable continue Y ∈ ℝ⁺ ( temps de transit ou durée exacte du retard) L'approche par régression cherche à prédire une valeur numérique précise, comme la durée exacte du trajet ou le nombre de minutes de retard [26]. Dans ce cas, la variable à prédire $Y$ est continue [26] : Y ∈ ℝ⁺ Le modèle de régression cherche à estimer cette valeur en fonction des caractéristiques de livraison X [26] : Y = f(X) + ε où f(X) représente le temps estimé par le modèle et ε désigne l'erreur de prédiction, c'est-à-dire l'écart entre le temps de livraison réel et l'estimation théorique [26]. Cette méthode permet d'obtenir des données de livraison chiffrées très précises pour planifier les tournées [6], [26]. 3.2.2.3.

Analyse comparative de la valeur décisionnelle et de l'utilité métier de chaque approche pour le gestionnaire transport Le choix entre la classification et la régression dépend des objectifs pratiques du gestionnaire logistique [5], [26]. La classification est simple à mettre en place et très efficace pour le service client, car elle permet d'envoyer des alertes claires (par exemple, une notification verte pour "à l'heure" et rouge pour "à risque de retard") [15], [25]. En revanche, la régression offre une précision plus grande, essentielle pour le service logistique qui doit organiser les plannings des chauffeurs et optimiser l'enchaînement des arrêts [26]. Cependant, la régression est plus sensible aux variations de trafic imprévues, ce qui peut rendre ses estimations chiffrées moins stables que la simple détection de statut de la classification [5], [27]. Pour résumer, la classification est idéale pour le service client tandis que la régression est indispensable pour optimiser les plannings [26], [27].

En synthèse, l'arbitrage entre classification et régression dépend des objectifs métiers, entre envoi d'alertes clients et planification des tournées [5], [26]. Synthèse comparative des approches de Classification et de Régression Le tableau ci-dessous résume les différences, formules et applications métiers de chaque méthode de prédiction : Approche de modélisation Nature de la variable prédite (Y) Formulation mathématique générale Utilité pratique (Métier) Exemples d'algorithmes Références Classification Discrète Binaire ou multiclasse (ex. À l'heure, En retard, Perdu). $P(Y = 1 \vert X) = f(X)$ Déclenchement d'alertes automatiques et gestion SAV. Régression logistique, Arbre de décision, LightGBM.

[5], [13], [27] Régression Continue Valeur numérique réelle (ex. Temps de transit en minutes). $Y = f(X) + \epsilon$ Planification des tournées et calcul précis des heures d'arrivée (ETA). Régression linéaire, Forêt aléatoire, Gradient Boosting. [5], [26] Schéma conceptuel : Choix et impact des modèles de prédiction logistique Ce schéma DOT (Graphviz) illustre comment les caractéristiques d'une commande sont traitées par les deux approches de modélisation et leurs actions associées : 3.3.

Modèles candidats et cadre d'évaluation pour la prédiction logistique Cette dernière section évalue les algorithmes de Machine Learning candidats et leur cadre de validation. Nous décrivons les modèles de régression, d'arbres, de forêts aléatoires et de boosting, ainsi que les métriques associées [28], [29]. 3.3.1. Les modèles prédictifs candidats Nous analysons ici les modèles linéaires, les arbres CART (Gini et entropie), la forêt aléatoire (bagging) et les méthodes de boosting de gradient (XGBoost et LightGBM) [28], [30]. 3.3.1.1.

Modèles linéaires (Régression linéaire et logistique) et limites face aux relations non linéaires Les modèles de régression linéaire et logistique cherchent à établir des relations simples et directes entre les caractéristiques d'une livraison et le résultat attendu (comme la durée du transport ou le risque de retard) [28], [29]. Dans la logistique moderne du commerce en ligne, les données de transport dépendent cependant de nombreux facteurs imprévisibles, comme les embouteillages urbains, les variations de la météo ou le comportement du livreur [13], [28]. Les modèles linéaires supposent que l'effet de chaque variable est constant et cumulatif, ce qui est incapable de représenter ces relations complexes et changeantes [29], [30]. En conséquence, face à des données logistiques réelles qui évoluent de façon imprévisible, ces outils simples s'avèrent peu performants et commettent souvent des erreurs importantes d'estimation [28], [29]. 3.3.1.2.

Arbres de décision (CART) et critères de choix de séparation (Gini, Entropie) L'arbre de décision, notamment l'algorithme CART (Classification and Regression Trees), découpe le jeu de données logistiques en sous-groupes homogènes à l'aide de règles successives simples (comme "si le poids est supérieur à 5 kg, alors...") [27], [28]. Pour réaliser ces découpes de la manière la plus efficace possible, l'algorithme s'appuie sur deux indicateurs principaux : l'indice d'impureté de Gini et l'entropie de Shannon [25], [28]. L'indice de Gini mesure la probabilité de mal classer un colis dans un sous-groupe, tandis que l'entropie évalue le niveau de désordre ou d'incertitude dans le choix des catégories [27], [28]. Bien que ces arbres de décision soient très faciles à comprendre et à visualiser par les gestionnaires de transport, ils ont tendance à trop s'adapter aux données d'apprentissage (surapprentissage), ce qui les rend moins précis lorsqu'ils doivent prédire de nouvelles livraisons [25], [28]. 3.3.1.3.

Forêts aléatoires (Random Forest) et principe de bagging (agrégation de modèles) La forêt aléatoire permet de corriger la fragilité des arbres de décision simples en combinant les prédictions de plusieurs dizaines d'arbres indépendants pour obtenir un résultat final unique [26], [29]. Cet algorithme repose sur le principe du bagging (ou agrégation de bootstrap), qui consiste à entraîner chaque arbre sur un échantillon de données différent, sélectionné de manière aléatoire avec remise [26], [28]. De plus, lors de la construction des branches, seule une partie des variables d'entrée est évaluée à chaque étape pour garantir la diversité des arbres créés [28], [29]. En faisant voter l'ensemble des arbres, la forêt aléatoire réduit considérablement le risque d'erreur aléatoire, ce qui en fait un modèle robuste et performant pour estimer les temps de trajet ou les tarifs de transport sur les plateformes collaboratives [28], [29]. 3.3.1.4.

Algorithmes de Boosting de Gradient (XGBoost et LightGBM) et leur performance en logistique À l'inverse du bagging où les arbres sont créés de façon indépendante, les modèles de boosting construisent les arbres de décision les uns après les autres [29], [30]. Chaque nouvel arbre est entraîné pour corriger spécifiquement les erreurs de prédiction faites par les arbres précédents, ce qui permet d'améliorer la précision du modèle à chaque étape [13], [30]. Les algorithmes XGBoost (eXtreme Gradient Boosting) et LightGBM (Light Gradient Boosting Machine) sont deux évolutions majeures et très populaires de cette approche, réputées pour leur vitesse de calcul et leur grande précision sur les données complexes [29], [30]. XGBoost intègre des fonctions mathématiques qui limitent la complexité des arbres pour éviter le surapprentissage, tandis que LightGBM accélère l'entraînement en triant les données par feuilles plutôt que par niveaux entiers [29], [30]. Ces outils de boosting sont aujourd'hui considérés comme les plus performants pour prédire les revenus logistiques, les temps de livraison et pour optimiser les performances opérationnelles [28], [30].

En synthèse, les modèles d'ensemble comme le boosting de gradient s'avèrent plus performants car ils savent capturer les relations complexes et non linéaires [28], [29]. 3.3.2. Cadre de validation et métriques d'évaluation de la performance Nous décrivons la division des données (Train/Val/Test), la validation croisée (K-Fold) et les métriques de performance comme la MAE, la RMSE, le coefficient $R^2$, la précision, le rappel et l'AUC-ROC [29], [30]. 3.3.2.1. Prévention du surapprentissage (overfitting) et séparation des données (Train/Val/Test) Le surapprentissage est un problème courant en Machine Learning où le modèle apprend par cœur les données historiques logistiques, y compris les bruits et les cas anormaux, au point de ne plus pouvoir généraliser à de nouvelles commandes [25], [30].

Pour éviter ce défaut, les concepteurs de modèles divisent obligatoirement le jeu de données initial en trois parties distinctes [27], [30] : Le jeu d'apprentissage (Train) , qui sert à entraîner le modèle et à ajuster ses paramètres internes [27], [30]. Le jeu de validation (Val) , utilisé pour régler la configuration de l'algorithme (les hyperparamètres) et détecter le début du surapprentissage [28], [30]. Le jeu de test (Test) , gardé totalement secret jusqu'à la fin, qui sert à évaluer l'efficacité réelle du modèle sur des livraisons inédites [28], [30]. Cette séparation stricte garantit que les prédictions logistiques restent fiables dans des conditions de transport réelles [28], [30]. 3.3.2.2.

Le mécanisme de validation croisée (K-Fold Cross-Validation) Pour estimer de manière stable la performance d'un modèle et limiter l'influence du hasard lors du découpage des données, on utilise la validation croisée en K étapes (K-Fold Cross-Validation) [29], [30]. Cette méthode consiste à diviser le jeu de données en $K$ sous-groupes de taille égale [29], [30]. L'algorithme est entraîné $K$ fois de suite : à chaque itération, $K-1$ sous-groupes servent à l'apprentissage et le groupe restant sert à tester le modèle [29], [30]. En calculant la moyenne des performances sur l'ensemble des étapes, on obtient une évaluation beaucoup plus fiable et représentative [29], [30]. De plus, cette approche répétée permet de proposer des estimations logistiques (comme les tarifs de transport) sous forme d'intervalles de confiance, offrant une information plus riche aux gestionnaires par rapport à une simple valeur fixe [29].

3.3.2.3. Métriques d'évaluation pour les tâches de Classification Dans le cas d'une classification binaire (par exemple, déterminer si un colis arrivera à l'heure ou en retard), l'évaluation s'appuie sur la matrice de confusion, qui croise les états réels et les prédictions du modèle [13], [27]. À partir de cette matrice, on calcule les métriques suivantes [13], [25] : La Précision : le pourcentage de colis réellement en retard parmi tous ceux prédits en retard par le modèle [13], [27]. Le Rappel (ou sensibilité) : le pourcentage de retards réels que le modèle a réussi à identifier [13], [25]. Le score F1 : la moyenne équilibrée de la précision et du rappel, particulièrement utile lorsque les retards sont rares et le jeu de données déséquilibré [13], [27].

La courbe ROC et son indicateur AUC (Area Under the Curve) : l'AUC mesure la capacité générale du modèle à différencier correctement les livraisons conformes de celles en retard [13], [27]. Un score proche de 1 indique un modèle d'une grande fiabilité [27]. 3.3.2.4. Métriques d'évaluation pour les tâches de Régression Lorsque le modèle cherche à prédire une valeur continue (comme le temps de transport en minutes ou le coût exact de la livraison), on utilise d'autres mesures de performance [28], [29], [30] : L'erreur absolue moyenne (MAE) : elle calcule l'écart moyen, en minutes ou en euros, entre la réalité et la prédiction [26], [30] : MAE = (1/n) × Σ(i=1 → n) |yᵢ − ŷᵢ| La racine de l'erreur quadratique moyenne (RMSE) : elle pénalise plus sévèrement les grosses erreurs de prévision en élevant les écarts au carré avant d'en calculer la racine [28], [30] : RMSE = sqrt( (1/n) sum_{i=1}^{n} (y_i - \hat{y}_i)^2 ) Le coefficient de détermination ($R^2$) : il indique la proportion de la variation des données expliquée par le modèle [29], [30] : R^2 = 1 - ( sum_{i=1}^{n} (y_i - \hat{y}_i)^2 ) / ( sum_{i=1}^{n} (y_i - \bar{y})^2 ) Une valeur de R^2 proche de 1 démontre que le modèle décrit fidèlement le comportement réel des flux de livraison [29], [30]. Pour finir, l'utilisation de plusieurs métriques conjointes et de la validation croisée évite le surapprentissage et sécurise la fiabilité des modèles en production [28], [30].

En conclusion, les algorithmes de boosting de gradient (XGBoost, LightGBM) associés à une validation croisée rigoureuse offrent les meilleures garanties de précision et de stabilité logistique [29], [30]. Synthèse comparative des algorithmes candidats en logistique Le tableau ci-dessous résume les caractéristiques, les avantages et les limites des modèles de prédiction logistique : Famille de modèles Algorithmes Avantages clés Limites principales Métriques adaptées Références Linéaire Régression linéaire / logistique Très simple à interpréter, calcul instantané. Incapable de modéliser les relations complexes et non rectilignes. MAE, RMSE, R^2 (Régression) ; Précision, AUC (Classification). [28], [29] Arbre simple CART (Arbre de décision) Règles logiques faciles à lire sous forme de conditions.

Forte sensibilité aux petites variations, risque de surapprentissage. Indice de Gini, Entropie. [27], [28] Bagging Forêt aléatoire (Random Forest) Robuste, réduit la variance, gère bien les données manquantes. Modèle lourd en mémoire, interprétation moins directe. MAE, RMSE, R^2.

[26], [28], [29] Boosting XGBoost / LightGBM Très haute précision, traitement rapide, gère les variables complexes. Nécessite un réglage minutieux des paramètres de configuration. RMSE, MSE, MAE, RAE, R^2. [13], [29], [30] Schéma conceptuel : Processus de validation et choix du modèle optimal Ce schéma DOT (Graphviz) illustre le cheminement des données logistiques à travers le processus de séparation, d'entraînement avec validation croisée et d'évaluation finale : En synthèse, la transition vers la Logistique 4.0 et les modèles de Machine Learning permet aux entreprises de passer d'une gestion réactive à une prédiction proactive des délais [24], [25], [28]. Bilan analytique du chapitre La Logistique 4.0 fournit la matière première, et l'apprentissage supervisé la machinerie inférentielle qui la convertit en anticipation [22], [25].

Le déplacement décisif tient à la temporalité : capter l'information en continu ne suffit pas, encore faut-il l'exploiter avant l'expédition pour transformer un constat en levier d'action [13], [24]. Ce déplacement justifie le choix d'une cible binaire, dont la sortie probabiliste se traduit directement en règle opérationnelle de surveillance [5], [27]. Le spectre des modèles candidats s'ordonne selon un compromis entre interprétabilité et capacité à capter les interactions : le modèle linéaire expose ses coefficients au décideur, tandis que les méthodes d'ensemble gagnent en précision au prix d'une lisibilité moindre [28], [29]. Ce compromis ne se tranche pas dans l'absolu ; il dépend de l'usage visé, et c'est pourquoi la phase expérimentale confronte ces familles avant de retenir l'architecture la mieux adaptée au déséquilibre des classes et aux contraintes de déploiement [13], [30]. La sélection finale obéit donc à une logique d'utilité décisionnelle autant qu'à une logique de performance brute [27], [29].

L'appareillage de validation, du partitionnement stratifié à la validation croisée et au choix de métriques robustes au déséquilibre, conditionne la crédibilité des résultats annoncés [29], [30]. C'est sur ce socle théorique consolidé que s'appuie l'ingénierie algorithmique conduite sur les quatre cent mille expéditions, où les principes énoncés ici deviennent des choix d'implémentation mesurables [13], [30]. Ingénierie Algorithmique et Déploiement du Modèle Prédictif Note méthodologique d'introduction : Cette phase pratique matérialise le cadre théorique précédemment établi. S'appuyant sur un jeu de données représentatif de la supply chain nigériane (400 000 expéditions), l'objectif est de construire une architecture algorithmique robuste capable d'anticiper les défaillances temporelles. La démarche adoptée suit une logique d'ingénierie stricte : de l'assainissement de la donnée brute à la sélection intrinsèque des variables, jusqu'à l'industrialisation du modèle champion sous la forme d'une application web interactive d'aide à la décision.

4.1. Cadrage du périmètre analytique et ingénierie des caractéristiques L'élaboration d'un modèle d'apprentissage supervisé requiert une formalisation rigoureuse de la variable cible et un contrôle strict des informations fournies à l'algorithme. 4.1.1. Définition de la variable cible et traitement du déséquilibre Le spectre initial des statuts de livraison a été restreint pour isoler exclusivement les flux dont le cycle logistique était achevé. Les expéditions marquées en transit ou annulées ont été exclues afin d'éviter toute pollution sémantique.

La problématique a ainsi été traduite en un problème de classification binaire où la variable is_delayed isole les succès (0) des échecs de ponctualité (1). L'audit de cette variable a révélé un fort déséquilibre des classes, le taux de retard effectif s'établissant à environ 8 %. Cette asymétrie structurelle a conditionné l'ensemble des choix méthodologiques ultérieurs, de la stratification des données à la sélection des métriques d'évaluation. 4.1.2. Ingénierie temporelle et imperméabilité du modèle (Anti-Leakage) L'un des défis majeurs de la modélisation prédictive réside dans la prévention de la fuite de données ( Data Leakage ).

Fournir à l'algorithme des variables postérieures au moment de l'expédition (notamment la date de livraison réelle) aurait artificiellement gonflé les performances en rendant la prédiction triviale. Pour pallier ce risque, de nouvelles caractéristiques temporelles ont été extraites à partir des dates autorisées : Le calcul du délai de transit estimé ( estimated_transit_days ), défini par l'écart entre la date de commande et la date de livraison promise. L'extraction des cycles calendaires (mois et jour de la semaine d'expédition) pour capturer les phénomènes de saisonnalité et les congestions de fin de semaine. Immédiatement après cette extraction, l'intégralité des dates brutes a été définitivement supprimée de l'espace vectoriel. 4.2.

Prétraitement dimensionnel et Modélisation Afin de préserver la représentativité du phénomène, le partitionnement des données a été réalisé avec une stratification stricte, allouant 80 % des observations à l'apprentissage et 20 % à la validation, tout en figeant la proportion de retards dans les deux ensembles. 4.2.1. Stratégie d'encodage et sélection intrinsèque des variables La richesse catégorielle du jeu de données (villes d'origine et de destination) menaçait l'architecture d'une explosion dimensionnelle si un encodage binaire classique ( One-Hot ) avait été appliqué. Une stratégie hybride a donc été déployée : Les variables à faible cardinalité (prestataires logistiques) ont subi un encodage disjonctif complet. Les localisations géographiques ont bénéficié d'un encodage ciblé ( Target Encoding ), traduisant le risque historique de retard associé à chaque nœud urbain sans surcharger la mémoire informatique.

Les grandeurs continues (coûts, volumes) ont été standardisées pour aligner leurs échelles. À l'issue de ce prétraitement, une sélection de variables intégrée ( Embedded Feature Selection ) a été opérée à l'aide d'un modèle basé sur les arbres de décision ( Random Forest ). En calculant l'importance de chaque variable dans la réduction de l'impureté des nœuds, l'algorithme a automatiquement exclu les caractéristiques générant du bruit, concentrant ainsi la puissance de calcul sur les signaux pertinents. 4.2.2. Entraînement et optimisation des hyperparamètres Trois architectures distinctes ont été mises en concurrence : un modèle linéaire pénalisé (Régression Logistique) et deux méthodes d'ensemble (Random Forest par Bagging , et XGBoost par Boosting ).

Le déséquilibre des classes a été mathématiquement compensé au sein de chaque algorithme (via la pondération des classes ou l'ajustement du poids des observations positives). Compte tenu de la volumétrie des données, la recherche des hyperparamètres optimaux a été confiée à un processus stochastique ( RandomizedSearchCV ), adossé à une validation croisée à plusieurs plis. Cette méthode a permis d'explorer efficacement l'espace des hyperparamètres tout en maîtrisant les temps de calcul. Tableau 4.1 : Synthèse de l'architecture du pipeline Machine Learning Phase du Pipeline Techniques employées Objectif conceptuel Ingénierie temporelle Extraction (Mois, Jour, Écart estimé) Transformer des timestamps en vecteurs d'apprentissage Prévention du Leakage Suppression des timestamps originaux Garantir la validité opérationnelle de la prédiction Encodage Target Encoding / One-Hot Encoding Gérer la forte cardinalité spatiale (Villes nigérianes) Sélection de variables Embedded Selection (Random Forest) Isoler les variables à fort pouvoir de discrimination Optimisation Randomized Search / Stratified K-Fold Maximiser le pouvoir prédictif sans surapprentissage 4.3. Évaluation discriminante et Industrialisation de la solution Face à une classe minoritaire critique, l'exactitude globale ( Accuracy ) s'avère trompeuse.

L'évaluation a donc reposé sur l'analyse de la matrice de confusion et l'optimisation du F1-score, métrique de compromis mesurant l'équilibre entre la capacité à détecter tous les retards (Rappel) et la capacité à ne pas générer de fausses alertes (Précision). La surface sous la courbe (AUC- ROC) a également permis de valider la robustesse de la séparation spatiale entre les deux classes. Le modèle ayant maximisé ces critères a été figé et encapsulé sous la forme d'un objet sérialisé indépendant. 4.3.1. Déploiement et interface d'aide à la décision (Streamlit) La finalité de ce projet dépassant le simple exercice statistique, le modèle champion a été déployé au cœur d'une application web analytique.

Conçue comme un véritable tableau de bord managérial, cette interface permet à un décideur d'importer une extraction de ses futures expéditions. Le système opère silencieusement l'ensemble des transformations du pipeline (du calcul des jours de transit à l'encodage) pour restituer une probabilité de retard individuelle pour chaque colis. L'outil intègre un volet de visualisation interactive (via la bibliothèque Plotly) traduisant les sorties algorithmiques en leviers d'action : comparaison de la fiabilité des prestataires (DHL, GIG Logistics, etc.), cartographie des axes routiers sous tension, et identification des volumes critiques. L'intégration de cette couche applicative clôture le cycle de la donnée, transformant l'inférence mathématique en une décision stratégique proactive. Schéma 4.1 : Architecture globale du système prédictif et de son déploiement
<!-- OLD_MEMOIRE_EXPANSION_END -->

### Synthèse du chapitre

Le Machine Learning appliqué à la prédiction logistique exige une articulation entre méthode statistique et connaissance métier. La donnée historique permet d'apprendre certaines régularités de durée, mais elle ne garantit pas la prédictibilité du retard binaire lorsque les aléas déterminants ne sont pas observés. Le choix de la cible devient donc un acte méthodologique central.

La suite du mémoire met cette logique à l'épreuve. Le Chapitre 3 partira du fichier Parquet nigérian, construira les cibles, testera la classification du retard, reformulera le problème en régression de durée et évaluera le modèle avec des baselines, une validation croisée, une validation temporelle et une interprétation des variables. Le déploiement Streamlit ne viendra qu'après cette vérification expérimentale, comme matérialisation de la décision et non comme substitut à l'évaluation scientifique.

## Chapitre 3 - Méthodologie expérimentale, entraînement et évaluation du modèle

### Introduction

La méthodologie expérimentale transforme le problème logistique en un protocole vérifiable. Le fichier de données ne suffit pas à définir le modèle ; il faut déterminer quelles variables sont connues avant la livraison, quelle cible peut être construite sans fuite, quel baseline représente la pratique métier minimale et quelle validation simule le mieux l'usage futur. Le présent chapitre suit cette logique dans l'ordre où elle a été exécutée dans le notebook.

Le point de départ est un fichier Parquet contenant des expéditions e-commerce nigérianes. Chaque ligne représente une livraison et rassemble des informations sur le fournisseur, les villes, les dates, la quantité, le coût d'expédition, le transporteur et le statut. Les dates permettent de calculer deux grandeurs distinctes : la durée planifiée, obtenue par différence entre la date attendue et la date d'expédition, et la durée réelle, obtenue par différence entre la date réelle de livraison et la date d'expédition.

### 3.1 Source des données et structure du fichier

Le jeu de données initial contient 400 000 expéditions et 12 variables brutes. Les colonnes principales décrivent l'identifiant d'expédition, le produit, le fournisseur, la ville d'origine, la ville de destination, les trois dates logistiques, la quantité, le coût d'expédition, le transporteur et le statut de livraison. Les identifiants ne sont pas utilisés dans le modèle, car ils individualisent les lignes sans fournir une régularité généralisable.

Le tableau 1 représente le dictionnaire opérationnel des variables conservées pour l'apprentissage. Il distingue les variables directement présentes dans le fichier, les variables construites à partir des dates et les cibles utilisées pour l'évaluation. Cette séparation est importante, car elle empêche de confondre une variable d'entrée disponible en production avec une information connue seulement après la livraison.

**Tableau à insérer : dictionnaire des variables du dataset.**

| Variable | Type | Rôle méthodologique |
|---|---|---|
| `quantity` | numérique | Variable d'entrée décrivant le volume commandé. |
| `shipping_cost_ngn` | numérique | Variable d'entrée décrivant le coût d'expédition. |
| `origin_city` | catégorielle | Variable d'entrée décrivant le point de départ. |
| `destination_city` | catégorielle | Variable d'entrée décrivant le point d'arrivée. |
| `supplier_name` | catégorielle | Variable d'entrée décrivant le fournisseur. |
| `logistics_company` | catégorielle | Variable d'entrée décrivant le transporteur. |
| `ship_month` | catégorielle construite | Mois d'expédition, construit à partir de `ship_date`. |
| `ship_day_of_week` | catégorielle construite | Jour de semaine, construit à partir de `ship_date`. |
| `estimated_transit_days` | numérique construite | Durée planifiée entre expédition et ETA. |
| `realized_transit_days` | cible de régression | Durée réelle entre expédition et livraison. |
| `delay_days` | cible de contre-épreuve | Dépassement positif de l'ETA. |
| `is_delayed` | cible de classification | Indicateur binaire de dépassement ETA. |

### 3.2 Construction des cibles et garde-fou anti-fuite

Les dates brutes sont d'abord converties en format temporel. Les lignes incohérentes, où les durées calculées sont négatives ou impossibles, sont retirées de l'apprentissage. Dans l'exécution expérimentale du notebook, 396 498 lignes restent exploitables après ces contrôles. Dans le déploiement Streamlit, le fichier uploadé est préparé pour la prédiction et conserve 400 000 lignes exploitables, car l'interface ne filtre pas de la même manière les observations destinées à l'inférence.

La cible principale retenue est `realized_transit_days`. Elle mesure la durée réellement observée entre l'expédition et la livraison. La cible secondaire `delay_days` est construite uniquement pour vérifier si le retard pur est prédictible. Le statut de livraison et la date réelle ne sont jamais introduits comme variables d'entrée. Cette exclusion respecte le principe anti-fuite : un modèle déployé ne peut pas utiliser une information qui n'existe qu'après la livraison [5].

La figure 4 représente la distribution des durées planifiées et réelles. Elle montre que la durée réelle est globalement plus élevée que la durée planifiée, ce qui prépare l'interprétation du modèle comme outil de recalibrage de l'ETA plutôt que comme simple détecteur binaire.

**Figure à insérer :** `report_assets/figures/01_distribution_durees.png`

La figure 5 illustre la relation entre la durée planifiée et la durée réelle. L'alignement général indique que l'ETA contient un signal fort, mais la dispersion autour de la diagonale montre que cette promesse n'est pas parfaitement calibrée. Le modèle supervisé exploite précisément cet écart : il apprend à corriger la durée planifiée en fonction des autres variables disponibles.

**Figure à insérer :** `report_assets/figures/02_planifie_vs_reel.png`

### 3.3 Formulation initiale du retard et résultat nul

La première formulation du projet consiste à prédire directement le retard. Cette formulation est intuitive pour le décideur, car elle répond à une question simple : le colis dépassera-t-il l'ETA ou non ? Le notebook teste cette hypothèse par une classification binaire, avec une validation croisée stratifiée et une mesure AUC-ROC. Le résultat se situe autour du hasard, ce qui signifie que les variables disponibles ne discriminent pas correctement les livraisons qui dépasseront l'ETA.

Ce résultat nul n'est pas supprimé de la démarche, car il joue un rôle méthodologique décisif. Il montre que le retard binaire dépend probablement d'informations non présentes dans la base : incidents de tournée, congestion réelle, scans intermédiaires, comportement du client, événements météo locaux plus fins ou capacité effective du transporteur. La démarche bascule alors vers une cible plus stable : la durée réelle de livraison.

Cette décision évite une erreur fréquente dans les projets appliqués. Forcer un classifieur de retard malgré une AUC proche de 0,5 aurait produit une interface séduisante mais fragile. Reformuler le problème en régression revient à reconnaître que la donnée sait mieux estimer une durée probable qu'identifier individuellement chaque retard.

### 3.4 Prétraitement et encodage

Le pipeline de prétraitement est construit avec un `ColumnTransformer`. Les variables numériques sont imputées par la médiane puis standardisées. Cette transformation stabilise les modèles sensibles à l'échelle et évite qu'une variable comme le coût d'expédition domine uniquement par son ordre de grandeur. Les variables catégorielles de faible cardinalité, telles que le fournisseur, le transporteur, le mois et le jour de semaine, sont encodées par One-Hot Encoding.

Les villes d'origine et de destination sont traitées par Target Encoding. Cette stratégie est adaptée aux variables catégorielles plus riches, car elle évite de produire un nombre excessif de colonnes tout en conservant une information moyenne liée à la cible. L'encodage est intégré au pipeline pour être appris sur l'entraînement et appliqué ensuite au test, ce qui réduit le risque de fuite [6].

L'ensemble de ces opérations est encapsulé dans un pipeline unique. Cette architecture assure la cohérence entre le notebook, le script d'entraînement et l'application Streamlit. Le même fichier `delivery_duration_model_bundle.joblib` contient le prétraitement et le modèle, ce qui empêche le site de reproduire manuellement des transformations susceptibles de diverger.

### 3.5 Modèles candidats, baselines et optimisation

Trois familles de modèles sont confrontées : Ridge, Random Forest et HistGradientBoostingRegressor. La comparaison commence par deux baselines. La première prédit la durée moyenne, ce qui fixe un plancher statistique. La seconde recopie la durée planifiée, ce qui constitue le vrai concurrent métier. Un modèle n'a de valeur que s'il améliore cette seconde référence.

Le tableau 2 représente les performances finales obtenues sur le test aléatoire et sur le split temporel. Le modèle final atteint un R² de 0,751 sur le test aléatoire, contre 0,639 pour le baseline métier. La MAE est de 1,987 jour, ce qui signifie qu'en moyenne la prédiction s'écarte de la durée réelle d'environ deux jours.

**Tableau à insérer :** `report_assets/tables/model_metrics.csv`

La figure 6 représente la comparaison entre le modèle et le baseline métier. Elle montre que le gain existe dans les deux protocoles, mais qu'il reste modéré. Cette nuance est importante : le modèle apporte une correction mesurable, sans transformer la donnée en prédiction parfaite.

**Figure à insérer :** `report_assets/figures/04_comparaison_r2.png`

L'optimisation des hyperparamètres est réalisée par recherche aléatoire. Cette méthode permet d'explorer plusieurs combinaisons de `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf` et `l2_regularization` sans parcourir toute la grille [11]. Le modèle final utilisé dans le site est entraîné avec `learning_rate=0.2`, `max_iter=200`, `max_leaf_nodes=31`, `min_samples_leaf=100` et `l2_regularization=5.0`.

### 3.6 Évaluation finale, interprétabilité et validation temporelle

La figure 7 représente l'alignement entre les durées réelles et les durées prédites. Le nuage suit une tendance diagonale, ce qui confirme que le modèle apprend la structure principale de la durée de transit. Les écarts restants traduisent la part d'incertitude que les variables disponibles ne peuvent pas absorber.

**Figure à insérer :** `report_assets/figures/03_reel_vs_predit.png`

La figure 8 présente l'importance par permutation des variables. La durée planifiée domine nettement les autres variables, tandis que le coût, la météo, le jour d'expédition, la destination, le fournisseur ou le transporteur contribuent très faiblement. Cette hiérarchie impose une lecture précise : le modèle fonctionne principalement comme un recalibreur de l'ETA.

**Figure à insérer :** `report_assets/figures/05_importance_permutation.png`

La validation temporelle constitue un test supplémentaire. Le modèle est entraîné sur les expéditions du 1er janvier 2024 au 28 août 2024, puis testé sur les expéditions du 28 août 2024 au 27 octobre 2024. Le R² temporel atteint 0,751, presque identique au R² du split aléatoire. Cette stabilité désamorce la critique d'un découpage favorable.

La règle de dépassement ETA est évaluée comme un classifieur en comparant la durée prédite à la durée planifiée. Elle obtient une AUC-ROC de 0,510 et un taux de fausses alertes de 100 %. Le tableau 3 représente ces métriques. Le rappel et le F1 peuvent sembler élevés, mais ils sont gonflés par le taux de base et par le fait que la règle alerte presque toutes les observations.

**Tableau à insérer :** `report_assets/tables/risk_rule_metrics.csv`

La figure 9 représente la matrice de confusion de cette règle. Elle confirme que le système ne distingue pas finement les retards individuels. Sa valeur ne réside donc pas dans une alerte binaire fiable, mais dans l'estimation d'une durée probable et dans le recalibrage des promesses ETA.

**Figure à insérer :** `report_assets/figures/06_matrice_confusion_regle_eta.png`

La figure 10 représente la distribution de la marge ETA prédite. Une marge positive indique que la durée prédite dépasse la promesse actuelle. Cette visualisation justifie le passage d'un score brut à une règle de tampon, car elle montre que l'action métier ne consiste pas seulement à signaler un retard, mais à quantifier l'ajustement nécessaire.

**Figure à insérer :** `report_assets/figures/07_distribution_marge_eta.png`

La figure 11 représente les groupes prioritaires obtenus après agrégation. Elle prépare directement le déploiement Streamlit : le modèle ne livre pas uniquement une colonne prédite, il produit une hiérarchie opérationnelle par axe et par transporteur.

**Figure à insérer :** `report_assets/figures/08_groupes_prioritaires.png`

### 3.7 Lecture méthodologique détaillée du pipeline expérimental

Le pipeline expérimental suit une logique de réduction progressive de l'incertitude. La première opération consiste à convertir les dates en objets temporels fiables, puis à reconstruire les durées qui serviront à la fois de variables explicatives et de cibles. Cette étape paraît technique, mais elle porte un enjeu scientifique important : une durée mal construite transforme immédiatement le problème d'apprentissage en problème bruité. La cohérence entre `ship_date`, `expected_delivery_date` et `actual_delivery_date` devient donc le premier filtre de validité.

Le choix de conserver `estimated_transit_days` comme variable d'entrée peut sembler discutable si l'on confond prédiction et reprise de la promesse existante. Dans ce projet, cette variable est au contraire centrale, car elle représente l'information réellement disponible au moment où l'entreprise annonce une ETA au client. Le modèle n'apprend pas à ignorer cette promesse ; il apprend à la corriger. Cette nuance justifie toute la reformulation du projet. Si l'ETA porte déjà une grande partie du signal, le travail du modèle consiste à mesurer l'écart probable entre cette promesse et la durée observée.

La séparation entre variables disponibles avant livraison et variables connues après livraison protège la validité du protocole. La date réelle, le statut final et les indicateurs directement construits après réception ne peuvent pas entrer dans les variables d'entrée, car ils ne seraient pas connus au moment du déploiement. Le notebook applique ainsi un raisonnement temporel : une variable n'est acceptable que si elle existe avant que la décision opérationnelle ne soit prise. Cette règle simple évite le data leakage, qui peut produire des scores élevés mais inutilisables en production.

Les variables catégorielles sont traitées selon leur nature. Les transporteurs, fournisseurs, mois et jours de semaine possèdent une cardinalité limitée ; le One-Hot Encoding reste alors lisible et peu coûteux. Les villes d'origine et de destination produisent davantage de combinaisons ; le Target Encoding permet de condenser leur information sans créer une matrice trop large. Le point délicat de cet encodage est le risque de fuite statistique. Pour cette raison, l'encodage est encapsulé dans le pipeline et appris uniquement sur les données d'entraînement avant d'être appliqué au test.

La comparaison entre baselines et modèles complexes constitue une autre protection contre l'illusion de performance. Le baseline naïf par moyenne mesure la difficulté minimale du problème. Le baseline métier, qui reprend la durée planifiée, est plus exigeant, car il représente la pratique déjà disponible sans Machine Learning. Le modèle final n'est donc intéressant que parce qu'il dépasse cette référence métier. Le gain reste modéré, mais il est réel et stable. Cette stabilité importe davantage qu'un score spectaculaire isolé.

Le choix du HistGradientBoostingRegressor repose sur trois arguments. D'abord, il traite correctement les relations non linéaires et les interactions sans imposer une forme linéaire stricte. Ensuite, il reste performant sur des données tabulaires de grande taille. Enfin, il s'intègre proprement dans un pipeline scikit-learn exportable. Le modèle Ridge reste une référence utile, car sa proximité avec le modèle final montre que la relation est dominée par une structure quasi linéaire autour de l'ETA. Le modèle de gradient boosting est donc retenu comme compromis entre performance, robustesse et déploiement.

La validation temporelle joue le rôle de simulation du futur. Un split aléatoire mélange des expéditions anciennes et récentes dans les deux ensembles, ce qui peut lisser les changements de distribution. Le split temporel entraîne le modèle sur les expéditions passées et l'évalue sur les expéditions plus récentes. Le R² temporel proche du R² aléatoire montre que le modèle ne dépend pas uniquement d'une répartition favorable. Ce résultat donne un argument fort devant l'encadrant : le protocole se rapproche davantage d'un usage réel.

L'évaluation de la règle de risque est volontairement traitée comme une contre-épreuve. Le score `durée prédite - ETA actuelle` est utile pour ordonner les actions, mais il ne devient pas automatiquement un classifieur fiable. La matrice de confusion et l'AUC proche de 0,5 confirment que la règle signale presque toutes les expéditions, ce qui produit un rappel élevé et un F1 trompeur. Le taux de fausses alertes révèle la limite. Cette limite n'annule pas le projet ; elle le précise. Le système priorise le recalibrage de l'ETA, il ne prédit pas individuellement chaque retard.

Cette lecture méthodologique permet de justifier chaque étape devant le jury. Le nettoyage garantit la cohérence temporelle. L'anti-fuite garantit l'utilité en production. Les baselines donnent une référence métier. L'optimisation ajuste le modèle sans surpromettre. L'interprétabilité explique la domination de l'ETA. La validation temporelle rend l'évaluation crédible. Enfin, l'évaluation classification de la règle de risque empêche de transformer un indicateur de priorisation en probabilité de retard. L'ensemble forme un pipeline défendable, car il reconnaît ses propres limites.

### Synthèse du chapitre

L'expérience montre que la durée réelle de livraison est prédictible à un niveau exploitable, tandis que le retard binaire reste très faiblement discriminable. Cette différence valide la décision méthodologique principale du mémoire. Le modèle final améliore le baseline métier, conserve une performance stable en validation temporelle et révèle que la durée planifiée porte l'essentiel du signal.

La portée opérationnelle doit rester mesurée. Le modèle ne permet pas d'attribuer causalement les retards à un transporteur, une ville ou un fournisseur. Il permet de corriger une ETA trop optimiste et de produire une marge de décision. Le chapitre suivant décrit comment cette marge est transformée en interface Streamlit, en table de décision et en indicateurs exploitables par un utilisateur logistique.

## Chapitre 4 - Déploiement Streamlit et aide à la décision opérationnelle

Le passage du notebook à une interface web modifie la nature du résultat produit. Le notebook sert à construire, tester et interpréter le modèle. L'application Streamlit sert à rendre ce modèle manipulable par un utilisateur métier, sans lui demander d'exécuter du code Python ni d'inspecter les objets internes du pipeline. Cette distinction est essentielle dans le mémoire, car le site ne remplace pas l'expérimentation scientifique. Il en est la traduction opérationnelle.

L'application développée, nommée LOGI-PREDICT, reçoit un fichier d'expéditions au format tabulaire, applique le bundle entraîné localement, calcule les durées prédites, estime les marges de dépassement de l'ETA, agrège les résultats par axe et par transporteur, puis propose une table de priorisation. Le déploiement garde la même philosophie que le modèle : les données disponibles permettent surtout de recalibrer l'ETA, pas de discriminer finement les causes du retard.

### 4.1 Rôle fonctionnel de l'application

Le site répond à quatre besoins concrets. Le premier est l'inférence en lot : un responsable peut déposer un fichier contenant plusieurs centaines de milliers d'expéditions et obtenir une durée prédite pour chaque ligne. Le deuxième est le recalibrage de l'ETA : la différence entre la durée prédite et la durée planifiée indique si la promesse actuelle paraît trop courte. Le troisième est la priorisation : les axes et transporteurs sont classés selon le risque moyen, le volume et le tampon ETA recommandé. Le quatrième est l'explication : l'interface rappelle les limites du score de risque et affiche l'importance des variables pour éviter une lecture causale abusive.

La figure 12 représente la page d'accueil après le chargement du fichier de test nigérian. Elle montre les paramètres latéraux, le niveau de service visé, l'option d'enrichissement météo, les colonnes minimales attendues et les premières métriques globales. Cette page fixe immédiatement le contrat d'utilisation : l'utilisateur fournit des expéditions, le site produit une durée prédite, un score de priorisation et une recommandation de tampon ETA.

**Figure à insérer :** `report_assets/figures/10_site_accueil_upload.png`

Le fichier déposé contient 400 000 lignes. L'interface affiche 400 000 lignes exploitables lors de l'inférence opérationnelle, ce qui signifie que les colonnes nécessaires à la prédiction sont présentes et que le pipeline peut produire un résultat pour chaque expédition importée. Lorsque l'on compare ce chiffre avec les 396 498 lignes utilisées dans le notebook, l'écart vient de la différence entre une base d'apprentissage, qui retire les observations incohérentes pour évaluer le modèle, et une base d'inférence, qui applique le modèle dès que les variables d'entrée sont disponibles.

### 4.2 Chaîne de traitement appliquée au fichier importé

Le fonctionnement du site suit une chaîne stable. Le fichier est d'abord lu et contrôlé. Les dates `ship_date` et `expected_delivery_date` sont converties, puis la durée planifiée `estimated_transit_days` est reconstruite. Les variables calendaires, comme le mois et le jour de semaine, sont dérivées de la date d'expédition. Les variables numériques et catégorielles sont ensuite transmises au pipeline enregistré dans `delivery_duration_model_bundle.joblib`. Le site ne réécrit pas les transformations du notebook ; il charge le même objet de prétraitement et de prédiction.

La figure 13 représente l'aperçu des prédictions ligne par ligne. Chaque observation conserve ses informations d'origine, puis reçoit des colonnes calculées comme la durée planifiée, la durée prédite et la marge par rapport à l'ETA. Ce niveau de détail est utile pour l'audit, car il permet de relier une recommandation agrégée aux expéditions individuelles qui l'ont produite.

**Figure à insérer :** `report_assets/figures/11_site_table_predictions.png`

Le tableau 4 représente la synthèse opérationnelle extraite du fichier importé. La durée planifiée moyenne est proche de 7,51 jours, tandis que la durée prédite moyenne atteint environ 9,04 jours. L'écart moyen est donc de 1,54 jour. Ce résultat ne doit pas être lu comme une erreur du modèle, mais comme une correction moyenne de l'ETA actuelle : sur cette base, la promesse initiale semble systématiquement trop courte.

**Tableau à insérer :** `report_assets/tables/site_summary.csv`

Le taux de dépassement prédit atteint 100 %. Cette valeur paraît extrême, mais elle reste cohérente avec la logique du modèle et avec l'évaluation de la règle de risque. Le site ne dit pas que chaque colis sera certainement en retard. Il indique que, pour presque toutes les lignes, la durée prédite dépasse légèrement la durée promise. Cette nuance change toute l'interprétation : l'interface révèle une ETA globalement sous-estimée, non une capacité magique à connaître le destin exact de chaque livraison.

### 4.3 Table de décision et tampon ETA

L'onglet d'aide à la décision transforme les prédictions ligne par ligne en recommandations agrégées. L'agrégation se fait par axe, défini comme la combinaison ville d'origine vers ville de destination, puis par transporteur. Pour chaque groupe, l'application calcule le volume, le risque moyen, le taux de dépassement prévu, la durée prédite moyenne, la durée planifiée moyenne et un tampon ETA recommandé. Le tampon correspond au nombre de jours à ajouter à la promesse actuelle pour atteindre le niveau de service sélectionné.

La figure 14 représente l'écran d'aide à la décision. Les indicateurs affichés résument la situation opérationnelle : tampon moyen de 3 jours, risque élevé sur 399 986 expéditions, expéditions à risque à 100 % et risque moyen de 78,7 %. Les cartes de recommandation ne cherchent pas à expliquer causalement les retards. Elles orientent l'action : ajuster d'abord l'axe prioritaire, surveiller les axes dont le taux de dépassement est maximal, comparer l'allocation transporteur et démarrer par un nombre limité de groupes.

**Figure à insérer :** `report_assets/figures/12_site_aide_decision.png`

La figure 15 représente la table de décision détaillée. Les premières lignes concernent notamment `Abuja -> Port Harcourt` avec `Dispatch Riders`, `Kano -> Kaduna` avec `Dispatch Riders`, `Kano -> Jos` avec `Dispatch Riders` et `Abuja -> Warri` avec `Fedex Nigeria`. Ces groupes ont un volume élevé, un risque moyen très proche de 0,79 et un tampon recommandé de 3 jours. La confiance affichée est forte parce que le volume de lignes est suffisant pour rendre l'agrégation stable.

**Figure à insérer :** `report_assets/figures/13_site_table_decision.png`

Le tableau 5 représente les premiers groupes de décision exportés par le site. Il doit être utilisé dans la partie pratique comme preuve que le déploiement ne se limite pas à afficher une prédiction brute. Il transforme les sorties du modèle en objets de pilotage : axe, transporteur, volume, marge ETA, priorité et action recommandée.

**Tableau à insérer :** `report_assets/tables/top_decision_groups.csv`

La lecture de cette table doit rester prudente. Les transporteurs listés en haut ne sont pas nécessairement responsables du retard. Ils apparaissent dans des groupes où la promesse ETA est sous-calibrée et où le volume donne une priorité opérationnelle. Le professeur peut interroger cette différence ; la réponse correcte est que le modèle priorise des couples axe-transporteur, mais ne prouve pas une causalité transporteur.

### 4.4 Visualisation des risques par axe et par transporteur

L'onglet risques et axes apporte une lecture graphique des agrégations. La figure 16 représente les axes à risque et le risque par transporteur. Les barres les plus longues indiquent les groupes dont le risque moyen de dépassement ETA est le plus élevé. La couleur résume le taux de dépassement prévu, mais cette couleur doit être lue comme un indicateur de priorisation, non comme une probabilité calibrée de retard individuel.

**Figure à insérer :** `report_assets/figures/14_site_risques_axes_transporteurs.png`

Le graphique par transporteur montre des valeurs très proches, autour de 79 %. Cette proximité est informative. Elle suggère que la structure du problème n'est pas portée par un seul transporteur défaillant, mais par une sous-estimation plus générale des délais. La décision managériale ne doit donc pas consister à sanctionner un prestataire sur la base du score. Elle consiste plutôt à vérifier les promesses ETA par axe, à discuter les seuils de service et à tester si certains transporteurs offrent une meilleure stabilité sur des trajets précis.

La figure 17 représente la matrice risque x tampon ETA. Les points regroupent les axes et transporteurs selon leur risque moyen et le nombre de jours à ajouter. Dans l'exécution fournie, la plupart des points se situent autour d'un tampon de 3 jours, ce qui confirme une recommandation homogène de recalibrage. Cette homogénéité est cohérente avec l'importance dominante de `estimated_transit_days` : le modèle ajuste surtout une promesse existante plutôt que de découvrir une segmentation très fine.

**Figure à insérer :** `report_assets/figures/15_site_matrice_risque_eta.png`

La figure 18 représente la distribution de la marge ETA prédite. Cette visualisation est moins riche que les autres, car la masse de données se concentre fortement autour d'une marge positive. Elle peut être placée en annexe si l'espace du chapitre devient trop chargé. Son intérêt principal est de montrer que la règle `durée prédite - ETA actuelle` génère majoritairement des marges positives, ce qui explique pourquoi l'application recommande un tampon.

**Figure à insérer :** `report_assets/figures/16_site_distribution_marge_eta.png`

### 4.5 Traçabilité, validation et interprétation dans l'interface

Le déploiement ne se limite pas à produire une table finale. Il expose aussi un journal de préparation. La figure 19 représente les contrôles effectués au moment de l'import : nombre de lignes brutes, lignes retirées, lignes exploitables, disponibilité de la durée réelle et nombre de variables météo. Ce journal rend le processus vérifiable par un lecteur non technique. Il montre aussi que `actual_delivery_date`, lorsqu'elle existe, sert à comparer le prédit au réel, mais n'entre jamais comme variable explicative dans la prédiction.

**Figure à insérer :** `report_assets/figures/17_site_journal_preparation.png`

La figure 40 représente l'onglet modèle et interprétation. L'interface affiche le modèle déployé, la MAE de 1,99 jour, l'état de l'enrichissement météo, le nombre de lignes utilisées pour l'entraînement, le R² temporel de 0,751, l'AUC risque de 0,510, le taux de fausses alertes de 100 % et le taux réel de dépassement de 62,9 %. Cette concentration d'indicateurs permet de présenter le système avec honnêteté : la régression de durée est stable, mais la règle de risque n'est pas un classifieur fiable de retard individuel.

**Figure à insérer :** `report_assets/figures/18_site_modele_interpretation.png`

La figure 41 représente l'importance des variables affichée dans le site. `estimated_transit_days` domine toutes les autres variables. Cette domination doit être assumée dans la soutenance. Elle prouve que le modèle exploite principalement l'ETA existante pour la recalibrer, tandis que les variables de transporteur, de ville, de fournisseur, de coût et de météo apportent un complément marginal. Ce résultat peut sembler moins spectaculaire qu'une explication complexe, mais il est plus solide scientifiquement.

**Figure à insérer :** `report_assets/figures/19_site_importance_validation.png`

L'onglet assistant fournit enfin une couche de lecture en langage naturel. La figure 42 représente une question posée à l'assistant décisionnel et la réponse générée localement. L'assistant n'est pas un chatbot connecté à un grand modèle de langage. Il est local et fondé sur des règles. Son rôle est d'expliquer les métriques du fichier déposé et de rappeler les priorités visibles dans la table de décision. Cette limitation est volontaire : elle évite de donner à l'utilisateur une illusion d'intelligence conversationnelle non maîtrisée.

**Figure à insérer :** `report_assets/figures/20_site_assistant_decisionnel.png`

### 4.6 Modalités de mise en accès ouvert

Le site fonctionne localement à l'adresse `http://localhost:8501`. Ce lien est suffisant pour une démonstration sur la machine de développement, mais il n'est pas un accès public. Pour rendre l'application consultable par l'encadrant ou par un jury, il faut déposer le code sur un dépôt GitHub, ajouter les fichiers nécessaires au lancement, puis connecter ce dépôt à Streamlit Community Cloud. La commande locale de référence reste :

```bash
streamlit run app.py
```

Le dépôt public doit contenir au minimum `app.py`, `pipeline_utils.py`, `requirements.txt`, le bundle du modèle ou une procédure de génération contrôlée, ainsi qu'un fichier `README.md` expliquant les colonnes attendues. Si le bundle est trop volumineux ou si les données ne doivent pas être exposées, le modèle peut être stocké dans un espace privé ou régénéré à partir d'un script d'entraînement documenté. Pour une soutenance, la solution la plus simple consiste à publier seulement l'application et le bundle, sans publier la base complète.

Le lien public final aura la forme `https://nom-application.streamlit.app`. Tant que l'application n'est pas effectivement publiée, le mémoire doit mentionner le lien local de test et la procédure de publication, pas inventer un lien public. Cette précision protège la crédibilité du rapport : le déploiement est réalisé techniquement, mais l'ouverture au public dépend de la création du dépôt et de l'autorisation de publier les fichiers.

### 4.7 Lecture opérationnelle du déploiement et conditions d'usage

Le déploiement Streamlit doit être compris comme une couche d'aide à la décision, non comme une automatisation fermée. L'utilisateur conserve la responsabilité de l'arbitrage final, mais il dispose d'une information plus structurée que dans un fichier brut. Cette logique correspond bien à un contexte logistique : les décisions doivent souvent être prises rapidement, avec des volumes élevés, des contraintes de service et une incertitude résiduelle impossible à supprimer.

L'import du fichier constitue la première étape métier. L'application impose des colonnes minimales pour éviter une prédiction silencieuse sur des données incomplètes. Cette vérification a une valeur pratique importante, car un modèle déployé échoue rarement uniquement à cause de son algorithme ; il échoue souvent parce que les données envoyées ne respectent pas le format attendu. La présence d'un journal de préparation permet donc de rendre visible ce qui a été accepté, rejeté ou transformé.

La table de prédictions ligne par ligne répond à un besoin d'audit. Un responsable peut revenir à une expédition précise et comprendre pourquoi elle entre dans les lignes à surveiller. Cette granularité protège le système contre une lecture purement agrégée. Elle facilite aussi la comparaison avec la réalité lorsque les livraisons sont terminées. Dans une phase de déploiement pilote, cette table peut être conservée pour analyser les écarts entre durée prédite et durée réellement observée.

La table de décision répond à un besoin différent. Elle ne cherche pas à examiner chaque colis, mais à organiser l'action. Le regroupement par axe et transporteur permet de passer d'une prédiction individuelle à une priorité opérationnelle. Un groupe combinant volume élevé, marge ETA positive et risque moyen élevé devient plus important qu'une expédition isolée. Cette agrégation transforme le modèle en outil de pilotage : elle indique où commencer, quel tampon appliquer et quels flux surveiller.

Le tampon ETA recommandé doit être présenté comme un paramètre métier. Il ne s'agit pas seulement d'une sortie statistique ; il traduit une politique de service. Un niveau premium peut exiger un tampon plus prudent, tandis qu'un service standard peut accepter une marge plus faible. Le site rend cette logique visible à travers le niveau de service sélectionné. Cette fonctionnalité permet de relier le modèle aux arbitrages réels de l'entreprise : satisfaire le client, éviter les promesses irréalistes, mais ne pas allonger inutilement tous les délais annoncés.

Les graphiques par axe et transporteur servent à repérer des concentrations de risque. Leur interprétation doit toutefois rester prudente. Si plusieurs transporteurs affichent des risques proches, le résultat ne permet pas de conclure qu'un transporteur est structurellement défaillant. Il indique plutôt que la promesse ETA est globalement sous-calibrée ou que les variables disponibles ne suffisent pas à séparer finement les comportements. Cette prudence est essentielle dans la rédaction, car elle évite de transformer une corrélation opérationnelle en accusation causale.

L'onglet modèle et interprétation joue un rôle de transparence. Il rappelle la MAE, le R² temporel, l'AUC de la règle de risque et l'importance des variables. Un outil métier qui cache ses limites peut sembler plus convaincant au premier regard, mais il devient fragile dès qu'un expert interroge ses métriques. Ici, l'interface affiche explicitement que l'AUC de risque est proche de 0,5. Cette information oriente la lecture : la valeur du système réside dans le recalibrage de l'ETA et la priorisation des flux, non dans une alerte individuelle parfaitement discriminante.

L'assistant décisionnel local complète l'interface en reformulant les résultats. Il ne remplace pas une analyse experte, mais il aide un utilisateur non technique à comprendre les termes clés : ETA, tampon, score de risque, axe, transporteur. Sa nature locale et rule-based doit être mentionnée dans le mémoire. Un véritable chatbot IA demanderait une API LLM, une gestion des prompts, un contrôle des réponses et une politique de confidentialité. Le choix actuel est plus simple et plus défendable pour un PFE, car il évite d'ajouter une dépendance externe non nécessaire.

La publication ouverte via Streamlit Community Cloud demande plusieurs précautions. Le dépôt GitHub ne doit pas exposer de données sensibles. Si le fichier d'entraînement contient des informations commerciales ou clients, il doit rester privé. Le bundle du modèle peut être publié seulement s'il ne permet pas de reconstruire des données confidentielles. Le fichier `requirements.txt` doit fixer les versions nécessaires, notamment scikit-learn, afin d'éviter qu'un changement de version casse le chargement du modèle. Le fichier README doit documenter les colonnes attendues et la commande locale de lancement.

Dans la phase de soutenance, le site peut être présenté en trois temps. D'abord, l'import du fichier montre que le modèle accepte une base volumineuse et applique le pipeline sans intervention manuelle. Ensuite, l'onglet d'aide à la décision montre la transformation de la prédiction en recommandations concrètes. Enfin, l'onglet modèle et interprétation montre que le système reste scientifiquement honnête. Cette démonstration raconte le projet en entier : préparation des données, apprentissage, validation, interprétation et déploiement.

La limite principale du déploiement vient de l'absence de boucle de retour en production. Le site applique un modèle déjà entraîné, mais il ne réentraîne pas automatiquement le modèle lorsque de nouvelles livraisons réelles sont observées. Pour une version industrielle, il faudrait ajouter une base d'historisation, une procédure de monitoring des performances, une détection de dérive et un mécanisme de réentraînement contrôlé. Ces extensions dépassent le périmètre du PFE, mais elles constituent une perspective crédible et directement liée au travail réalisé.

Le déploiement final doit donc être décrit comme un prototype opérationnel avancé. Il est capable de recevoir un fichier, produire des prédictions, agréger les résultats, recommander des tampons ETA, afficher des visualisations et exporter des tables. Il n'est pas encore une plateforme industrielle intégrée à un ERP ou à un TMS. Cette formulation protège le rapport contre l'exagération. Elle montre que le projet atteint bien la phase de déploiement applicatif, tout en reconnaissant les conditions nécessaires à une mise en production réelle.

### Synthèse du chapitre

Le déploiement Streamlit rend le modèle utilisable dans une logique de pilotage. L'utilisateur charge un fichier, obtient des prédictions, visualise les marges ETA, télécharge une table de décision et consulte une interprétation compacte du modèle. Les résultats de la base nigériane montrent une sous-estimation nette des délais planifiés : la durée prédite moyenne dépasse la durée estimée d'environ 1,54 jour, et les groupes prioritaires reçoivent un tampon recommandé de 3 jours.

La limite principale reste visible dans l'interface elle-même. Le score de risque ne constitue pas une probabilité calibrée de retard individuel. Il sert à prioriser des axes, des transporteurs et des actions de recalibrage. Cette transparence transforme une faiblesse apparente en qualité méthodologique. Le projet ne promet pas de prédire chaque retard ; il fournit un outil mesuré pour ajuster les promesses ETA et organiser la surveillance opérationnelle.

## Conclusion générale

Ce mémoire est parti d'une question opérationnelle simple : peut-on exploiter les données d'expédition e-commerce pour anticiper les délais de livraison et améliorer la promesse ETA ? La réponse obtenue est nuancée, mais robuste. Les données disponibles permettent de prédire la durée réelle de livraison avec une performance stable, tandis qu'elles ne permettent pas de discriminer finement le retard individuel. Cette distinction constitue le résultat central du projet.

La première formulation, orientée vers la classification du retard, a montré une limite nette. L'AUC proche de 0,5 indique que les variables observées ne séparent pas correctement les expéditions qui dépasseront l'ETA de celles qui la respecteront. Ce résultat n'a pas été masqué, car il éclaire le mécanisme du problème. Le retard binaire dépend d'événements fins qui ne figurent pas dans la base : incidents de tournée, capacité réelle du transporteur, congestion locale, disponibilité du client ou rupture ponctuelle dans la chaîne de traitement.

La reformulation en régression a donné un résultat plus exploitable. En prédisant la durée réelle de livraison, le modèle HistGradientBoosting atteint un R² de 0,751 sur le test aléatoire et un R² de 0,751 sur le split temporel. Cette stabilité montre que la performance ne vient pas d'un découpage artificiellement favorable. La MAE proche de 1,99 jour rappelle cependant que le modèle reste un outil d'aide à la décision, pas un oracle opérationnel.

L'interprétabilité confirme le rôle central de `estimated_transit_days`. Cette variable domine très largement les autres dans l'importance par permutation et dans les analyses du notebook. Le transporteur, la ville, le fournisseur, le coût d'expédition et la météo ajoutent peu de signal mesurable. Ce résultat impose une lecture honnête : le modèle apprend principalement à recalibrer une ETA déjà existante. Il ne découvre pas une structure causale complète des retards.

Le déploiement Streamlit transforme cette connaissance en usage. Sur la base nigériane testée, le site traite 400 000 expéditions, prédit une durée moyenne d'environ 9,04 jours contre une durée planifiée moyenne de 7,51 jours, et recommande un tampon moyen de 3 jours sur les groupes prioritaires. Les tableaux de décision classent les couples axe-transporteur selon leur volume, leur risque moyen et leur besoin de recalibrage. Cette sortie est plus utile qu'une simple prédiction ligne par ligne, car elle fournit directement des objets de pilotage.

La contribution principale du projet se situe donc dans l'alignement entre science des données et décision logistique. Le notebook démontre ce qui est prédictible et ce qui ne l'est pas. Le site applique cette frontière sans la maquiller. Les données disponibles permettent surtout de recalibrer l'ETA, pas de prédire avec certitude chaque retard. Cette conclusion est moins spectaculaire qu'une promesse d'automatisation totale, mais elle est plus défendable devant un encadrant et plus utile pour un déploiement réel.

Les perspectives les plus pertinentes concernent l'enrichissement de la donnée. L'ajout de scans intermédiaires, de statuts de tracking, de mesures de congestion, de coordonnées géographiques, d'incidents transporteurs et d'informations clients pourrait renforcer la prédiction du retard binaire. Une expérimentation en production permettrait aussi de tester l'effet réel du tampon ETA sur la satisfaction client et sur le nombre de réclamations. Le modèle actuel constitue ainsi une base solide : il établit une méthode, une limite et un premier outil opérationnel.

## Résumé

Ce projet de fin d'études porte sur l'entraînement d'un modèle de Machine Learning pour la prédiction et le recalibrage des délais de livraison e-commerce. La problématique part d'un écart fréquent entre l'ETA promise au client et la durée réellement observée. Les données utilisées décrivent les dates d'expédition et de livraison attendue, les villes d'origine et de destination, les transporteurs, les fournisseurs, les quantités, les coûts d'expédition et des variables météo lorsque l'enrichissement est disponible.

La démarche expérimentale compare d'abord la prédiction du retard binaire et la prédiction de la durée réelle. Les résultats montrent que le retard individuel est très faiblement discriminable, avec une AUC de risque proche de 0,5. La régression de durée donne en revanche une performance stable, avec un R² d'environ 0,751 en validation aléatoire et temporelle. L'analyse d'importance montre que la durée planifiée domine le signal, ce qui conduit à interpréter le modèle comme un outil de recalibrage de l'ETA.

Le modèle final est intégré dans une application Streamlit nommée LOGI-PREDICT. Le site permet d'importer un fichier d'expéditions, de produire les durées prédites, d'estimer les marges ETA, de classer les axes à risque et de recommander un tampon opérationnel. Sur la base nigériane de 400 000 lignes, la durée prédite moyenne dépasse la durée planifiée d'environ 1,54 jour et les groupes prioritaires reçoivent un tampon recommandé de 3 jours. Le système ne promet pas une prédiction certaine de chaque retard ; il fournit une aide mesurée au pilotage des promesses de livraison.

**Mots-clés :** e-commerce, logistique, ETA, Machine Learning, régression, Streamlit, aide à la décision, recalibrage des délais.

## Abstract

This final-year project focuses on training a Machine Learning model to predict and recalibrate e-commerce delivery lead times. The study starts from a recurring gap between the ETA promised to customers and the actual observed delivery duration. The dataset includes shipment dates, expected delivery dates, origin and destination cities, carriers, suppliers, quantities, shipping costs and weather variables when enrichment is available.

The experimental workflow first compares binary delay prediction with actual duration prediction. The results show that individual delay is poorly discriminated, with a risk AUC close to 0.5. Duration regression is more stable and reaches an R² of about 0.751 under both random and temporal validation. Feature importance analysis shows that the planned transit duration carries most of the signal, which positions the model mainly as an ETA recalibration tool.

The final model is deployed in a Streamlit application called LOGI-PREDICT. The website allows users to upload shipment data, generate predicted durations, compute ETA margins, rank risky routes and recommend an operational ETA buffer. On the Nigerian dataset containing 400,000 rows, the predicted average duration exceeds the planned average duration by about 1.53 days, while priority groups receive a recommended buffer of 3 days. The system does not claim to predict every individual delay with certainty; it provides a controlled decision-support layer for delivery promise management.

**Keywords:** e-commerce, logistics, ETA, Machine Learning, regression, Streamlit, decision support, lead-time recalibration.

## Bibliographie finale avec liens de téléchargement ou de consultation

[1] S. F. W. T. Lim, X. Jin et J. S. Srai, « Consumer-driven e-commerce: A literature review, design framework, and research agenda on last-mile logistics models », *International Journal of Physical Distribution & Logistics Management*, 2018. DOI : https://doi.org/10.1108/IJPDLM-02-2017-0081

[2] L. Ranieri, S. Digiesi, B. Silvestri et M. Roccotelli, « A Review of Last Mile Logistics Innovations in an Externalities Cost Reduction Vision », *Sustainability*, 2018. DOI : https://doi.org/10.3390/su10030782

[3] N. Boysen, S. Fedtke et S. Schwerdfeger, « Last-mile delivery concepts: a survey from an operational research perspective », *OR Spectrum*, 2021. DOI : https://doi.org/10.1007/s00291-020-00607-8

[4] World Economic Forum, *The Future of the Last-Mile Ecosystem*, 2020. Lien : https://www.weforum.org/reports/the-future-of-the-last-mile-ecosystem/

[5] S. Kaufman, S. Rosset, C. Perlich et O. Stitelman, « Leakage in data mining: Formulation, detection, and avoidance », *ACM Transactions on Knowledge Discovery from Data*, 2012. DOI : https://doi.org/10.1145/2382577.2382579

[6] D. Micci-Barreca, « A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems », *ACM SIGKDD Explorations Newsletter*, 2001. DOI : https://doi.org/10.1145/507533.507538

[7] I. Guyon et A. Elisseeff, « An introduction to variable and feature selection », *Journal of Machine Learning Research*, 2003. Lien : https://www.jmlr.org/papers/v3/guyon03a.html

[8] R. Kohavi, « A study of cross-validation and bootstrap for accuracy estimation and model selection », *International Joint Conference on Artificial Intelligence*, 1995. Lien : https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf

[9] J. H. Friedman, « Greedy function approximation: A gradient boosting machine », *The Annals of Statistics*, 2001. DOI : https://doi.org/10.1214/aos/1013203451

[10] L. Breiman, « Random Forests », *Machine Learning*, 2001. DOI : https://doi.org/10.1023/A:1010933404324

[11] J. Bergstra et Y. Bengio, « Random search for hyper-parameter optimization », *Journal of Machine Learning Research*, 2012. Lien : https://www.jmlr.org/papers/v13/bergstra12a.html

[12] S. M. Lundberg et S.-I. Lee, « A unified approach to interpreting model predictions », *Advances in Neural Information Processing Systems*, 2017. Lien : https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

## Annexes à intégrer

L'annexe A contient les extraits du notebook académique, notamment les cellules de préparation des données, de construction des cibles, de comparaison des modèles, de validation temporelle et d'évaluation de la règle de risque ETA.

L'annexe B contient les principaux graphiques expérimentaux : distribution des durées, relation durée planifiée versus durée réelle, prédiction versus réel, comparaison des R², importance par permutation et matrice de confusion de la règle ETA.

L'annexe C contient les captures de l'application Streamlit : page d'accueil, table de prédictions, aide à la décision, table de décision, risques par axe et par transporteur, journal de préparation, interprétation du modèle et assistant décisionnel.

L'annexe D contient les tables exportées par le site : synthèse des indicateurs, groupes prioritaires, métriques du modèle et métriques de la règle de risque.
