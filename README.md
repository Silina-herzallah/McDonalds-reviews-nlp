# McDonalds-reviews-nlp
NLP analysis of McDonald's customer reviews: sentiment classification using Machine Learning and Transformers, topic modeling, and interactive dashboard for customer insights.


# Analyse intelligente des avis clients McDonald's – NLP

**Matière :** Analyse et restitution  
**Formation :** Master 1 – Machine Learning et Science des Données (MLSD)  
**Année universitaire :** 2026–2027  
**Enseignante :** Ikram Chourib  
**Binôme :** Silina Herzallah et Mehdi El Gourche  
**Dépôt GitHub :** `McDonalds-reviews-nlp`

---

## 1. Présentation du projet

Les entreprises disposent aujourd'hui d'un volume important d'avis clients, dont l'exploitation manuelle est complexe et chronophage.

Le traitement automatique du langage naturel (NLP) permet d'analyser ces données textuelles afin de mieux comprendre la satisfaction des consommateurs et d'identifier les principaux axes d'amélioration.

Dans ce projet, nous nous intéressons aux avis clients des restaurants McDonald's. Nous souhaitons développer une chaîne d'analyse permettant de classifier les sentiments exprimés, d'identifier les thèmes récurrents et de restituer les résultats sous une forme exploitable pour la prise de décision.

### Problématique

**Comment les méthodes de traitement automatique du langage naturel peuvent-elles permettre d'évaluer la satisfaction des clients McDonald's, d'identifier les principales sources d'insatisfaction et d'orienter les décisions d'amélioration ?**

Nous chercherons également à déterminer si l'utilisation d'un modèle Transformer apporte un gain de performance significatif par rapport à une méthode classique de Machine Learning.

## 2. Présentation des données

Le projet repose sur le jeu de données **McDonald's Reviews**, disponible au format CSV.

**Fichier :** `McDonald_s_Reviews.csv`

Il contient des avis clients rédigés principalement en anglais, accompagnés de notes et d'informations sur les restaurants concernés.

Les principales variables étudiées seront :

| Variable | Description |
|---|---|
| `review` | Texte de l'avis client |
| `rating` | Note attribuée par le client |
| `store_address` | Adresse du restaurant |
| `review_time` | Indication temporelle de l'avis |
| `rating_count` | Nombre d'évaluations associées au restaurant |

### Construction de la variable cible

Pour la classification des sentiments, les notes seront regroupées en trois classes conformément au cahier des charges :

| Note | Sentiment |
|---|---|
| 1 à 2 étoiles | Négatif |
| 3 étoiles | Neutre |
| 4 à 5 étoiles | Positif |

Cette classification servira de référence pour l'entraînement et l'évaluation des modèles supervisés.

## 3. Niveau et périmètre de l'analyse

L'analyse sera réalisée à trois niveaux :

- **Avis individuel :** identifier le sentiment exprimé et les thèmes abordés.
- **Restaurant :** comparer la satisfaction des clients entre les différents établissements.
- **Global :** dégager les tendances générales et les principaux facteurs de satisfaction ou d'insatisfaction.

Le projet portera sur deux axes principaux :

1. **Classification des sentiments :** déterminer automatiquement si un avis est positif, neutre ou négatif.
2. **Analyse des thèmes :** identifier les sujets les plus fréquemment abordés par les clients, notamment la qualité de la nourriture, le service, les prix, la propreté ou le temps d'attente.

Les résultats seront ensuite exploités pour proposer des recommandations concrètes.

## 4. Objectifs

Les objectifs du projet sont les suivants :

- Explorer et comprendre la structure du jeu de données.
- Préparer les textes à l'aide d'un nettoyage paramétrable.
- Développer un premier modèle classique de classification des sentiments.
- Comparer ce modèle à un Transformer préentraîné puis ajusté sur nos données.
- Identifier automatiquement les thèmes récurrents dans les avis.
- Évaluer la fiabilité et les limites des méthodes utilisées.
- Construire un tableau de bord interactif destiné à faciliter l'interprétation des résultats.
- Formuler des recommandations à partir des principaux points de friction identifiés.

## 5. Méthodologie envisagée

Nous proposons une pipeline organisée en plusieurs étapes.

### Étape 1 – Exploration des données

Une première analyse exploratoire permettra d'étudier :

- Le nombre d'observations et les variables disponibles.
- Les valeurs manquantes et les doublons.
- La répartition des notes et des sentiments.
- La longueur des avis.
- Les mots les plus fréquents.
- La distribution des avis entre restaurants.

Cette étape permettra également d'identifier les éventuelles limites du dataset.

### Étape 2 – Prétraitement des textes

Nous mettrons en place une chaîne de nettoyage paramétrable afin de comparer différentes stratégies de préparation des textes.

Les opérations envisagées comprennent la normalisation, la gestion de la ponctuation et des caractères spéciaux, ainsi que la suppression éventuelle des mots vides.

Une attention particulière sera portée à la négation, qui peut modifier le sens d'un avis.

Les doublons seront supprimés avant la séparation des données en ensembles d'entraînement, de validation et de test.

### Étape 3 – Classification des sentiments

Nous comparerons deux approches principales.

**Approche classique : TF-IDF + régression logistique**

La méthode TF-IDF permettra de transformer les textes en représentations numériques. Une régression logistique sera ensuite entraînée pour prédire les trois classes de sentiments.

Cette approche constituera notre modèle de référence (*baseline*).

**Approche Transformer : DistilBERT**

Nous utiliserons DistilBERT, un modèle de langage préentraîné, que nous ajusterons à notre tâche de classification en trois classes.

L'objectif sera de mesurer son éventuel gain de performance par rapport à la baseline classique.

Les modèles seront comparés selon :

- Le F1-score macro.
- Les performances par classe.
- Les matrices de confusion.
- Le temps d'inférence.
- La consommation mémoire.

Nous étudierons également l'effet des différentes stratégies de nettoyage sur les résultats.

### Étape 4 – Identification des thèmes

Pour identifier les sujets récurrents dans les avis, nous envisageons d'utiliser :

**TF-IDF + K-means**

TF-IDF permettra de représenter les avis numériquement, tandis que K-means cherchera à regrouper les textes présentant des caractéristiques similaires.

Les groupes obtenus seront ensuite étudiés à partir de leurs mots les plus représentatifs et d'exemples d'avis.

L'objectif sera de déterminer si ces groupes correspondent à des thèmes interprétables et utiles pour l'entreprise.

Cette méthode sera évaluée de manière critique, notamment lorsque les groupes obtenus ne correspondent pas clairement à un thème unique.

### Étape 5 – Évaluation et interprétation

Nous réaliserons une analyse des performances et des erreurs des modèles.

Une attention particulière sera accordée :

- Aux avis neutres.
- Aux avis mixtes exprimant plusieurs sentiments.
- Aux contradictions éventuelles entre le texte et la note.
- À la pertinence des thèmes identifiés.

Conformément au cahier des charges, les deux membres du binôme annoteront indépendamment 100 avis.

L'accord entre les annotations sera mesuré avec le coefficient kappa de Cohen.

### Étape 6 – Restitution des résultats

Nous développerons un tableau de bord interactif avec Streamlit afin de présenter :

- La répartition des sentiments.
- Les principaux thèmes identifiés.
- Les facteurs d'insatisfaction.
- Les comparaisons entre restaurants.
- Les performances des modèles.
- L'analyse d'un nouvel avis client.

Une note de synthèse permettra de traduire les résultats en recommandations concrètes pour un décideur non technique.

## 6. Choix méthodologiques

Notre démarche privilégie un nombre limité de méthodes afin de pouvoir approfondir leur fonctionnement, comparer leurs résultats et justifier nos choix.

Nous souhaitons notamment :

- Conserver une baseline classique et interprétable.
- Étudier l'apport d'un modèle Transformer.
- Utiliser une approche de clustering accessible pour l'identification des thèmes.
- Accorder une place importante à l'évaluation et à l'interprétation des résultats.

Cette pipeline constitue une **proposition méthodologique à faire valider par l'enseignante**, notamment concernant l'utilisation d'un seul Transformer et le remplacement des méthodes de modélisation de sujets initialement proposées.

Elle pourra être adaptée en fonction des premiers résultats et des recommandations de l'enseignante.

## 7. Environnement et outils

Le projet sera principalement développé en Python sur **Google Colab**.

| Outil | Utilisation |
|---|---|
| Google Colab | Développement et exécution des notebooks |
| Python | Langage principal |
| pandas / NumPy | Manipulation des données |
| Matplotlib / Seaborn | Visualisation |
| scikit-learn | TF-IDF, régression logistique, K-means et évaluation |
| Hugging Face Transformers | Utilisation et ajustement de DistilBERT |
| Streamlit | Tableau de bord interactif |
| GitHub | Versionnement et collaboration |

L'utilisation de Google Colab permettra notamment d'accéder à des ressources de calcul adaptées à l'entraînement du Transformer, selon leur disponibilité.

## 8. Organisation du binôme

**Membres :**
- Silina Herzallah
- Mehdi El Gourche

### Fonctionnement initial

Nous avons choisi de travailler conjointement sur chaque étape du projet.

Cette organisation nous permettra de développer une compréhension commune des méthodes employées, de participer ensemble aux choix méthodologiques et de maîtriser l'ensemble du projet pour la soutenance.

Nous travaillerons donc en collaboration sur l'exploration des données, les prétraitements, la modélisation, l'évaluation et la restitution.

### Adaptation de l'organisation

Cette organisation pourra évoluer en fonction de l'avancement du projet et des contraintes rencontrées.

Si nécessaire, certaines tâches pourront être réparties afin d'optimiser notre temps de travail, tout en maintenant une compréhension globale du projet par les deux membres.

Nous utiliserons GitHub pour suivre nos contributions, partager nos développements et conserver l'historique des modifications.

## 9. Organisation prévisionnelle du dépôt

```text
McDonalds-reviews-nlp/
│
├── README.md
├── requirements.txt
├── CONTRIBUTIONS.md
├── .gitignore
│
├── configs/
│
├── data/
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_sentiment.ipynb
│   ├── 03_themes.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── cleaning/
│   ├── sentiment/
│   ├── topics/
│   └── evaluation/
│
├── results/
│   ├── experiences.csv
│   └── figures/
│
├── app/
│
└── docs/
```

Les notebooks seront développés sur Google Colab puis enregistrés sur GitHub.

Les données volumineuses et les modèles entraînés seront stockés séparément afin de conserver un dépôt léger et reproductible.

## 10. Planning prévisionnel

| Date | Objectif |
|---|---|
| 10 novembre 2026 | Cadrage, dépôt GitHub, exploration et première baseline |
| Novembre – décembre 2026 | Nettoyage, classification et premiers thèmes |
| 5 janvier 2027 | Point d'avancement |
| Janvier – février 2027 | Comparaison des modèles, évaluation et dashboard |
| 1er mars 2027 | Revue avant soutenance |
| 12 mars 2027 | Dépôt final prévisionnel, à confirmer |
| 15 mars 2027 | Soutenance |

## 11. Livrables attendus

- Code Python reproductible et documenté.
- Notebooks Google Colab.
- Rapport technique de 15 à 20 pages.
- Comparaison des modèles de classification.
- Analyse des thèmes et des mots-clés.
- Évaluation humaine des résultats.
- Tableau de bord interactif.
- Note de synthèse décisionnelle de 1 à 2 pages.
- Présentation finale de 6 à 10 diapositives.
- Vidéo de secours pour la démonstration.

## 12. État d'avancement – Séance 1

**Date : 10 novembre 2026**

### Décisions prises

- Sujet : analyse intelligente des avis clients McDonald's.
- Dataset : McDonald's Reviews.
- Langue principale des avis : anglais.
- Environnement de développement : Google Colab.
- Gestion du projet : GitHub.
- Organisation : travail collaboratif sur l'ensemble des étapes.
- Pipeline simplifiée proposée, en attente de validation.

### Prochaines étapes

1. Importer le dataset dans Google Colab.
2. Réaliser l'exploration descriptive.
3. Contrôler la qualité des données.
4. Construire les trois classes de sentiments.
5. Développer une première baseline TF-IDF + régression logistique.
6. Documenter les premiers résultats.

---

*Projet réalisé dans le cadre du Master 1 Machine Learning et Science des Données – Année universitaire 2026–2027.*
