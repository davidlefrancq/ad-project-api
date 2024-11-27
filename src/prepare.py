import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(__file__)

# Dictionnaire pour renommer les colonnes
column_names = {
  'carmodel': 'Modèle',
  'price': 'Prix',
  'année': 'Année',
  'kilométragecompteur': 'Kilométrage',
  'énergie': 'Carburant',
  'boîtedevitesse': 'Boîte',
  'couleurextérieure': 'Couleur',
  'nombredeportes': 'Nb portes',
  'premièremain(déclaratif)': '1ère main',
  'puissancefiscale': 'Puissance',
  'couleurextérieure_métallisée': 'Métallisée',
}

energy_list = [
  'gpl',
  'hybride',
  'hydrogène',
  'bioéthanol',
  'electrique',
  'essence',
  'diesel',
]

energy_dict = {
  'gpl': 'gpl',
  'hybride': 'hybride',
  'bioéthanol': 'bioéthanol',
  'bioethanol': 'bioéthanol',
  'hydrogène': 'hydrogène',
  'hydrogene': 'hydrogène',
  'electrique': 'electrique',
  'électrique': 'electrique',
  'essence': 'essence',
  'éssence': 'essence',
  'diesel': 'diesel',
  'diezel': 'diesel',
}

color_list = [
  'noir',
  'blanc',
  'gris',
  'bleu',
  'rouge',
  'vert',
  'jaune',
  'marron',
  'orange',
  'violet',
]

colors_dict = {
  # Noir
  'noir': 'noir',
  'black': 'noir',
  'schwarz': 'noir',
  'saphirschwarz': 'noir',
  'carbonschwarz': 'noir',
  'magnetic tech': 'noir',
  'schwartz': 'noir',
  'saphirschwar': 'noir',
  'anthracite': 'noir',
  'canna di fucile': 'noir',
  
  # Blanc
  'blanc': 'blanc',
  'white': 'blanc',
  'weiss': 'blanc',
  'alpinweiss': 'blanc',
  'craie': 'blanc',
  'blc banquise': 'blanc',
  'bela': 'blanc',
  'ivoire': 'blanc',
  'autre': 'blanc',
  'ibiswei': 'blanc',
  
  # Gris
  'gris': 'gris',
  'grey': 'gris',
  'gray': 'gris',
  'grau': 'gris',
  'argent': 'gris',
  'silver': 'gris',
  'platinium': 'gris',
  'platine': 'gris',
  'titanium': 'gris',
  'glaciersilber': 'gris',
  'storm bay': 'gris',
  'inc.': 'gris',
  'inconn': 'gris',
  'clair': 'gris',
  'inconnu': 'gris',
  'inconu': 'gris',
  '31490': 'gris',
  'tissus': 'gris',
  
  # Bleu
  'bleu': 'bleu',
  'blue': 'bleu',
  'azur': 'bleu',
  'mediterraneanblau': 'bleu',
  'plava': 'bleu',
  
  # Rouge
  'rouge': 'rouge',
  'red': 'rouge',
  'rot': 'rouge',
  'bordeaux': 'rouge',
  
  # Vert
  'vert': 'vert',
  'green': 'vert',
  'emeraude chrystal': 'vert',
  'emerald': 'vert',
  'hyhgland': 'vert',
  
  # Jaune
  'jaune': 'jaune',
  'yellow': 'jaune',
  'rauchtopas': 'jaune',
  'or': 'jaune',
  'sable': 'jaune',
  
  # Marron/Brun
  'marron': 'marron',
  'brun': 'marron',
  'brown': 'marron',
  'beige': 'marron',
  'platinsilber': 'marron',
  'cooper': 'marron',
  
  # Orange
  'orange': 'orange',
  
  # Violet
  'violet': 'violet',
  'purple': 'violet',
  'byzantin': 'violet',
}

color_metal_dict = {
  'metal': 'métallisée',
  'métallisé': 'métallisée',
  'métallisee': 'métallisée'
}

class FrenchSecondHandCarsDataPreparator:
  def __init__(self, data_path: str):
    path_file = os.path.join(script_dir, data_path)
    self.data_path = path_file
    self.original_data = None
    self.data = None
    self.x = None
    self.y = None
    self.x_train = None
    self.x_test = None
    self.y_train = None
    self.y_test = None
    self.labelEncoderCarmodel = LabelEncoder()
    self.labelEncoderEnergie = LabelEncoder()
    self.labelEncoderBoiteVitesse = LabelEncoder()
    self.labelEncoderPuissance = LabelEncoder()
    self.labelEncoderCouleur = LabelEncoder()
    self.standardScaler = StandardScaler()
    self._load_data()

  def _load_data(self):
    #check if the file exists
    if not os.path.exists(self.data_path):
      raise FileNotFoundError(f"File not found: {self.data_path}")
    self.original_data = pd.read_csv(self.data_path)
    self.original_data.info()

  def color_detection(self, color: str):
    color = color.lower()
    # TODO: if color from dict is contained in the color, set color to the value of the dict
    for key, value in colors_dict.items():
      if key in color:
        color = value
        break
    if color not in color_list:
      color = 'gris'
    return color

  def energy_detection(self, energy: str):
    energy = energy.lower()
    for key, value in energy_dict.items():
      if key in energy:
        energy = value
        break
    if energy not in energy_list:
      energy = None
    return energy

  def encode(self):
    # Colomn list: publishedsince,carmodel,price,année,miseencirculation,contrôletechnique,kilométragecompteur,énergie,boîtedevitesse,couleurextérieure,nombredeportes,nombredeplaces,garantie,premièremain(déclaratif),nombredepropriétaires,puissancefiscale,puissancedin,crit'air,émissionsdeco2,consommationmixte,normeeuro,options,departement,id,waranty,vendeur,vérifié&garanti,rechargeable,autonomiebatterie,capacitébatterie,conso.batterie,couleurintérieure,puissancemoteur,primeàlaconversion,garantieconstructeur,provenance,prixinclutlabatterie,voltagebatterie,intensitébatterie,prixinclutlabatterie
    #     Column                    Non-Null Count  Dtype
    #---  ------                    --------------  -----
    # 0   publishedsince            2441 non-null   object
    # 1   carmodel                  2441 non-null   object
    # 2   price                     2441 non-null   object
    # 3   ann▒e                     2440 non-null   float64
    # 4   miseencirculation         2440 non-null   object
    # 5   contr▒letechnique         2440 non-null   object
    # 6   kilom▒tragecompteur       2440 non-null   object
    # 7   ▒nergie                   2440 non-null   object
    # 8   bo▒tedevitesse            2440 non-null   object
    # 9   couleurext▒rieure         2440 non-null   object
    # 10  nombredeportes            2436 non-null   float64
    # 11  nombredeplaces            2362 non-null   float64
    # 12  garantie                  2062 non-null   object
    # 13  premi▒remain(d▒claratif)  2440 non-null   object
    # 14  nombredepropri▒taires     970 non-null    float64
    # 15  puissancefiscale          2434 non-null   object
    # 16  puissancedin              2370 non-null   object
    # 17  crit'air                  2363 non-null   float64
    # 18  ▒missionsdeco2            2234 non-null   object
    # 19  consommationmixte         2124 non-null   object
    # 20  normeeuro                 2365 non-null   object
    # 21  options                   2441 non-null   object
    # 22  departement               2441 non-null   int64
    # 23  id                        2441 non-null   int64
    # 24  waranty                   2252 non-null   object
    # 25  vendeur                   2441 non-null   object
    # 26  v▒rifi▒&garanti           431 non-null    object
    # 27  rechargeable              185 non-null    object
    # 28  autonomiebatterie         101 non-null    object
    # 29  capacit▒batterie          157 non-null    object
    # 30  conso.batterie            66 non-null     object
    # 31  couleurint▒rieure         1144 non-null   object
    # 32  puissancemoteur           418 non-null    object
    # 33  prime▒laconversion        410 non-null    object
    # 34  garantieconstructeur      695 non-null    object
    # 35  provenance                118 non-null    object
    # 36  prixinclutlabatterie      141 non-null    object
    # 37  voltagebatterie           111 non-null    object
    # 38  intensit▒batterie         59 non-null     object
    # 39  prixinclutlabatterie.1    3 non-null      object
    
    # Copy the original data but keep only the columns we need
    self.data = self.original_data[
      [
        'carmodel',
        'price',
        'année',
        'kilométragecompteur',
        'énergie',
        'boîtedevitesse',
        'couleurextérieure',
        'nombredeportes',
        'premièremain(déclaratif)',
        'puissancefiscale',
      ]
    ].copy()
    self.data.info()
    
    # Clean "price" column, split on €, remove blank space and convert to float
    self.data['price'] = self.data['price'].str.split('€').str[0].str.replace(' ', '').astype(float)
    
    # if "année" is empty, remove the row
    self.data = self.data.dropna(subset=['année'])
    
    # "année" is int
    self.data['année'] = self.data['année'].astype(int)
    
    # Remove rows for 'année' < 2000
    self.data = self.data[self.data['année'] >= 2000]
    
    # Clear "puissancefiscale" column, set to lower case, set blank space to 'mécanique'; remove rows for 'puissancefiscale' is empty
    self.data['puissancefiscale'] = self.data['puissancefiscale'].str.strip()
    self.data['puissancefiscale'] = self.data['puissancefiscale'].str.lower()
    self.data = self.data.dropna(subset=['puissancefiscale'])
    # Standardize "puissancefiscale" column, remove all characters and convert to int
    self.data['puissancefiscale'] = self.data['puissancefiscale'].str.replace(r'\D', '', regex=True).astype(int)
    
    # Clean "kilométragecompteur" column, split on km, remove blank space, use regex to remove all characters and convert to float
    self.data = self.data.dropna(subset=['kilométragecompteur'])
    self.data['kilométragecompteur'] = self.data['kilométragecompteur'].str.split('km').str[0].str.replace(' ', '')
    self.data['kilométragecompteur'] = self.data['kilométragecompteur'].str.replace(r'\D', '', regex=True).astype(int)
    # self.data['kilométragecompteur'] = self.data['kilométragecompteur'].str.split('km').str[0].str.replace(' ', '').astype(float)

    # "nombredeportes" is int, remove the row if empty
    self.data = self.data.dropna(subset=['nombredeportes'])
    self.data['nombredeportes'] = self.data['nombredeportes'].astype(int)
    
    # Clean "boîtedevitesse" column, set to lower case, set blank space to 'mécanique'; set NaN to 'mécanique'
    self.data['boîtedevitesse'] = self.data['boîtedevitesse'].str.strip()
    self.data['boîtedevitesse'] = self.data['boîtedevitesse'].str.lower()
    self.data['boîtedevitesse'] = self.data['boîtedevitesse'].fillna('mécanique')
    self.data['boîtedevitesse'] = self.data['boîtedevitesse'].apply(lambda x: 'mécanique' if len(x) < 3 else x)
    self.data['boîtedevitesse'] = self.labelEncoderBoiteVitesse.fit_transform(self.data['boîtedevitesse'])
    
    # Remove rows for 'énergie' is empty
    self.data = self.data.dropna(subset=['énergie'])
    # Remove rows for 'énergie' length < 3
    self.data = self.data[self.data['énergie'].str.len() >= 3]
    
    # Clean "carmodel" column, strip() and lower(), remove rows for 'carmodel' is empty
    self.data['carmodel'] = self.data['carmodel'].str.strip().str.lower()
    self.data = self.data.dropna(subset=['carmodel'])
    self.data['carmodel'] = self.labelEncoderCarmodel.fit_transform(self.data['carmodel'])
    
    # Clean "énérige" column, set to lower case, set blank space to 'essence'; set NaN to 'essence'
    self.data['énergie'] = self.data['énergie'].str.strip()
    self.data['énergie'] = self.data['énergie'].str.lower()
    self.data['énergie'] = self.data['énergie'].fillna('essence')
    # Standardize "énergie" column
    for index, row in self.data.iterrows():
      energy = self.energy_detection(row['énergie'])
      self.data.at[index, 'énergie'] = energy
    self.data['énergie'] = self.labelEncoderEnergie.fit_transform(self.data['énergie'])
        
    # Add metalisée boolean column if "couleurextérieure" contains "metal" "métal" or "métallisé" or "métallisee"
    self.data['couleurextérieure_métallisée'] = self.data['couleurextérieure'].apply(lambda x: 'metal' in x or 'métal' in x or 'métallisé' in x or 'métallisee' in x)
    
    # Standardize "couleurextérieure" column
    for index, row in self.data.iterrows():
      color = self.color_detection(row['couleurextérieure'])
      self.data.at[index, 'couleurextérieure'] = color
    self.data['couleurextérieure'] = self.labelEncoderCouleur.fit_transform(self.data['couleurextérieure'])      

    # "premièremain(déclaratif)" convert to bool if == 'oui'
    # self.data['premièremain(déclaratif)'] = self.data['premièremain(déclaratif)'].apply(lambda x: x == 'oui')
    self.data['premièremain(déclaratif)'] = self.data['premièremain(déclaratif)'].apply(lambda x: 1 if 'oui' in str(x).lower() else 0)
    
    self.data.reset_index(drop=True, inplace=True)

    self.data.info()
    
  def remobe_outliers(self):
    # Remove kilométragecompteur outliers
    Q1 = self.data['kilométragecompteur'].quantile(0.25)
    Q3 = self.data['kilométragecompteur'].quantile(0.75)
    IQR = Q3 - Q1
    self.data = self.data[(self.data['kilométragecompteur'] >= (Q1 - 1.5 * IQR)) & (self.data['kilométragecompteur'] <= (Q3 + 1.5 * IQR))]
    
    # Remove price outliers
    Q1 = self.data['price'].quantile(0.25)
    Q3 = self.data['price'].quantile(0.75)
    IQR = Q3 - Q1
    self.data = self.data[(self.data['price'] >= (Q1 - 1.5 * IQR)) & (self.data['price'] <= (Q3 + 1.5 * IQR))]
    
  def visualizations(self):
    # Matrice de corelation
    corr_matrix = self.data.corr()
    
    # Renommer les index et colonnes de la matrice
    corr_matrix.index = [column_names.get(col, col) for col in corr_matrix.index]
    corr_matrix.columns = [column_names.get(col, col) for col in corr_matrix.columns]
    
    # Visualize matrice de corrélation
    plt.figure(figsize=(16, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Matrice de corrélation')
    correlation_graph_path = os.path.join(script_dir, 'graph/correlation.png')
    plt.savefig(correlation_graph_path)

    # Visualize the couleurextérieure distribution whith couleurextérieure_métallisée and the price and use self.labelEncoder.inverse_transform to get the original value
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.labelEncoderCouleur.inverse_transform(self.data['couleurextérieure']), y=self.data['price'], hue=self.data['couleurextérieure_métallisée'])
    plt.xticks(rotation=90)
    plt.title('Distribution of the target variable: couleurextérieure')
    colors_graph_path = os.path.join(script_dir, 'graph/couleurs.png')
    plt.savefig(colors_graph_path)

    # Visualize the année distribution
    plt.figure(figsize=(12, 6))
    hist_annee = sns.histplot(data=self.data['année'], kde=True, bins=50, stat="count")
    # Récupérer les coordonnées des barres
    for patch in hist_annee.patches:
      # Obtenir les coordonnées x,y pour placer le texte
      x = patch.get_x() + patch.get_width()/2
      y = patch.get_height()      
      # Ajouter le texte avec la valeur
      if int(y) > 0: hist_annee.text(x, y, int(y), ha='center', va='bottom')
    plt.title('Distribution of the target variable: année')
    annee_graph_path = os.path.join(script_dir, 'graph/annee.png')
    plt.savefig(annee_graph_path)

    # Visualize the anneé distribution whith price
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['année'], y=self.data['price'])
    plt.title('Distribution of the target variable: année')
    annee_price_graph_path = os.path.join(script_dir, 'graph/annee_price.png')
    plt.savefig(annee_price_graph_path)

    # Visualize the année with price and puissancefiscale
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['année'], y=self.data['price'], hue=self.data['puissancefiscale'])
    plt.title('Distribution of the target variable: année')
    annee_price_puissance_graph_path = os.path.join(script_dir, 'graph/annee_price_puissance.png')
    plt.savefig(annee_price_puissance_graph_path)

    # Visualize the kilométragecompteur distribution
    plt.figure(figsize=(12, 6))
    hist_kilometrage = sns.histplot(self.data['kilométragecompteur'], kde=True, bins=50, stat="count", shrink=0.8)
    # Récupérer les coordonnées des barres
    for patch in hist_kilometrage.patches:
      # Obtenir les coordonnées x,y pour placer le texte
      x = patch.get_x() + patch.get_width()/2
      y = patch.get_height()      
      # Ajouter le texte avec la valeur
      if int(y) > 0: hist_kilometrage.text(x, y, f" {y}", ha='center', va='bottom', rotation=90)
    plt.title('Distribution of the target variable: kilométragecompteur')
    kilométragecompteur_graph_path = os.path.join(script_dir, 'graph/kilometrage.png')
    plt.savefig(kilométragecompteur_graph_path)

    # Visualize the kilométragecompteur with price
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['kilométragecompteur'], y=self.data['price'])
    plt.title('Distribution of the target variable: kilométragecompteur')
    kilométragecompteur_price_graph_path = os.path.join(script_dir, 'graph/kilometrage_price.png')
    plt.savefig(kilométragecompteur_price_graph_path)

    # Visualize the kilometrage with année
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['kilométragecompteur'], y=self.data['année'])
    plt.title('Distribution of the target variable: kilométragecompteur')
    kilométragecompteur_annee_graph_path = os.path.join(script_dir, 'graph/kilometrage_annee.png')
    plt.savefig(kilométragecompteur_annee_graph_path)

    # Visualize the année with kilométragecompteur
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['année'], y=self.data['kilométragecompteur'])
    plt.title('Distribution of the target variable: kilométragecompteur')
    annee_kilometrage_graph_path = os.path.join(script_dir, 'graph/annee_kilometrage.png')
    plt.savefig(annee_kilometrage_graph_path)

    # Visualize the année with kilométragecompteur and puissancefiscale
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['année'], y=self.data['kilométragecompteur'], hue=self.data['puissancefiscale'])
    plt.title('Distribution: année, kilométragecompteur and puissancefiscale')
    annee_kilometrage_puissance_graph_path = os.path.join(script_dir, 'graph/annee_kilometrage_puissance.png')
    plt.savefig(annee_kilometrage_puissance_graph_path)

    # Visualize the kilométragecompteur with price and année
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['kilométragecompteur'], y=self.data['price'], hue=self.data['année'])
    plt.title('Distribution of the target variable: kilométragecompteur')
    kilométragecompteur_price_annee_graph_path = os.path.join(script_dir, 'graph/kilometrage_price_annee.png')
    plt.savefig(kilométragecompteur_price_annee_graph_path)

    # Visualize the année with price and kilométragecompteur
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=self.data['année'], y=self.data['price'], hue=self.data['kilométragecompteur'])
    plt.title('Distribution: année, price and kilométragecompteur')
    annee_price_kilometrage_graph_path = os.path.join(script_dir, 'graph/annee_price_kilometrage.png')
    plt.savefig(annee_price_kilometrage_graph_path)
    
    # Visualize the price with boîtedevitesse
    unique_encoded = sorted(self.data['boîtedevitesse'].unique())
    original_labels = self.labelEncoderBoiteVitesse.inverse_transform(unique_encoded)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=self.data, x='boîtedevitesse', y='price')
    sns.stripplot(data=self.data, x='boîtedevitesse', y='price', color='blue', alpha=0.15, jitter=0.2)
    plt.xticks(unique_encoded, original_labels)
    # Ajouter les moyennes
    means = self.data.groupby('boîtedevitesse')['price'].mean()
    for i, mean in enumerate(means):
      plt.text(i, mean - 1000, f'Moyenne:\n{mean:,.0f}€', ha='center', va='bottom', color='white')
    
    boîtedevitesse_price_graph_path = os.path.join(script_dir, 'graph/boite_price.png')
    plt.savefig(boîtedevitesse_price_graph_path)

    # Visualize the énergie distribution
    unique_encoded = sorted(self.data['énergie'].unique())
    original_labels = self.labelEncoderEnergie.inverse_transform(unique_encoded)
    plt.figure(figsize=(12, 12))
    hist_energie = sns.histplot(self.data['énergie'], kde=True, bins=50)
    for patch in hist_energie.patches:
      # Obtenir les coordonnées x,y pour placer le texte
      x = patch.get_x() + patch.get_width()/2
      y = patch.get_height()      
      # Ajouter le texte avec la valeur
      if int(y) > 0: hist_energie.text(x, y, int(y), ha='center', va='bottom')
    plt.xticks(rotation=90)
    plt.xticks(unique_encoded, original_labels)
    plt.title('Distribution of the target variable: énergie')
    energie_graph_path = os.path.join(script_dir, 'graph/energie.png')
    plt.savefig(energie_graph_path)

    # Visualize the price with énergie
    unique_encoded = sorted(self.data['énergie'].unique())
    original_labels = self.labelEncoderEnergie.inverse_transform(unique_encoded)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=self.data, x='énergie', y='price')
    sns.stripplot(data=self.data, x='énergie', y='price', color='blue', alpha=0.15, jitter=0.2)
    plt.xticks(ticks=range(len(unique_encoded)), labels=original_labels, ha='center')
    # plt.xticks(rotation=90)
    # Ajouter les moyennes
    means = self.data.groupby('énergie')['price'].mean()
    for i, mean in enumerate(means):
      plt.text(i, mean - 1000, f'Moyenne:\n{mean:,.0f}€', ha='center', va='bottom', color='white')
    
    energie_price_graph_path = os.path.join(script_dir, 'graph/energie_price.png')
    plt.savefig(energie_price_graph_path)
       
    # Visualize the nombredeportes with price
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=self.data, x='nombredeportes', y='price')
    sns.stripplot(data=self.data, x='nombredeportes', y='price', color='blue', alpha=0.15, jitter=0.2)
    # Ajouter les moyennes
    means = self.data.groupby('nombredeportes')['price'].mean()
    for i, mean in enumerate(means):
      plt.text(i, mean - 1000, f'Moyenne:\n{mean:,.0f}€', ha='center', va='bottom', color='white')

    nombredeportes_price_graph_path = os.path.join(script_dir, 'graph/nombredeportes_price.png')
    plt.savefig(nombredeportes_price_graph_path)

    # Visualize the premièremain(déclaratif) with price    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=self.data, x='premièremain(déclaratif)', y='price')
    sns.stripplot(data=self.data, x='premièremain(déclaratif)', y='price', color='blue', alpha=0.15, jitter=0.2)
    # Remplacer les labels 0 et 1 par 'non' et 'oui'
    plt.xticks([0, 1], ['Non', 'Oui'])
    
    # Ajouter les moyennes
    means = self.data.groupby('premièremain(déclaratif)')['price'].mean()
    for i, mean in enumerate(means):
      plt.text(i, mean - 1000, f'Moyenne:\n{mean:,.0f}€', ha='center', va='bottom', color='white')
      
    premièremain_price_graph_path = os.path.join(script_dir, 'graph/premiere_main_price.png')
    plt.savefig(premièremain_price_graph_path)

  def _split(self):
    self.x = self.data.drop(columns=['price'])
    self.y = self.data['price']
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
      self.x,
      self.y,
      test_size=0.2,
      random_state=42
    )

  def save(self):
    # LabelEncoder
    ds_encoder_carmodel_path = os.path.join(script_dir, 'model/dataset_encoder_carmodel.pkl')
    with open(ds_encoder_carmodel_path, 'wb') as fichier:
      pickle.dump(self.labelEncoderCarmodel, fichier)
    ds_encoder_energie_path = os.path.join(script_dir, 'model/dataset_encoder_energie.pkl')
    with open(ds_encoder_energie_path, 'wb') as fichier:
      pickle.dump(self.labelEncoderEnergie, fichier)
    ds_encoder_boite_path = os.path.join(script_dir, 'model/dataset_encoder_boite.pkl')
    with open(ds_encoder_boite_path, 'wb') as fichier:
      pickle.dump(self.labelEncoderBoiteVitesse, fichier)
    ds_encoder_puissance_path = os.path.join(script_dir, 'model/dataset_encoder_puissance.pkl')
    with open(ds_encoder_puissance_path, 'wb') as fichier:
      pickle.dump(self.labelEncoderPuissance, fichier)
    ds_encoder_couleur_path = os.path.join(script_dir, 'model/dataset_encoder_couleur.pkl')
    with open(ds_encoder_couleur_path, 'wb') as fichier:
      pickle.dump(self.labelEncoderCouleur, fichier)
    
    # Save train and test data
    self._split()
    
    x_train_path = os.path.join(script_dir, 'data/x_train.npy')
    x_test_path = os.path.join(script_dir, 'data/x_test.npy')
    y_train_path = os.path.join(script_dir, 'data/y_train.npy')
    y_test_path = os.path.join(script_dir, 'data/y_test.npy')
  
    np.save(x_train_path, self.x_train)
    np.save(x_test_path, self.x_test)
    np.save(y_train_path, self.y_train)
    np.save(y_test_path, self.y_test)
        
if __name__ == '__main__':
  data_preparator = FrenchSecondHandCarsDataPreparator(data_path='data/dataset.csv')
  data_preparator.encode()
  data_preparator.remobe_outliers()
  data_preparator.visualizations()
  data_preparator.save()