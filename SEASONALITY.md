# Saisonnalité des ingrédients Mealie

Initialiser puis compléter le mapping avec `make seasonality`. Cette commande actualise le catalogue Mealie et soumet **une seule fois par nouvel ingrédient** les ingrédients et lignes inconnus à l'IA. Elle ne s'exécute ni au démarrage du serveur ni lors de la composition d'une semaine.

Le résultat est enregistré dans `data/seasonality.json`, conservé par le volume Docker et modifiable à la main. Relancer la commande ajoute uniquement les nouvelles entrées ; les modifications existantes sont préservées. Pour utiliser le cache de recettes déjà présent sans contacter Mealie, lancer `python3 seasonality.py --no-refresh` dans un environnement où `OPENAI_API_KEY` est défini.

Chaque entrée de `foods` est indexée par l'identifiant Mealie et contient `name`, `neutral` et `months`. Les clés de `months` vont de `"01"` à `"12"` : `0` signifie hors saison locale, `1` disponible localement et `2` pleine saison. Pour un ingrédient sans saison pertinente, mettre `neutral: true` et `months: {}`. Les entrées de `lines` relient les textes libres à un `food_id` ; `food_id: null` et `neutral: false` signifient que la ligne reste à rapprocher manuellement.

Exemple :

```json
{
  "version": 1,
  "region": "Sud-Ouest de la France",
  "foods": {
    "identifiant-mealie": {
      "name": "courge",
      "neutral": false,
      "months": {"01": 1, "02": 0, "03": 0, "04": 0, "05": 0, "06": 0, "07": 0, "08": 1, "09": 2, "10": 2, "11": 2, "12": 2}
    }
  },
  "lines": {
    "courge": {"display": "600 g courge", "food_id": "identifiant-mealie", "neutral": false}
  }
}
```

La sélection des recettes et leur classement utilisent ce fichier sans appel IA. Le bouton « Compléter avec l’IA » est le seul appel IA du parcours hebdomadaire.

Dans le choix des recettes, le score global est représenté par les images `img/A.png` à `img/E.png` : A pour un score d'au moins 90 %, B d'au moins 80 %, C d'au moins 70 %, D d'au moins 50 %, E en dessous de 50 %. Les ingrédients non évalués comptent dans le dénominateur du score ; le nombre d'ingrédients évalués est indiqué à côté de l'image. Le détail affiche pour chaque ingrédient son score mensuel 0/2, 1/2 ou 2/2, ou son état neutre ou non évalué.

Le catalogue `/mealie/catalog` présente toutes les recettes pour le mois choisi, y compris celles sans score. Dans la composition d'une semaine, seules les recettes Mealie notées A, B ou C pour le jour concerné sont suggérées ; un glissement vers la gauche (ou la flèche gauche au clavier) affiche la suggestion suivante. Un geste ramené vers la droite avant de relâcher le doigt annule le changement.
