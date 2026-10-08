# Données monde — Alertes Météo

Données lues par la carte interactive d'[app.alertes-meteo.com](https://app.alertes-meteo.com/carte-interactive), publiées sur la branche `data` par le workflow `update-monde.yml` (toutes les 30 minutes) :

| Fichier | Contenu | Source |
| --- | --- | --- |
| `satellite/index.json` + `satellite/*.jpg` | 10 images (toutes les 20 min, 3 dernières heures), Europe / Atlantique / Afrique du Nord, projection Web Mercator | EUMETSAT — Meteosat MTG, GeoColour RGB (EUMETView) |
| `satellite/monde.jpg` | Mosaïque mondiale la plus récente (65° S – 70° N) : GeoColor GOES-Ouest, GOES-Est et Meteosat ; infrarouge mis en forme « nuit » pour Meteosat Océan Indien et Himawari | NOAA / NASA GIBS, EUMETSAT, JMA |
| `maps/index.json` + `maps/values/*` | Pression au niveau de la mer et vent à 10 m, monde (80° S – 80° N), 0 à 120 h par pas de 3 h | NOAA — GFS 1° (NOMADS) |
| `cyclones.json` | Cyclones tropicaux actifs : trajectoire observée et prévue, catégories, cône d'incertitude | GDACS (ONU / Commission européenne) |
| `geo/monde.json` | Côtes et frontières | Natural Earth 50 m |

Les grilles `maps/values/*.hkv.gz` ont le même format que les dépôts régionaux (en-tête `CEV1`, largeur, hauteur, min, max, puis uint16 ; 65535 = manquant ; lignes régulières en projection Mercator, du nord au sud).

Chaque partie est indépendante : une source en panne n'empêche pas la mise à jour des autres, et les fichiers déjà publiés sont conservés.

Lancement manuel : onglet **Actions**, workflow « Données monde », **Run workflow** (case *force* pour régénérer le GFS).
