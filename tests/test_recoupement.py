"""Tests du recoupement de la tendance par le CEP et le GFS : python -m unittest discover -s tests (depuis la racine du dépôt)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import update_monde as um  # noqa: E402

STATION = ("66", "66136001", "Perpignan", 42.7372, 2.8728, 42)
# normales simplifiées : température moyenne = numéro du mois, pluie 31 mm par mois de 31 jours (1 mm par jour)
NORMALES = {m: {"mois": m, "tm_moy": float(m), "rr_moy": 31.0 if m in (1, 3, 5, 7, 8, 10, 12) else 30.0} for m in range(1, 13)}


def faux_modele(debut: datetime, heures: int, pas: int = 3, temperature: float = 10.0, pluie_par_pas: float = 1.0) -> dict:
    """Fichier départemental factice : un point à 142 m tout près de la station, un point à 60 km (hors du rayon de pluie)."""
    forecast = []
    for k in range(0, heures + 1, pas):
        t = debut + timedelta(hours=k)
        cumul = pluie_par_pas * k / pas
        forecast.append([t.strftime("%Y-%m-%dT%H:%M:%SZ"), [[temperature, cumul], [temperature + 5, cumul * 10]]])
    return {
        "columns": {"points": ["model_index", "latitude", "longitude", "altitude_m"], "values": ["temperature_c", "precipitation_total_mm"]},
        "points": [[0, 42.75, 2.875, 142.0], [1, 43.25, 2.875, 20.0]],
        "forecast": forecast,
    }


class NormaleDuJour(unittest.TestCase):
    def test_milieu_du_mois(self):
        tm, rr = um.normale_du_jour(NORMALES, date(2026, 10, 16))
        self.assertAlmostEqual(tm, 10.0)
        self.assertAlmostEqual(rr, 1.0)

    def test_interpolation_vers_le_mois_suivant(self):
        # 31 octobre : à mi-chemin (ou presque) entre le milieu d'octobre (10) et celui de novembre (11)
        tm, _ = um.normale_du_jour(NORMALES, date(2026, 10, 31))
        self.assertGreater(tm, 10.4)
        self.assertLess(tm, 10.6)

    def test_passage_d_annee(self):
        tm, _ = um.normale_du_jour(NORMALES, date(2026, 1, 1))
        self.assertGreater(tm, 1.0)
        self.assertLess(tm, 12.0)
        self.assertAlmostEqual(tm, 12 - (12 - 1) * 16 / 31, places=5)


class SemainesModele(unittest.TestCase):
    def test_semaine_complete_temperature_ramenee_a_la_station(self):
        m = faux_modele(datetime(2026, 10, 12, tzinfo=timezone.utc), 7 * 24)
        (s,) = um.semaines_modele(m, STATION, NORMALES)
        self.assertEqual((s["debut"], s["du"], s["au"], s["jours"]), ("2026-10-12", "2026-10-12", "2026-10-18", 7))
        self.assertAlmostEqual(s["t"], 10.7)  # 10 + 0,65 (point 100 m plus haut que la station)
        self.assertAlmostEqual(s["pluie"], 56.0)  # 8 pas de 1 mm par jour, le point lointain n'est pas compté
        self.assertAlmostEqual(s["pluie_normale"], 7.0)
        self.assertAlmostEqual(s["t_normale"], 10.0)  # semaine centrée sur le milieu d'octobre

    def test_jour_incomplet_au_debut_du_run_et_semaine_trop_courte(self):
        # run de 12 UTC le dimanche 11 : le dimanche n'a pas ses 4 températures ; la semaine suivante s'arrête au mercredi 21 à 12 UTC
        m = faux_modele(datetime(2026, 10, 11, 12, tzinfo=timezone.utc), 10 * 24)
        semaines = um.semaines_modele(m, STATION, NORMALES)
        self.assertEqual([s["debut"] for s in semaines], ["2026-10-12"])
        self.assertEqual(semaines[0]["jours"], 7)

    def test_semaine_partielle_d_au_moins_3_jours(self):
        m = faux_modele(datetime(2026, 10, 15, tzinfo=timezone.utc), 4 * 24, pas=6)
        (s,) = um.semaines_modele(m, STATION, NORMALES)
        self.assertEqual((s["du"], s["au"], s["jours"]), ("2026-10-15", "2026-10-18", 4))
        self.assertAlmostEqual(s["pluie"], 16.0)  # pas de 6 h : 4 pas de 1 mm par jour


class Recoupement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.runs = {"cep": "2026-10-11T12:00:00Z", "gfs": "2026-10-12T00:00:00Z"}
        self.gfs_en_panne = False
        self.appels: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def faux_get(self, url: str, **_):
        self.appels.append(url)
        r = mock.Mock()
        if url.endswith("normales.json"):
            r.json.return_value = {"months": list(NORMALES.values())}
        elif url.endswith("index.json"):
            cle = "cep" if "/cep/" in url else "gfs"
            if cle == "gfs" and self.gfs_en_panne:
                raise RuntimeError("HTTP 503")
            r.json.return_value = {"model": {"run_time": self.runs[cle]}}
        else:
            if "/gfs/" in url and self.gfs_en_panne:
                raise RuntimeError("HTTP 503")
            r.json.return_value = faux_modele(datetime(2026, 10, 12, tzinfo=timezone.utc), 14 * 24, temperature=12.0 if "/cep/" in url else 14.0)
        return r

    def lancer(self) -> dict:
        with mock.patch.object(um, "get", side_effect=self.faux_get), mock.patch.object(um, "log"):
            um.recoupement(self.out)
        return json.loads((self.out / "tendance-modeles.json").read_text(encoding="utf-8"))

    def test_publie_les_deux_modeles_pour_chaque_zone(self):
        doc = self.lancer()
        self.assertEqual(doc["modeles"]["cep"]["run"], self.runs["cep"])
        self.assertEqual(len(doc["zones"]), len(um.RECOUPEMENT_STATIONS))
        z = doc["zones"][0]
        self.assertEqual(z["station"]["nom"], "Perpignan")
        self.assertEqual([s["debut"] for s in z["modeles"]["cep"]], ["2026-10-12", "2026-10-19"])
        self.assertAlmostEqual(z["modeles"]["gfs"][0]["t"], 14.7)

    def test_rien_n_est_relu_si_les_runs_n_ont_pas_change(self):
        self.lancer()
        self.appels.clear()
        self.lancer()
        self.assertTrue(all(u.endswith("index.json") for u in self.appels), self.appels)

    def test_un_modele_en_panne_garde_ses_valeurs_precedentes(self):
        avant = self.lancer()
        self.gfs_en_panne = True
        self.runs["cep"] = "2026-10-12T00:00:00Z"
        apres = self.lancer()
        self.assertEqual(apres["modeles"]["cep"]["run"], "2026-10-12T00:00:00Z")
        self.assertEqual(apres["modeles"]["gfs"], avant["modeles"]["gfs"])
        self.assertEqual(apres["zones"][3]["modeles"]["gfs"], avant["zones"][3]["modeles"]["gfs"])


if __name__ == "__main__":
    unittest.main()
