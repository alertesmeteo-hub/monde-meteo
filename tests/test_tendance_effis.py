"""Tests de la tendance à 5 semaines (cartes EFFIS) : python -m unittest discover -s tests (depuis la racine du dépôt)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import update_monde as um  # noqa: E402

SOMBRE = (64, 64, 64)


def fausse_carte(variable: str, regions: dict[tuple[float, float], int] | None = None, sans_meridien: int | None = None, couleur_legende: dict[int, tuple] | None = None) -> Image.Image:
    """Carte EFFIS factice pleine taille : fond blanc, cadre, quadrillage pointillé, pastilles de légende, et des carrés de classe autour de points
    (lon, lat) -> indice de classe."""
    img = Image.new("RGB", um.EFFIS_TAILLE, (255, 255, 255))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = um.EFFIS_CADRE
    d.rectangle((x0 - 10, y0 - 10, x1 + 10, y0 - 1), fill=SOMBRE)
    d.rectangle((x0 - 10, y1 + 1, x1 + 10, y1 + 10), fill=SOMBRE)
    d.rectangle((x0 - 10, y0 - 10, x0 - 1, y1 + 10), fill=SOMBRE)
    d.rectangle((x1 + 1, y0 - 10, x1 + 10, y1 + 10), fill=SOMBRE)
    for lon in range(-25, 56, 10):
        if lon == sans_meridien:
            continue
        x = um.effis_pixel(lon, 50.0)[0]
        for y in range(y0, y1, 20):
            d.rectangle((x - 2, y, x + 2, y + 4), fill=SOMBRE)
    for lat in range(25, 76, 10):
        y = um.effis_pixel(0.0, lat)[1]
        for x in range(x0, x1, 20):
            d.rectangle((x, y - 2, x + 4, y + 2), fill=SOMBRE)
    couleurs = um.EFFIS_CLASSES[variable]["couleurs"]
    pastilles = couleurs[:8] + couleurs[9:]
    for i, x in enumerate(um.EFFIS_LEGENDE_X):
        c = (couleur_legende or {}).get(i, pastilles[i])
        d.rectangle((x - 100, um.EFFIS_LEGENDE_Y - 50, x + 100, um.EFFIS_LEGENDE_Y + 50), fill=c)
    for (lon, lat), k in (regions or {}).items():
        x, y = um.effis_pixel(lon, lat)
        d.rectangle((x - 60, y - 60, x + 60, y + 60), fill=couleurs[k])
    return img


def png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", compress_level=1)
    return b.getvalue()


class Reponse:
    def __init__(self, contenu: bytes | str):
        self.content = contenu if isinstance(contenu, bytes) else contenu.encode("utf-8")
        self.text = contenu if isinstance(contenu, str) else contenu.decode("utf-8", "replace")


class Calage(unittest.TestCase):
    def test_pixels_du_quadrillage(self):
        self.assertEqual(um.effis_pixel(-25, 75), (662, 1047))
        self.assertEqual(um.effis_pixel(55, 25), (6352, 4602))
        x, y = um.effis_pixel(5, 45)
        self.assertAlmostEqual(x, 2797, delta=3)
        self.assertAlmostEqual(y, 3182, delta=3)

    def test_index(self):
        txt = "Europe_MonthlyAnomalies_T2m_20261005_w3.png\nEurope_MonthlyAnomalies_T2m_20261005_w2.png\n\nautre.png\nEurope_MonthlyAnomalies_Rain_20261005_w2.png\n"
        self.assertEqual(
            um.effis_noms(txt, "T2m"),
            [(2, "Europe_MonthlyAnomalies_T2m_20261005_w2.png", date(2026, 10, 5)), (3, "Europe_MonthlyAnomalies_T2m_20261005_w3.png", date(2026, 10, 5))],
        )
        self.assertEqual(len(um.effis_noms(txt, "Rain")), 1)
        self.assertEqual(um.effis_noms("", "T2m"), [])

    def test_legende(self):
        for v, n_bornes in (("t2m", 18), ("pluie", 18)):
            d = um.EFFIS_CLASSES[v]
            self.assertEqual(len(d["bornes"]), n_bornes)
            self.assertEqual(len(d["couleurs"]), n_bornes - 1)
            self.assertEqual(d["bornes"], sorted(d["bornes"]))
            self.assertEqual(d["couleurs"][8], (255, 255, 255))  # classe du milieu : proche de la normale
            l = um.effis_legende(v)
            self.assertEqual(len(l["classes"]), 17)
            self.assertEqual((l["classes"][8]["min"], l["classes"][8]["max"]), (-0.5, 0.5) if v == "t2m" else (-5, 5))
            self.assertEqual(l["unite"], "°C" if v == "t2m" else "mm")
        self.assertEqual(um.effis_legende("t2m")["classes"][16], {"min": 7, "max": 8})
        self.assertEqual(um.effis_legende("pluie")["classes"][0], {"min": -100, "max": -80})
        # chaque classe est entre ses deux bornes, sans trou
        for v in ("t2m", "pluie"):
            c = um.effis_legende(v)["classes"]
            self.assertTrue(all(a["max"] == b["min"] for a, b in zip(c, c[1:])))


class Lecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zones = {(2.9, 42.7): 10, (1.44, 43.6): 3, (3.88, 43.61): 8}
        cls.t2m = np.asarray(fausse_carte("t2m", cls.zones))
        cls.pluie = np.asarray(fausse_carte("pluie", cls.zones))

    def test_mise_en_page_connue(self):
        um.effis_verifier(self.t2m, "t2m")
        um.effis_verifier(self.pluie, "pluie")

    def test_classe_lue_au_point(self):
        for (lon, lat), k in self.zones.items():
            self.assertEqual(um.effis_classe(self.t2m, lon, lat, "t2m"), k)
            self.assertEqual(um.effis_classe(self.pluie, lon, lat, "pluie"), k)

    def test_traits_et_contours_ne_changent_pas_la_classe(self):
        a = self.t2m.copy()
        x, y = um.effis_pixel(2.9, 42.7)
        a[y - 12 : y + 13, x - 3 : x + 4] = SOMBRE  # une frontière qui traverse la fenêtre
        a[y - 12 : y + 13, x + 5 : x + 7] = (192, 192, 192)  # et un contour
        self.assertEqual(um.effis_classe(a, 2.9, 42.7, "t2m"), 10)

    def test_aucune_couleur_de_legende(self):
        a = self.t2m.copy()
        x, y = um.effis_pixel(2.9, 42.7)
        a[y - 12 : y + 13, x - 12 : x + 13] = SOMBRE
        self.assertIsNone(um.effis_classe(a, 2.9, 42.7, "t2m"))

    def test_classe_dominante(self):
        a = self.t2m.copy()
        x, y = um.effis_pixel(2.9, 42.7)
        a[y - 12 : y + 13, x - 12 : x + 3] = um.EFFIS_CLASSES["t2m"]["couleurs"][12]  # 15 colonnes sur 25 dans une autre classe
        self.assertEqual(um.effis_classe(a, 2.9, 42.7, "t2m"), 12)

    def test_taille_inattendue(self):
        with self.assertRaisesRegex(RuntimeError, "pixels"):
            um.effis_verifier(np.zeros((100, 100, 3), dtype=np.uint8), "t2m")

    def test_legende_modifiee(self):
        img = fausse_carte("t2m", couleur_legende={3: (10, 20, 30)})
        with self.assertRaisesRegex(RuntimeError, "légende"):
            um.effis_verifier(np.asarray(img), "t2m")

    def test_quadrillage_deplace(self):
        img = fausse_carte("t2m", sans_meridien=15)
        with self.assertRaisesRegex(RuntimeError, "méridien 15"):
            um.effis_verifier(np.asarray(img), "t2m")


class Orchestration(unittest.TestCase):
    NOMS = {v: [f"Europe_MonthlyAnomalies_{v}_20261005_w{w}.png" for w in range(2, 6)] for v in ("T2m", "Rain")}

    @classmethod
    def setUpClass(cls):
        zones = {(z["lon"], z["lat"]): 10 for z in um.TENDANCE_ZONES}
        cls.png = {"T2m": png(fausse_carte("t2m", zones)), "Rain": png(fausse_carte("pluie", {k: 5 for k in zones}))}

    def installer(self, etag: str = "1", noms: dict | None = None):
        noms = noms or self.NOMS

        def faux_get(url, **kw):
            if url.endswith("_index.txt"):
                v = url.split("Europe_Monthly_")[1].split("_index")[0]
                return Reponse("\n".join(noms[v]) + "\n")
            v = "T2m" if "_T2m_" in url else "Rain"
            return Reponse(self.png[v])

        def fausse_tete(nom):
            return {"etag": f'"{etag}"', "modifie": "Fri, 09 Oct 2026 20:20:08 GMT"}

        return mock.patch.multiple(um, get=faux_get, effis_tete=fausse_tete)

    def test_bout_en_bout(self):
        with tempfile.TemporaryDirectory() as tmp, self.installer():
            out = Path(tmp)
            um.tendance(out)
            doc = json.loads((out / "tendance-effis.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["schema_version"], 2)
            self.assertEqual(doc["publie"], "2026-10-09T20:20:08Z")
            self.assertEqual(doc["semaines"], [{"debut": "2026-10-12", "fin": "2026-10-18"}, {"debut": "2026-10-19", "fin": "2026-10-25"}, {"debut": "2026-10-26", "fin": "2026-11-01"}, {"debut": "2026-11-02", "fin": "2026-11-08"}])
            self.assertEqual(len(doc["zones"]), 8)
            for z in doc["zones"]:
                self.assertEqual(z["semaines"], [{"t2m": 10, "pluie": 5}] * 4)
            self.assertEqual(doc["legendes"]["t2m"]["classes"][10], {"min": 1, "max": 2})
            self.assertEqual(doc["legendes"]["pluie"]["classes"][5], {"min": -30, "max": -20})
            self.assertEqual(doc["source"]["licence"], "CC BY 4.0")
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["tendance-effis.json"])  # aucune carte publiée

    def test_pas_de_retraitement_si_rien_n_a_change(self):
        with tempfile.TemporaryDirectory() as tmp, self.installer():
            out = Path(tmp)
            um.tendance(out)
            avant = (out / "tendance-effis.json").read_text(encoding="utf-8")
            with mock.patch.object(um, "effis_verifier", side_effect=AssertionError("ne doit pas relire les cartes")):
                um.tendance(out)
            self.assertEqual((out / "tendance-effis.json").read_text(encoding="utf-8"), avant)

    def test_retraitement_quand_une_carte_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.installer(etag="1"):
                um.tendance(out)
            avant = json.loads((out / "tendance-effis.json").read_text(encoding="utf-8"))
            with self.installer(etag="2"):
                um.tendance(out)
            apres = json.loads((out / "tendance-effis.json").read_text(encoding="utf-8"))
            self.assertNotEqual(avant["fichiers"], apres["fichiers"])

    def test_ancien_format_relance_le_traitement(self):
        with tempfile.TemporaryDirectory() as tmp, self.installer():
            out = Path(tmp)
            um.tendance(out)
            doc = json.loads((out / "tendance-effis.json").read_text(encoding="utf-8"))
            doc["schema_version"] = 1
            (out / "tendance-effis.json").write_text(json.dumps(doc), encoding="utf-8")
            um.tendance(out)
            self.assertEqual(json.loads((out / "tendance-effis.json").read_text(encoding="utf-8"))["schema_version"], 2)

    def test_index_incoherent(self):
        noms = {"T2m": self.NOMS["T2m"], "Rain": [n.replace("20261005", "20261012") for n in self.NOMS["Rain"]]}
        with tempfile.TemporaryDirectory() as tmp, self.installer(noms=noms):
            with self.assertRaisesRegex(RuntimeError, "index EFFIS"):
                um.tendance(Path(tmp))

    def test_trop_peu_de_semaines(self):
        noms = {v: n[:2] for v, n in self.NOMS.items()}
        with tempfile.TemporaryDirectory() as tmp, self.installer(noms=noms):
            with self.assertRaisesRegex(RuntimeError, "index EFFIS"):
                um.tendance(Path(tmp))

    def test_une_carte_illisible_ne_publie_rien(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.installer(), mock.patch.object(um, "effis_verifier", side_effect=RuntimeError("mise en page")):
                with self.assertRaises(RuntimeError):
                    um.tendance(out)
            self.assertFalse((out / "tendance-effis.json").exists())


class Nettoyage(unittest.TestCase):
    def test_effis_inchange(self):
        f = {"a.png": {"etag": '"1"', "modifie": "x"}}
        self.assertTrue(um.effis_inchange({"schema_version": 2, "fichiers": f}, f))
        self.assertFalse(um.effis_inchange({"schema_version": 1, "fichiers": f}, f))
        self.assertFalse(um.effis_inchange({"schema_version": 2, "fichiers": {"a.png": {"etag": '"2"', "modifie": "x"}}}, f))
        self.assertFalse(um.effis_inchange(None, f))

    def test_anciens_fichiers_retires(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "tendance").mkdir()
            (out / "tendance" / "t2m-1.webp").write_bytes(b"x")
            for nom in ("randonnee.json", "tendance.json", "tendance-effis.json", "cyclones.json"):
                (out / nom).write_text("{}", encoding="utf-8")
            um.retirer_anciens_fichiers(out)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["cyclones.json", "tendance-effis.json"])
            um.retirer_anciens_fichiers(out)  # sans erreur quand il n'y a plus rien


if __name__ == "__main__":
    unittest.main()
