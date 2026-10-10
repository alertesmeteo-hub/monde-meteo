#!/usr/bin/env python3
"""Données « monde » de la carte interactive Alertes Météo.

Publie dans un dossier (branche `data` du dépôt) :

- satellite/ : images Meteosat MTG « GeoColour » (EUMETSAT, EUMETView) sur l'Europe, l'Atlantique et
  l'Afrique du Nord, une image toutes les 20 minutes sur les 3 dernières heures ;
- maps/ : pression au niveau de la mer et vent à 10 m du modèle GFS 1° (NOAA) sur le monde, grilles de valeurs
  au même format que les dépôts régionaux (en-tête CEV1, uint16, gzip, lignes en projection Mercator) ;
- cyclones.json : cyclones tropicaux actifs (GDACS) avec trajectoire observée et prévue, catégories et cône ;
- geo/monde.json : côtes et frontières (Natural Earth 50 m), pour le fond de carte.
- tendance-effis.json : tendance à 5 semaines, écarts hebdomadaires à la normale (température, pluie) du système de prévision mensuelle
  d'ECMWF, lus par zone sur les cartes publiées par l'EFFIS (Copernicus) ;

Chaque partie est indépendante : une source en panne ne bloque pas les autres, et les fichiers déjà publiés
sont conservés (le dossier de sortie part du contenu actuel de la branche `data`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import struct
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

UA = {"User-Agent": "alertes-meteo.com (donnees monde; contact alertes.meteo@gmail.com)"}
PROBE_MAGIC = b"CEV1"

# --------------------------------------------------------------------------------------------------- outils


def log(*parts: object) -> None:
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), "|", *parts, flush=True)


def get(url: str, *, params: dict | None = None, timeout: int = 60, tries: int = 4, stream: bool = False) -> requests.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA, stream=stream)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} sur {r.url}")
            if r.status_code in (403, 404):
                break
        except requests.RequestException as e:  # réseau
            last = e
        time.sleep(3 * (attempt + 1))
    raise last or RuntimeError(url)


def mercator(lat: np.ndarray | float) -> np.ndarray | float:
    r = np.radians(np.clip(lat, -85.0, 85.0))
    return np.log(np.tan(np.pi / 4.0 + r / 2.0))


def inverse_mercator(y: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arctan(np.exp(y)) - np.pi / 2.0)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_probe(values: np.ndarray, vmin: float, vmax: float, dest: Path) -> None:
    """Grille de valeurs (lignes du nord au sud, régulières en Mercator) : uint16, 65535 = manquant, gzip."""
    enc = np.full(values.shape, 65535, dtype="<u2")
    ok = np.isfinite(values)
    enc[ok] = np.rint((np.clip(values[ok], vmin, vmax) - vmin) / (vmax - vmin) * 65534.0).astype("<u2")
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<4sHHff", PROBE_MAGIC, enc.shape[1], enc.shape[0], vmin, vmax)
    with dest.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
        gz.write(header)
        gz.write(enc.tobytes(order="C"))


# --------------------------------------------------------------------------------------------------- satellite

WMS = "https://view.eumetsat.int/geoserver/ows"
SAT_LAYER = "mtg_fd:rgb_geocolour"
SAT_BOUNDS = {"south": 15.0, "west": -45.0, "north": 72.0, "east": 50.0}
SAT_WIDTH = 2000
SAT_FRAMES = 10
SAT_STEP_MIN = 20


def satellite(out: Path) -> None:
    caps = get(WMS, params={"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"}, timeout=90).text
    i = caps.find(f"<Name>{SAT_LAYER}</Name>")
    if i < 0:
        raise RuntimeError("couche satellite absente des capacités EUMETView")
    seg = caps[i : i + 4000]
    marker = 'name="time" default="'
    j = seg.find(marker)
    if j < 0:
        raise RuntimeError("dimension temporelle absente")
    latest = datetime.strptime(seg[j + len(marker) : j + len(marker) + 20], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    R = 6378137.0
    x0 = math.radians(SAT_BOUNDS["west"]) * R
    x1 = math.radians(SAT_BOUNDS["east"]) * R
    y0 = float(mercator(SAT_BOUNDS["south"])) * R
    y1 = float(mercator(SAT_BOUNDS["north"])) * R
    height = round(SAT_WIDTH * (y1 - y0) / (x1 - x0))
    sat_dir = out / "satellite"
    sat_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in range(SAT_FRAMES - 1, -1, -1):
        t = latest - timedelta(minutes=SAT_STEP_MIN * k)
        name = t.strftime("%Y%m%dT%H%M") + ".jpg"
        dest = sat_dir / name
        if not dest.exists():
            params = {
                "service": "WMS",
                "version": "1.3.0",
                "request": "GetMap",
                "layers": SAT_LAYER,
                "styles": "",
                "crs": "EPSG:3857",
                "bbox": f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}",
                "width": str(SAT_WIDTH),
                "height": str(height),
                "format": "image/jpeg",
                "time": iso(t),
            }
            try:
                r = get(WMS, params=params, timeout=120)
            except Exception as e:  # une image manquante ne bloque pas les autres
                log("satellite : image", iso(t), "indisponible :", e)
                continue
            if not r.headers.get("content-type", "").startswith("image/"):
                log("satellite : réponse non image pour", iso(t))
                continue
            dest.write_bytes(r.content)
        frames.append({"time": iso(t), "file": f"satellite/{name}"})
    if not frames:
        raise RuntimeError("aucune image satellite")
    keep = {Path(f["file"]).name for f in frames} | {"monde.jpg"}
    for old in sat_dir.glob("*.jpg"):
        if old.name not in keep:
            old.unlink()
    previous = {}
    try:
        previous = json.loads((sat_dir / "index.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    (sat_dir / "index.json").write_text(
        json.dumps(
            {
                **({"world": previous["world"]} if "world" in previous else {}),
                "generated_at": iso(datetime.now(timezone.utc)),
                "source": "EUMETSAT — Meteosat MTG, GeoColour RGB (EUMETView)",
                "layer": SAT_LAYER,
                "bounds": [[SAT_BOUNDS["south"], SAT_BOUNDS["west"]], [SAT_BOUNDS["north"], SAT_BOUNDS["east"]]],
                "frames": frames,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log(f"satellite : {len(frames)} images, dernière {frames[-1]['time']}")


# Mosaïque mondiale (image la plus récente) : couleurs naturelles GeoColor là où elles existent librement
# (GOES-Ouest, GOES-Est, Meteosat MTG), infrarouge mis en forme « nuit » ailleurs (Meteosat Océan Indien,
# Himawari) : fond bleu nuit, nuages blancs, pour rester homogène avec GeoColor.
GIBS = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
WORLD_SAT_SOUTH = -65.0
WORLD_SAT_NORTH = 70.0
WORLD_SAT_PX_PER_DEG = 10
WORLD_SAT_BANDS = [
    (-180.0, -110.0, GIBS, "GOES-West_ABI_GeoColor", "rgb"),
    (-110.0, -40.0, GIBS, "GOES-East_ABI_GeoColor", "rgb"),
    (-40.0, 55.0, WMS, SAT_LAYER, "rgb"),
    (55.0, 95.0, WMS, "msg_iodc:ir108", "ir:95"),
    (95.0, 180.0, GIBS, "Himawari_AHI_Band13_Clean_Infrared", "ir:140"),
]


def ir_to_night(img, floor: float = 95.0):
    """Infrarouge (gris, ou palette colorée pour les sommets très froids) -> nuages blancs sur fond bleu nuit."""
    from PIL import Image

    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    gray = a.mean(axis=2)
    # sommets les plus froids (palette colorée de la NASA) : nuages très blancs, sans aplat uniforme
    gray = np.where(sat > 0.25, 225.0 + 30.0 * np.clip(sat, 0, 1), gray)
    # `floor` : niveau de gris du fond (mer, sol) propre à chaque source, rendu en bleu nuit
    t = np.clip((gray - floor) / (255.0 - floor), 0.0, 1.0) ** 1.1
    navy = np.array([14.0, 20.0, 40.0])
    white = np.array([235.0, 238.0, 244.0])
    out = navy[None, None, :] * (1 - t[..., None]) + white[None, None, :] * t[..., None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def satellite_world(out: Path) -> None:
    from PIL import Image
    import io as _io

    R = 6378137.0
    y0 = float(mercator(WORLD_SAT_SOUTH)) * R
    y1 = float(mercator(WORLD_SAT_NORTH)) * R
    total_w = int(360 * WORLD_SAT_PX_PER_DEG)
    height = round(total_w * (y1 - y0) / (2 * math.pi * R))
    mosaic = Image.new("RGB", (total_w, height), (14, 20, 40))
    ok = 0
    for west, east, url, layer, kind in WORLD_SAT_BANDS:
        w = int(round((east - west) * WORLD_SAT_PX_PER_DEG))
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": layer,
            "STYLES": "",
            "CRS": "EPSG:3857",
            "BBOX": f"{math.radians(west) * R:.0f},{y0:.0f},{math.radians(east) * R:.0f},{y1:.0f}",
            "WIDTH": str(w),
            "HEIGHT": str(height),
            "FORMAT": "image/jpeg",
        }
        try:
            r = get(url, params=params, timeout=150)
            if not r.headers.get("content-type", "").startswith("image/"):
                raise RuntimeError("réponse non image")
            img = Image.open(_io.BytesIO(r.content)).convert("RGB")
            if kind.startswith("ir"):
                img = ir_to_night(img, float(kind.split(":")[1]))
            mosaic.paste(img.resize((w, height)), (int(round((west + 180) * WORLD_SAT_PX_PER_DEG)), 0))
            ok += 1
        except Exception as e:  # une bande manquante reste en bleu nuit
            log("satellite monde :", layer, "indisponible :", e)
    if ok < 3:
        raise RuntimeError(f"mosaïque incomplète ({ok}/5 bandes)")
    dest = out / "satellite" / "monde.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(dest, "JPEG", quality=78, optimize=True, progressive=True)
    idx_path = out / "satellite" / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    idx["world"] = {
        "file": "satellite/monde.jpg",
        "time": iso(datetime.now(timezone.utc)),
        "bounds": [[WORLD_SAT_SOUTH, -180.0], [WORLD_SAT_NORTH, 180.0]],
        "sources": "NOAA GOES-Ouest et GOES-Est GeoColor (NASA GIBS), EUMETSAT Meteosat MTG GeoColour et Meteosat Océan Indien IR, JMA Himawari IR (NASA GIBS)",
    }
    idx_path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    log(f"satellite monde : mosaïque {total_w}x{height}, {ok}/5 bandes, {dest.stat().st_size // 1024} Ko")


# --------------------------------------------------------------------------------------------------- cyclones

GDACS = "https://www.gdacs.org/gdacsapi/api"


def cyclones(out: Path) -> None:
    lst = get(f"{GDACS}/events/geteventlist/SEARCH", params={"eventlist": "TC"}, timeout=60).json()
    storms = []
    for f in lst.get("features", []):
        p = f.get("properties", {})
        if str(p.get("iscurrent")).lower() != "true":
            continue
        try:
            g = get(
                f"{GDACS}/polygons/getgeometry",
                params={"eventtype": "TC", "eventid": p["eventid"], "episodeid": p["episodeid"]},
                timeout=60,
            ).json()
        except Exception as e:
            log("cyclones : géométrie indisponible pour", p.get("eventname"), e)
            continue
        segs = []
        points = []
        cone = None
        for feat in g.get("features", []):
            cls = str(feat.get("properties", {}).get("Class", ""))
            geom = feat.get("geometry") or {}
            props = feat.get("properties", {})
            if cls.startswith("Line_Line") and geom.get("type") == "LineString":
                segs.append(
                    {
                        "cat": str(props.get("polygonlabel", "")),
                        "forecast": bool(props.get("forecast")),
                        "coords": [[round(x, 2), round(y, 2)] for x, y in geom["coordinates"]],
                    }
                )
            elif cls.startswith("Point_Polygon") and geom.get("type") == "Polygon":
                ring = geom["coordinates"][0]
                lon = sum(c[0] for c in ring) / len(ring)
                lat = sum(c[1] for c in ring) / len(ring)
                points.append({"lon": round(lon, 2), "lat": round(lat, 2), "label": str(props.get("polygonlabel", "")), "key": str(props.get("key", ""))})
            elif cls == "Poly_Cones" and geom.get("type") in ("Polygon", "MultiPolygon"):
                rings = geom["coordinates"] if geom["type"] == "Polygon" else [r for poly in geom["coordinates"] for r in poly]
                cone = [[[round(x, 2), round(y, 2)] for x, y in ring[:: max(1, len(ring) // 400)]] for ring in rings]
        sev = p.get("severitydata") or {}
        storms.append(
            {
                "id": p["eventid"],
                "name": str(p.get("eventname", "")),
                "alert": str(p.get("alertlevel", "")),
                "severity": str(sev.get("severitytext", "")),
                "max_wind_kmh": round(float(sev.get("severity", 0) or 0)),
                "from": p.get("fromdate"),
                "to": p.get("todate"),
                "countries": p.get("country", ""),
                "segments": segs,
                "points": points,
                "cone": cone,
                "report": (p.get("url") or {}).get("report"),
            }
        )
    (out / "cyclones.json").write_text(
        json.dumps({"generated_at": iso(datetime.now(timezone.utc)), "source": "GDACS (ONU / Commission européenne)", "storms": storms}, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"cyclones : {len(storms)} cyclone(s) actif(s)")


# --------------------------------------------------------------------------------------------------- côtes

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"


def geo(out: Path) -> None:
    dest = out / "geo" / "monde.json"
    if dest.exists():
        return

    def lines(name: str) -> list:
        gj = get(NE + name + ".geojson", timeout=120).json()
        res = []
        for f in gj["features"]:
            g = f["geometry"]
            parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
            for part in parts:
                pts = [[round(x, 2), round(y, 2)] for x, y in part]
                if len(pts) >= 2:
                    res.append(pts)
        return res

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"coast": lines("ne_50m_coastline"), "borders": lines("ne_50m_admin_0_boundary_lines_land")}, separators=(",", ":")), encoding="utf-8")
    log("geo : côtes et frontières écrites")


# --------------------------------------------------------------------------------------------------- GFS monde

NOMADS = "https://nomads.ncep.noaa.gov"
GFS_HOURS = list(range(0, 121, 3))
WORLD_NORTH = 80.0
WORLD_SOUTH = -80.0
PRESSURE_STOPS = [
    [970, "#2f7d92"], [990, "#56a6b2"], [1000, "#7fbfc4"], [1008, "#a6d2d1"], [1011, "#c9e2de"], [1013, "#eceae4"],
    [1015, "#f0dccd"], [1018, "#eac3a8"], [1024, "#dda283"], [1032, "#c9805e"], [1045, "#a9573f"],
]


def latest_gfs_run() -> datetime:
    now = datetime.now(timezone.utc)
    base = now.replace(minute=0, second=0, microsecond=0, hour=(now.hour // 6) * 6)
    for k in range(0, 5):
        run = base - timedelta(hours=6 * k)
        url = f"{NOMADS}/pub/data/nccf/com/gfs/prod/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.1p00.f{GFS_HOURS[-1]:03d}.idx"
        try:
            r = requests.head(url, timeout=30, headers=UA)
            if r.status_code == 200:
                return run
        except requests.RequestException:
            pass
    raise RuntimeError("aucun run GFS complet trouvé sur NOMADS")


def decode_step(content: bytes) -> dict[str, np.ndarray]:
    """Décode le GRIB2 filtré (pression mer, vent 10 m) en grilles lat (nord→sud) x lon (-180→180)."""
    from eccodes import codes_get, codes_get_array, codes_grib_new_from_file, codes_release

    fields: dict[str, np.ndarray] = {}
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name
    with open(path, "rb") as fh:
        while True:
            gid = codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                short = codes_get(gid, "shortName")
                ni = codes_get(gid, "Ni")
                nj = codes_get(gid, "Nj")
                lat0 = codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                vals = np.asarray(codes_get_array(gid, "values"), dtype=np.float64).reshape(nj, ni)
                if lat0 < 0:  # balayage sud → nord
                    vals = vals[::-1]
                # longitudes 0..360 → -180..180
                vals = np.roll(vals, ni // 2, axis=1)
                fields[short] = vals
            finally:
                codes_release(gid)
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:  # Windows : fichier encore verrouillé par eccodes, sans conséquence
        pass
    return fields


def to_mercator_rows(grid: np.ndarray, lat_north: float, lat_step: float, height: int) -> np.ndarray:
    """Rééchantillonne une grille régulière en latitude vers des lignes régulières en Mercator (80° N → 80° S)."""
    ys = np.linspace(float(mercator(WORLD_NORTH)), float(mercator(WORLD_SOUTH)), height)
    lats = inverse_mercator(ys)
    fi = (lat_north - lats) / lat_step
    i0 = np.clip(np.floor(fi).astype(int), 0, grid.shape[0] - 2)
    t = (fi - i0)[:, None]
    return grid[i0] * (1 - t) + grid[i0 + 1] * t


def gfs_world(out: Path, force: bool) -> None:
    run = latest_gfs_run()
    idx_path = out / "maps" / "index.json"
    if idx_path.exists() and not force:
        try:
            if json.loads(idx_path.read_text(encoding="utf-8")).get("run_time") == iso(run):
                log("GFS monde : run", iso(run), "déjà publié")
                return
        except Exception:
            pass
    log("GFS monde : run", iso(run))
    tmp_dir = out / "maps.tmp"
    if tmp_dir.exists():
        for f in sorted(tmp_dir.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
    steps = []
    for h in GFS_HOURS:
        params = {
            "dir": f"/gfs.{run:%Y%m%d}/{run:%H}/atmos",
            "file": f"gfs.t{run:%H}z.pgrb2.1p00.f{h:03d}",
            "var_PRMSL": "on",
            "var_UGRD": "on",
            "var_VGRD": "on",
            "lev_mean_sea_level": "on",
            "lev_10_m_above_ground": "on",
        }
        r = get(f"{NOMADS}/cgi-bin/filter_gfs_1p00.pl", params=params, timeout=120)
        f = decode_step(r.content)
        if not {"prmsl", "10u", "10v"} <= f.keys():
            raise RuntimeError(f"champs manquants à l'échéance {h} : {sorted(f)}")
        ni = f["prmsl"].shape[1]
        height = round(ni * (float(mercator(WORLD_NORTH)) - float(mercator(WORLD_SOUTH))) / (2 * math.pi))
        lat_step = 180.0 / (f["prmsl"].shape[0] - 1)
        pres = to_mercator_rows(f["prmsl"] / 100.0, 90.0, lat_step, height)
        u = to_mercator_rows(f["10u"] * 3.6, 90.0, lat_step, height)
        v = to_mercator_rows(f["10v"] * 3.6, 90.0, lat_step, height)
        name = f"{h:03d}.hkv.gz"
        # valeurs arrondies (0,1 hPa ; 1 km/h) : fichiers ~4x plus légers, sans perte visible
        write_probe(pres.astype(np.float32), 940.0, 940.0 + 6553.4, tmp_dir / "values" / "pression" / name)
        write_probe(u.astype(np.float32), -200.0, -200.0 + 65534.0, tmp_dir / "values" / "vent_u" / name)
        write_probe(v.astype(np.float32), -200.0, -200.0 + 65534.0, tmp_dir / "values" / "vent_v" / name)
        steps.append(
            {
                "lead_hour": h,
                "valid_time": iso(run + timedelta(hours=h)),
                "files": {},
                "probes": {"pression": f"maps/values/pression/{name}", "vent_u": f"maps/values/vent_u/{name}", "vent_v": f"maps/values/vent_v/{name}"},
            }
        )
        time.sleep(0.6)  # politesse envers NOMADS
    # cellules centrées sur les points de grille GFS (1°) : bords décalés d'une demi-maille
    half = 180.0 / ni
    index = {
        "generated_at": iso(datetime.now(timezone.utc)),
        "run_time": iso(run),
        "model": "GFS 1° (NOAA/NCEP)",
        "bounds": {"south": WORLD_SOUTH, "west": -180.0 - half, "north": WORLD_NORTH, "east": 180.0 - half},
        "layers": {"pression": {"label": "Pression au niveau de la mer", "unit": "hPa", "decimals": 0, "stops": [{"value": v, "color": c} for v, c in PRESSURE_STOPS]}},
        "steps": steps,
    }
    (tmp_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    # remplacement atomique du dossier maps/
    final = out / "maps"
    if final.exists():
        for f in sorted(final.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        final.rmdir()
    tmp_dir.rename(final)
    log(f"GFS monde : {len(steps)} échéances écrites")


# --------------------------------------------------------------------------------------------------- tendance à 5 semaines (cartes EFFIS)

# L'EFFIS (Copernicus, Commission européenne) publie des cartes d'anomalies hebdomadaires (température à 2 m en °C, pluie en mm) du système de
# prévision mensuelle d'ECMWF sur l'Europe. Contenu de l'Union européenne sous licence CC BY 4.0 (https://forest-fire.emergency.copernicus.eu/
# about-effis/data-license) : réutilisation permise avec mention de la source et des modifications. On lit la classe de couleur de la carte au
# point de chaque zone ; les cartes elles-mêmes ne sont pas republiées.
EFFIS_CARTES = "https://maps.effis.emergency.copernicus.eu/LongTermForecasts/Monthly/Europe/"
EFFIS_APP = "https://forest-fire.emergency.copernicus.eu/apps/effis.longterm.forecasts/"
EFFIS_VARIABLES = {"T2m": "t2m", "Rain": "pluie"}  # nom dans les fichiers EFFIS -> nom publié

# Classes des cartes lues sur leurs légendes : bornes (°C ou mm) et couleurs exactes. La classe du milieu (blanche) est « proche de la normale » ;
# au-delà des bornes extrêmes, une valeur garde la couleur de la classe extrême.
EFFIS_CLASSES = {
    "t2m": {
        "unite": "°C",
        "bornes": [-8, -7, -6, -5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5, 6, 7, 8],
        "couleurs": [
            (0, 0, 176), (0, 36, 224), (0, 72, 255), (0, 108, 255), (0, 144, 255), (0, 180, 255), (0, 216, 255), (0, 255, 255),
            (255, 255, 255),
            (255, 224, 168), (255, 192, 144), (255, 160, 120), (255, 128, 96), (255, 96, 72), (255, 64, 48), (224, 32, 24), (176, 0, 0),
        ],
    },
    "pluie": {
        "unite": "mm",
        "bornes": [-100, -80, -60, -50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50, 60, 80, 100],
        "couleurs": [
            (176, 0, 0), (224, 32, 24), (255, 64, 48), (255, 96, 72), (255, 128, 96), (255, 160, 120), (255, 192, 144), (255, 224, 168),
            (255, 255, 255),
            (0, 255, 255), (0, 216, 255), (0, 180, 255), (0, 144, 255), (0, 108, 255), (0, 72, 255), (0, 36, 224), (0, 0, 176),
        ],
    },
}

# Géométrie des cartes : projection géographique (même échelle en longitude et en latitude), 7015 x 4960 pixels. Calage lu sur les lignes
# pointillées du quadrillage (méridiens -25 à 55° et parallèles 25 à 75°, tous les 10°).
EFFIS_TAILLE = (7015, 4960)
EFFIS_X0, EFFIS_PX_LON = 662.0, 71.125  # abscisse du méridien -25° ; pixels par degré de longitude
EFFIS_Y0, EFFIS_PX_LAT = 1047.0, 71.1  # ordonnée du parallèle 75° ; pixels par degré de latitude
EFFIS_CADRE = (310, 880, 6699, 4779)  # intérieur du cadre de la carte : x0, y0, x1, y1
EFFIS_LEGENDE_Y = 650
# centre des 16 pastilles de légende (8 négatives puis 8 positives ; la zone blanche du milieu n'a pas de pastille)
EFFIS_LEGENDE_X = [734, 1019, 1302, 1584, 1869, 2154, 2439, 2724, 4289, 4574, 4859, 5142, 5424, 5709, 5994, 6279]
EFFIS_RAYON = 12  # lecture d'une zone : fenêtre de (2 x rayon + 1) pixels de côté autour du point, soit environ 0,35°

# Zones de la tendance : la grille du modèle étendu ECMWF fait environ 36 km, un point par grand secteur suffit.
TENDANCE_ZONES = [
    {"id": "roussillon", "nom": "Plaine du Roussillon", "groupe": "66", "lat": 42.70, "lon": 2.90},
    {"id": "vallespir", "nom": "Vallespir et Albères", "groupe": "66", "lat": 42.45, "lon": 2.75},
    {"id": "conflent", "nom": "Conflent et Fenouillèdes", "groupe": "66", "lat": 42.68, "lon": 2.40},
    {"id": "cerdagne", "nom": "Cerdagne et Capcir", "groupe": "66", "lat": 42.50, "lon": 2.05},
    {"id": "aude", "nom": "Aude (Carcassonne)", "groupe": "Occitanie", "lat": 43.21, "lon": 2.35},
    {"id": "herault", "nom": "Hérault (Montpellier)", "groupe": "Occitanie", "lat": 43.61, "lon": 3.88},
    {"id": "toulouse", "nom": "Haute-Garonne (Toulouse)", "groupe": "Occitanie", "lat": 43.60, "lon": 1.44},
    {"id": "ariege", "nom": "Ariège (Foix)", "groupe": "Occitanie", "lat": 42.96, "lon": 1.61},
]


def effis_pixel(lon: float, lat: float) -> tuple[int, int]:
    """Pixel (x, y) d'un point sur les cartes EFFIS pleine taille."""
    return round(EFFIS_X0 + (lon + 25.0) * EFFIS_PX_LON), round(EFFIS_Y0 + (75.0 - lat) * EFFIS_PX_LAT)


def effis_legende(variable: str) -> dict:
    """Classes d'une variable (« t2m » ou « pluie ») pour la page : unité et bornes de chaque classe, dans l'ordre des couleurs de la carte."""
    d = EFFIS_CLASSES[variable]
    b = d["bornes"]
    return {"unite": d["unite"], "classes": [{"min": b[i], "max": b[i + 1]} for i in range(len(d["couleurs"]))]}


def effis_noms(texte: str, variable: str) -> list[tuple[int, str, date]]:
    """Fichiers de l'index EFFIS d'une variable : (numéro de semaine, nom, date du lundi de la semaine 1), triés par semaine."""
    rx = re.compile(rf"^Europe_MonthlyAnomalies_{variable}_(\d{{8}})_w(\d+)\.png$")
    out = []
    for ligne in texte.splitlines():
        m = rx.match(ligne.strip())
        if m:
            out.append((int(m.group(2)), m.group(0), datetime.strptime(m.group(1), "%Y%m%d").date()))
    return sorted(out)


def effis_verifier(a: np.ndarray, variable: str) -> None:
    """Vérifie que la carte a la mise en page connue (taille, cadre, quadrillage, légende). Sinon on ne lit rien : une mauvaise lecture vaut moins
    que l'absence de donnée, et les cartes précédentes restent publiées."""
    h, w = a.shape[:2]
    if (w, h) != EFFIS_TAILLE:
        raise RuntimeError(f"carte EFFIS de {w} x {h} pixels au lieu de {EFFIS_TAILLE[0]} x {EFFIS_TAILLE[1]}")
    x0, y0, x1, y1 = EFFIS_CADRE
    sombre = a.max(axis=-1) < 90
    for x in (x0 - 5, x1 + 5):
        if sombre[y0 : y1 + 1, x].mean() < 0.9:
            raise RuntimeError("carte EFFIS : cadre vertical introuvable")
    for y in (y0 - 5, y1 + 5):
        if sombre[y, x0 : x1 + 1].mean() < 0.9:
            raise RuntimeError("carte EFFIS : cadre horizontal introuvable")
    for lon in range(-25, 56, 10):
        x = effis_pixel(lon, 50.0)[0]
        if sombre[y0 : y1 + 1, x - 2 : x + 3].any(axis=1).mean() < 0.15:
            raise RuntimeError(f"carte EFFIS : méridien {lon}° absent du quadrillage")
    for lat in range(25, 76, 10):
        y = effis_pixel(0.0, lat)[1]
        if sombre[y - 2 : y + 3, x0 : x1 + 1].any(axis=0).mean() < 0.15:
            raise RuntimeError(f"carte EFFIS : parallèle {lat}° absent du quadrillage")
    couleurs = EFFIS_CLASSES[variable]["couleurs"]
    for x, c in zip(EFFIS_LEGENDE_X, couleurs[:8] + couleurs[9:]):
        if tuple(int(v) for v in a[EFFIS_LEGENDE_Y, x]) != c:
            raise RuntimeError("carte EFFIS : couleurs de la légende changées")


def effis_classe(a: np.ndarray, lon: float, lat: float, variable: str) -> int | None:
    """Classe de la carte (indice dans EFFIS_CLASSES) au point donné : couleur de légende la plus fréquente dans la fenêtre autour du point
    (les traits de côtes, de frontières et de contours ne sont pas des couleurs de légende). None si aucune."""
    x, y = effis_pixel(lon, lat)
    r = EFFIS_RAYON
    fen = a[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
    meilleur, n_max = None, 0
    for i, c in enumerate(EFFIS_CLASSES[variable]["couleurs"]):
        n = int(np.all(fen == np.array(c, dtype=fen.dtype), axis=-1).sum())
        if n > n_max:
            meilleur, n_max = i, n
    return meilleur if n_max >= 100 else None


def effis_tete(nom: str) -> dict:
    """ETag et date de modification d'une carte (requête HEAD, sans la télécharger)."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.head(EFFIS_CARTES + nom, timeout=30, headers=UA, allow_redirects=True)
            if r.status_code == 200:
                return {"etag": r.headers.get("ETag", ""), "modifie": r.headers.get("Last-Modified", "")}
            last = RuntimeError(f"HTTP {r.status_code} sur {r.url}")
        except requests.RequestException as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise last or RuntimeError(nom)


EFFIS_SCHEMA = 2


def effis_inchange(prev: dict | None, fichiers: dict[str, dict]) -> bool:
    """Les cartes EFFIS n'ont pas changé depuis la dernière lecture (et le fichier publié a la forme actuelle)."""
    return bool(prev) and prev.get("schema_version") == EFFIS_SCHEMA and prev.get("fichiers") == fichiers


def tendance(out: Path) -> None:
    """Tendance à 5 semaines : écarts hebdomadaires à la normale (température, pluie) du système de prévision mensuelle d'ECMWF, lus par zone sur
    les cartes de l'EFFIS -> tendance-effis.json. Rien n'est relu tant que l'EFFIS n'a pas republié ses cartes."""
    import io
    from email.utils import parsedate_to_datetime

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 100_000_000
    index = {v: effis_noms(get(f"{EFFIS_CARTES}Europe_Monthly_{v}_index.txt").text, v) for v in EFFIS_VARIABLES}
    bases = {b for noms in index.values() for _, _, b in noms}
    jeux = {tuple(w for w, _, _ in noms) for noms in index.values()}
    if len(bases) != 1 or len(jeux) != 1 or len(next(iter(jeux))) < 4:
        raise RuntimeError(f"index EFFIS inattendu : dates {sorted(bases)}, semaines {sorted(jeux)}")
    base = next(iter(bases))
    fichiers = {nom: effis_tete(nom) for noms in index.values() for _, nom, _ in noms}
    dest = out / "tendance-effis.json"
    prev = None
    try:
        prev = json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        pass
    if effis_inchange(prev, fichiers):
        log("tendance : cartes EFFIS inchangées")
        return

    zones = [{k: z[k] for k in ("id", "nom", "groupe", "lat", "lon")} | {"semaines": []} for z in TENDANCE_ZONES]
    semaines = []
    for rang in range(len(index["T2m"])):
        debut = base + timedelta(days=7 * (index["T2m"][rang][0] - 1))
        semaines.append({"debut": debut.isoformat(), "fin": (debut + timedelta(days=6)).isoformat()})
        lectures: dict[str, dict] = {z["id"]: {} for z in TENDANCE_ZONES}
        for var, noms in index.items():
            pub = EFFIS_VARIABLES[var]
            img = Image.open(io.BytesIO(get(EFFIS_CARTES + noms[rang][1], timeout=180).content)).convert("RGB")
            arr = np.asarray(img)
            effis_verifier(arr, pub)
            for z in TENDANCE_ZONES:
                lectures[z["id"]][pub] = effis_classe(arr, z["lon"], z["lat"], pub)
            del img, arr
        for z in zones:
            z["semaines"].append(lectures[z["id"]])
    modifie = max(parsedate_to_datetime(f["modifie"]) for f in fichiers.values() if f["modifie"])
    doc = {
        "schema_version": EFFIS_SCHEMA,
        "generated_at": iso(datetime.now(timezone.utc)),
        "publie": iso(modifie),
        "source": {
            "nom": "EFFIS (Copernicus, Union européenne), d'après le système de prévision mensuelle d'ECMWF",
            "url": EFFIS_APP,
            "licence": "CC BY 4.0",
            "modifications": "valeurs des zones lues sur les cartes (classe de la légende au point de la zone)",
        },
        "semaines": semaines,
        "legendes": {v: effis_legende(v) for v in EFFIS_VARIABLES.values()},
        "zones": zones,
        "fichiers": fichiers,
    }
    dest.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log(f"tendance : {len(semaines)} semaines (du {semaines[0]['debut']} au {semaines[-1]['fin']}), cartes EFFIS publiées le {doc['publie']}")


def retirer_anciens_fichiers(out: Path) -> None:
    """Fichiers d'anciennes versions, à ne plus publier : météo randonnée (calculée par le site avec ses modèles), tendance EC46 d'Open-Meteo et
    cartes recadrées de la tendance."""
    import shutil

    (out / "randonnee.json").unlink(missing_ok=True)
    (out / "tendance.json").unlink(missing_ok=True)
    shutil.rmtree(out / "tendance", ignore_errors=True)


# --------------------------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pub")
    ap.add_argument("--only", default="satellite,satmonde,cyclones,geo,gfs,tendance")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    retirer_anciens_fichiers(out)
    parts = {"satellite": satellite, "satmonde": satellite_world, "cyclones": cyclones, "geo": geo, "gfs": lambda o: gfs_world(o, a.force), "tendance": tendance}
    failures = []
    for name in a.only.split(","):
        try:
            parts[name](out)
        except Exception as e:  # chaque partie est indépendante
            failures.append(name)
            log(f"ÉCHEC {name} :", e)
    if failures:
        print(f"::warning::parties en échec : {', '.join(failures)} (données précédentes conservées)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
