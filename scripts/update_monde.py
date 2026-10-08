#!/usr/bin/env python3
"""Données « monde » de la carte interactive Alertes Météo.

Publie dans un dossier (branche `data` du dépôt) :

- satellite/ : images Meteosat MTG « GeoColour » (EUMETSAT, EUMETView) sur l'Europe, l'Atlantique et
  l'Afrique du Nord, une image toutes les 20 minutes sur les 3 dernières heures ;
- maps/ : pression au niveau de la mer et vent à 10 m du modèle GFS 1° (NOAA) sur le monde, grilles de valeurs
  au même format que les dépôts régionaux (en-tête CEV1, uint16, gzip, lignes en projection Mercator) ;
- cyclones.json : cyclones tropicaux actifs (GDACS) avec trajectoire observée et prévue, catégories et cône ;
- geo/monde.json : côtes et frontières (Natural Earth 50 m), pour le fond de carte.
- tendance.json : tendance à 6 semaines du modèle étendu ECMWF (EC46), anomalies hebdomadaires par zone ;
- randonnee.json : prévisions horaires à 7 jours au sommet et au départ des randonnées des Pyrénées-Orientales.

Chaque partie est indépendante : une source en panne ne bloque pas les autres, et les fichiers déjà publiés
sont conservés (le dossier de sortie part du contenu actuel de la branche `data`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import tempfile
import time
from datetime import datetime, timedelta, timezone
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


# --------------------------------------------------------------------------------------------------- tendance 5 semaines

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
SEASONAL = "https://seasonal-api.open-meteo.com/v1/seasonal"

# Zones de la tendance : la grille du modèle étendu ECMWF (EC46) fait ~36 km, un point par grand secteur suffit.
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
TENDANCE_VARS = [
    "temperature_2m_mean",
    "temperature_2m_anomaly",
    "temperature_2m_anomaly_gt1",
    "temperature_2m_anomaly_ltm1",
    "precipitation_mean",
    "precipitation_anomaly",
    "precipitation_anomaly_gt0",
    "pressure_msl_anomaly",
    "pressure_msl_anomaly_gt0",
]


def tendance(out: Path) -> None:
    """Anomalies hebdomadaires du modèle étendu ECMWF (EC46, 51 membres) sur 6 semaines -> tendance.json."""
    r = get(
        SEASONAL,
        params={
            "latitude": ",".join(str(z["lat"]) for z in TENDANCE_ZONES),
            "longitude": ",".join(str(z["lon"]) for z in TENDANCE_ZONES),
            "models": "ecmwf_ec46",
            "weekly": ",".join(TENDANCE_VARS),
        },
        timeout=90,
    ).json()
    data = r if isinstance(r, list) else [r]
    zones = []
    for z, d in zip(TENDANCE_ZONES, data):
        w = d.get("weekly") or {}
        weeks = []
        for i, start in enumerate(w.get("time", [])):
            row = {v: w.get(v, [None] * (i + 1))[i] for v in TENDANCE_VARS}
            if row["temperature_2m_mean"] is None:
                continue
            weeks.append({"debut": start, **row})
        if weeks:
            zones.append({k: z[k] for k in ("id", "nom", "groupe", "lat", "lon")} | {"semaines": weeks})
    if not zones:
        raise RuntimeError("EC46 : aucune semaine disponible")
    (out / "tendance.json").write_text(
        json.dumps(
            {
                "generated_at": iso(datetime.now(timezone.utc)),
                "source": "ECMWF, modèle étendu EC46 (51 membres), via Open-Meteo",
                "zones": zones,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    log(f"tendance : {len(zones)} zones, {len(zones[0]['semaines'])} semaines")


# --------------------------------------------------------------------------------------------------- randonnée

# Coordonnées approchées des sommets et des départs ; l'altitude donnée sert à corriger la température du modèle.
RANDOS = [
    {"id": "canigou", "nom": "Pic du Canigou", "massif": "Canigou", "sommet": [42.5189, 2.4567, 2784], "depart": ["Refuge de Mariailles", 42.5414, 2.3975, 1718]},
    {"id": "puigmal", "nom": "Puigmal d'Err", "massif": "Cerdagne", "sommet": [42.3836, 2.1167, 2910], "depart": ["Ancienne station Puigmal 2900", 42.4080, 2.1070, 1900]},
    {"id": "carlit", "nom": "Pic Carlit", "massif": "Carlit", "sommet": [42.5711, 1.9353, 2921], "depart": ["Lac des Bouillouses", 42.5603, 1.9958, 2015]},
    {"id": "cambre-aze", "nom": "Cambre d'Aze", "massif": "Cerdagne", "sommet": [42.4700, 2.0720, 2750], "depart": ["Saint-Pierre-dels-Forcats", 42.4878, 2.1086, 1600]},
    {"id": "bastiments", "nom": "Pic de Bastiments", "massif": "Haut-Vallespir", "sommet": [42.4240, 2.2260, 2881], "depart": ["Vallter 2000", 42.4250, 2.2650, 2000]},
    {"id": "costabonne", "nom": "Pic de Costabonne", "massif": "Haut-Vallespir", "sommet": [42.4025, 2.3608, 2465], "depart": ["La Preste", 42.4044, 2.4053, 1130]},
    {"id": "madres", "nom": "Pic de Madrès", "massif": "Madrès", "sommet": [42.7375, 2.2253, 2469], "depart": ["Col de Jau", 42.6920, 2.2620, 1506]},
    {"id": "roc-france", "nom": "Roc de France", "massif": "Vallespir", "sommet": [42.4194, 2.6778, 1450], "depart": ["Las Illas", 42.4325, 2.7203, 550]},
    {"id": "neulos", "nom": "Pic Neulós", "massif": "Albères", "sommet": [42.4836, 2.9439, 1256], "depart": ["Col de l'Ouillat", 42.4760, 2.9130, 936]},
    {"id": "massane", "nom": "Tour de la Massane", "massif": "Albères", "sommet": [42.4908, 3.0400, 792], "depart": ["Argelès, château de Valmy", 42.5350, 3.0240, 70]},
    {"id": "madeloc", "nom": "Tour de Madeloc", "massif": "Côte Vermeille", "sommet": [42.4940, 3.0840, 652], "depart": ["Collioure", 42.5256, 3.0833, 10]},
    {"id": "forca-real", "nom": "Ermitage de Força Réal", "massif": "Aspres et Ribéral", "sommet": [42.7556, 2.6578, 507], "depart": ["Millas", 42.6939, 2.6967, 100]},
    {"id": "galamus", "nom": "Gorges de Galamus", "massif": "Fenouillèdes", "sommet": [42.8440, 2.4700, 450], "depart": ["Saint-Paul-de-Fenouillet", 42.8100, 2.5050, 260]},
]
RANDO_HOURLY = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "precipitation_probability",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "cloud_cover",
    "freezing_level_height",
    "snowfall",
    "cape",
    "visibility",
]
# Valeurs utiles au départ (le reste n'est lu qu'au sommet).
RANDO_DEPART = ["temperature_2m", "apparent_temperature", "precipitation", "wind_gusts_10m", "weather_code"]


def _compact(v):
    if v is None:
        return None
    return round(v) if abs(v) >= 10 else round(v, 1)


def randonnee(out: Path) -> None:
    """Prévisions horaires à 7 jours au sommet et au départ des randonnées -> randonnee.json."""
    pts = [(r["sommet"][0], r["sommet"][1], r["sommet"][2]) for r in RANDOS] + [(r["depart"][1], r["depart"][2], r["depart"][3]) for r in RANDOS]
    res = get(
        OPEN_METEO,
        params={
            "latitude": ",".join(str(p[0]) for p in pts),
            "longitude": ",".join(str(p[1]) for p in pts),
            "elevation": ",".join(str(p[2]) for p in pts),
            "hourly": ",".join(RANDO_HOURLY),
            "daily": "sunrise,sunset,uv_index_max",
            "timezone": "Europe/Paris",
            "forecast_days": 7,
        },
        timeout=90,
    ).json()
    if not isinstance(res, list) or len(res) != len(pts):
        raise RuntimeError("Open-Meteo : réponse inattendue pour les randonnées")
    n = len(RANDOS)
    times = res[0]["hourly"]["time"]
    items = []
    for i, r in enumerate(RANDOS):
        top, low = res[i], res[n + i]
        items.append(
            {
                "id": r["id"],
                "nom": r["nom"],
                "massif": r["massif"],
                "sommet": {"lat": r["sommet"][0], "lon": r["sommet"][1], "alt": r["sommet"][2]},
                "depart": {"nom": r["depart"][0], "lat": r["depart"][1], "lon": r["depart"][2], "alt": r["depart"][3]},
                "h": {v: [_compact(x) for x in top["hourly"][v]] for v in RANDO_HOURLY},
                "hd": {v: [_compact(x) for x in low["hourly"][v]] for v in RANDO_DEPART},
                "jours": {
                    "date": top["daily"]["time"],
                    "lever": [s[-5:] for s in top["daily"]["sunrise"]],
                    "coucher": [s[-5:] for s in top["daily"]["sunset"]],
                    "uv": top["daily"]["uv_index_max"],
                },
            }
        )
    (out / "randonnee.json").write_text(
        json.dumps(
            {
                "generated_at": iso(datetime.now(timezone.utc)),
                "source": "Open-Meteo (Météo-France AROME/ARPEGE, ECMWF), température corrigée de l'altitude",
                "fuseau": "Europe/Paris",
                "heures": times,
                "randos": items,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    log(f"randonnée : {n} itinéraires, {len(times)} heures")


# --------------------------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pub")
    ap.add_argument("--only", default="satellite,satmonde,cyclones,geo,gfs,tendance,randonnee")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"satellite": satellite, "satmonde": satellite_world, "cyclones": cyclones, "geo": geo, "gfs": lambda o: gfs_world(o, a.force), "tendance": tendance, "randonnee": randonnee}
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
