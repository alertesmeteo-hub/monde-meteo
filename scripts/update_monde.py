#!/usr/bin/env python3
"""Données « monde » de la carte interactive Alertes Météo.

Publie dans un dossier (branche `data` du dépôt) :

- satellite/ : images Meteosat MTG « GeoColour » (EUMETSAT, EUMETView) sur l'Europe, l'Atlantique et
  l'Afrique du Nord, une image toutes les 20 minutes sur les 3 dernières heures ;
- maps/ : pression au niveau de la mer et vent à 10 m du modèle GFS 1° (NOAA) sur le monde, grilles de valeurs
  au même format que les dépôts régionaux (en-tête CEV1, uint16, gzip, lignes en projection Mercator) ;
- cyclones.json : cyclones tropicaux actifs (GDACS) avec trajectoire observée et prévue, catégories et cône ;
- geo/monde.json : côtes et frontières (Natural Earth 50 m), pour le fond de carte.

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


# --------------------------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pub")
    ap.add_argument("--only", default="satellite,satmonde,cyclones,geo,gfs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"satellite": satellite, "satmonde": satellite_world, "cyclones": cyclones, "geo": geo, "gfs": lambda o: gfs_world(o, a.force)}
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
