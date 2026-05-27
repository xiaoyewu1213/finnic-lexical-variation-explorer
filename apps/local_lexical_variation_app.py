from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency


BASE = Path("/Users/xiaoye/Documents/New project")
DEFAULT_WORD_ANALYSIS = BASE / "outputs" / "dhh26_full" / "word_analysis.parquet"
DEFAULT_P_PL = BASE / "outputs" / "dhh26_full" / "p_pl.parquet"
DEFAULT_PLACES = BASE / "outputs" / "dhh26_full" / "places.parquet"
DEFAULT_AREAS = BASE / "apps" / "assets" / "areas.geojson"

REQUIRED_COLUMNS = [
    "p_id",
    "v_id",
    "v_pos",
    "w_pos",
    "word",
    "word_normalised",
    "local_lemma",
    "standard_lemma",
    "root",
    "word_in_english",
]

PALETTE = [
    "#2f7d6d",
    "#3568b8",
    "#b35c2e",
    "#7b5ca8",
    "#c05f5f",
    "#8a7b2f",
    "#24728f",
    "#c17a21",
    "#6b8f28",
    "#a34a7b",
    "#5f6a72",
    "#8a8a8a",
]


st.set_page_config(page_title="Lexical Variation Generator", layout="wide")


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "nan", "None", "-", "[Same as above]"}:
        return ""
    return re.sub(r"\s+", " ", text)


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def norm_key(value: object, drop_modifiers: bool = True) -> str:
    text = strip_accents(clean(value).casefold().replace("’", "'"))
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", text)
    text = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [word for word in re.sub(r"\s+", " ", text).strip().split() if word]
    if drop_modifiers:
        drops = {
            "to",
            "from",
            "into",
            "onto",
            "in",
            "on",
            "at",
            "by",
            "of",
            "with",
            "for",
            "as",
            "the",
            "a",
            "an",
            "my",
            "your",
            "our",
            "his",
            "her",
            "their",
            "its",
            "little",
            "dear",
        }
        words = [word for word in words if word not in drops]
    return " ".join(words)


def singularize_basic(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("es"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def concept_aliases(concept: str, extra_aliases: str) -> list[str]:
    raw = [concept]
    raw.extend(part.strip() for part in re.split(r"[,;\n]", extra_aliases or "") if part.strip())
    aliases = set()
    for item in raw:
        key = norm_key(item)
        if not key:
            continue
        aliases.add(key)
        aliases.add(singularize_basic(key))
    return sorted(aliases)


def concept_matcher(aliases: list[str], mode: str):
    alias_set = set(aliases)
    alias_regex = re.compile(r"\b(" + "|".join(re.escape(alias) for alias in sorted(alias_set, key=len, reverse=True)) + r")\b")

    def matches(value: object) -> bool:
        key = norm_key(value)
        if not key:
            return False
        if mode == "Exact cleaned concept":
            return key in alias_set or singularize_basic(key) in alias_set
        return bool(alias_regex.search(key))

    return matches


def norm_place(value: object) -> str:
    text = strip_accents(clean(value).casefold())
    text = re.sub(r"[^a-z0-9 -]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def feature_name(props: dict) -> str:
    for key in ["parish_name", "NAME_ALT", "Parname_ne", "parish_nam", "name"]:
        value = props.get(key)
        if value not in (None, ""):
            return str(value)
    return str(props.get("id", ""))


def geometry_rings(geom: dict) -> list[list[tuple[float, float]]]:
    if not geom:
        return []
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    rings: list[list[tuple[float, float]]] = []
    if gtype == "Polygon" and coords:
        rings.append([(float(x), float(y)) for x, y, *rest in coords[0]])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append([(float(x), float(y)) for x, y, *rest in poly[0]])
    return rings


def path_from_rings(rings: list[list[tuple[float, float]]], xmin: float, ymax: float) -> str:
    parts = []
    for ring in rings:
        if not ring:
            continue
        coords = [f"{x - xmin:.1f},{ymax - y:.1f}" for x, y in ring]
        parts.append("M" + "L".join(coords) + "Z")
    return "".join(parts)


@st.cache_data(show_spinner=False)
def load_areas(areas_path: str) -> tuple[list[dict], dict[str, str], dict[str, float]]:
    data = json.load(open(areas_path))
    raw = []
    xs, ys = [], []
    aliases: dict[str, str] = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        rings = geometry_rings(feature.get("geometry") or {})
        if not rings:
            continue
        name = feature_name(props)
        for ring in rings:
            for x, y in ring:
                xs.append(x)
                ys.append(y)
        for key in ["parish_name", "NAME_ALT", "Parname_ne", "parish_nam", "name"]:
            value = props.get(key)
            if value not in (None, ""):
                aliases.setdefault(norm_place(value), name)
        aliases.setdefault(norm_place(name), name)
        raw.append((name, props, rings))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    areas = []
    for name, props, rings in raw:
        areas.append(
            {
                "name": name,
                "path": path_from_rings(rings, xmin, ymax),
                "language": props.get("parish_language") or props.get("language") or "",
            }
        )
    bounds = {"x": 0, "y": 0, "width": xmax - xmin, "height": ymax - ymin}
    return areas, aliases, bounds


@st.cache_data(show_spinner=False)
def load_place_lookup(p_pl_path: str, places_path: str) -> pd.DataFrame:
    p_pl = pd.read_parquet(p_pl_path)
    places = pd.read_parquet(places_path)
    place_by_id = places.set_index("pl_id")

    def county_name(row: pd.Series) -> str:
        if row.get("type") == "county":
            return row.get("name", "")
        parent = row.get("par_id")
        if parent in place_by_id.index:
            return place_by_id.loc[parent, "name"]
        return ""

    places = places.copy()
    places["county"] = places.apply(county_name, axis=1)
    p_places = p_pl.merge(places[["pl_id", "name", "type", "county"]], on="pl_id", how="left")
    return p_places.rename(columns={"name": "place_name", "type": "place_type"})


@st.cache_data(show_spinner=False)
def read_local_parquet(path: str) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    columns = pd.read_parquet(parquet_path, engine="pyarrow").columns.tolist()
    usecols = [column for column in REQUIRED_COLUMNS if column in columns]
    if "p_id" not in usecols or "word_in_english" not in usecols or "standard_lemma" not in usecols:
        raise ValueError("The parquet file must include at least p_id, word_in_english, and standard_lemma.")
    return pd.read_parquet(parquet_path, columns=usecols, engine="pyarrow")


def add_area(df: pd.DataFrame, aliases: dict[str, str], p_pl_path: str, places_path: str) -> pd.DataFrame:
    df = df.copy()
    if "area_name" in df.columns:
        df["area_name"] = df["area_name"].map(norm_place).map(aliases).fillna(df["area_name"])
        return df
    if "place_name" not in df.columns:
        places = load_place_lookup(p_pl_path, places_path)
        df = df.merge(places[["p_id", "place_name", "county"]], on="p_id", how="left")
    df["area_name"] = df["place_name"].map(norm_place).map(aliases)
    return df[df["area_name"].notna()].copy()


def categorize_variants(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["standard_lemma_clean"] = df["standard_lemma"].map(clean)
    df = df[df["standard_lemma_clean"].ne("")].copy()
    counts = df["standard_lemma_clean"].value_counts()
    labels = counts.head(top_n).index.astype(str).tolist()
    top_set = set(labels)
    df["variant"] = df["standard_lemma_clean"].where(df["standard_lemma_clean"].isin(top_set), "Other")
    if (df["variant"] == "Other").any() and "Other" not in labels:
        labels.append("Other")
    return df, labels


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    area_counts = (
        df.groupby(["area_name", "variant"], as_index=False)
        .agg(mentions=("p_id", "size"), poems=("p_id", "nunique"))
        .sort_values(["area_name", "mentions"], ascending=[True, False])
    )
    summary = (
        df.groupby("variant", as_index=False)
        .agg(mentions=("p_id", "size"), poems=("p_id", "nunique"), areas=("area_name", "nunique"))
        .sort_values("mentions", ascending=False)
    )
    total = summary["mentions"].sum()
    summary["share"] = summary["mentions"] / total if total else 0
    return area_counts, summary


def variation_stats(area_counts: pd.DataFrame) -> dict[str, float | int | str]:
    if area_counts.empty:
        return {"areas": 0, "mentions": 0, "variant_count": 0, "cramers_v": ""}
    pivot = area_counts.pivot_table(index="area_name", columns="variant", values="mentions", aggfunc="sum", fill_value=0)
    cramers_v = ""
    if pivot.shape[0] > 1 and pivot.shape[1] > 1:
        chi2, _, _, _ = chi2_contingency(pivot)
        n = pivot.to_numpy().sum()
        k = min(pivot.shape[0] - 1, pivot.shape[1] - 1)
        if n and k:
            cramers_v = round(math.sqrt(chi2 / (n * k)), 3)
    return {
        "areas": int(pivot.shape[0]),
        "mentions": int(pivot.to_numpy().sum()),
        "variant_count": int(pivot.shape[1]),
        "cramers_v": cramers_v,
    }


def color_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def render_svg_map(areas: list[dict], bounds: dict, area_counts: pd.DataFrame, labels: list[str]) -> str:
    palette = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}
    counts_by_area: dict[str, dict[str, int]] = {}
    for row in area_counts.itertuples(index=False):
        counts_by_area.setdefault(row.area_name, {})[row.variant] = int(row.mentions)
    max_total = max((sum(values.values()) for values in counts_by_area.values()), default=1)
    paths = []
    for area in areas:
        counts = counts_by_area.get(area["name"], {})
        total = sum(counts.values())
        if total:
            dominant = max(counts.items(), key=lambda item: item[1])[0]
            alpha = 0.24 + 0.68 * math.log1p(total) / math.log1p(max_total)
            fill = color_to_rgba(palette.get(dominant, "#8a8a8a"), min(alpha, 0.95))
            title = f"{area['name']} | {dominant} | {total} mentions"
        else:
            fill = "rgba(235,232,224,0.5)"
            title = f"{area['name']} | no matches"
        paths.append(
            f'<path d="{area["path"]}" fill="{fill}" stroke="#74706a" stroke-width="0.55" '
            f'vector-effect="non-scaling-stroke"><title>{html.escape(title)}</title></path>'
        )
    legend = "".join(
        f'<span class="legend-item"><i style="background:{palette.get(label, "#8a8a8a")}"></i>{html.escape(label)}</span>'
        for label in labels
    )
    return f"""
    <style>
      .lv-map-wrap {{ border:1px solid #e1ded8; height:72vh; min-height:520px; overflow:hidden; background:#fff; }}
      .lv-map {{ width:100%; height:100%; display:block; }}
      .lv-legend {{ display:flex; flex-wrap:wrap; gap:8px 14px; margin:10px 0 0; font:12px ui-monospace, monospace; }}
      .legend-item {{ display:inline-flex; align-items:center; gap:6px; color:#3f3b35; }}
      .legend-item i {{ display:inline-block; width:12px; height:12px; border:1px solid rgba(0,0,0,.15); }}
    </style>
    <div class="lv-map-wrap">
      <svg class="lv-map" viewBox="{bounds['x']} {bounds['y']} {bounds['width']} {bounds['height']}" role="img">
        {''.join(paths)}
      </svg>
    </div>
    <div class="lv-legend">{legend}</div>
    """


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("Lexical Variation Pipeline")
st.caption("Use the bundled project corpus, group by English concept, then map regional standard-lemma variation.")


DEFAULT_INPUTS = {
    "concept_input": "smith",
    "aliases_input": "blacksmith\nsmith's",
    "match_mode_input": "Contains cleaned concept",
    "top_n_input": 8,
}


def reset_inputs() -> None:
    st.session_state["concept_input"] = ""
    st.session_state["aliases_input"] = ""
    st.session_state["match_mode_input"] = DEFAULT_INPUTS["match_mode_input"]
    st.session_state["top_n_input"] = DEFAULT_INPUTS["top_n_input"]
    st.session_state.pop("active_params", None)


for key, value in DEFAULT_INPUTS.items():
    st.session_state.setdefault(key, value)

with st.sidebar:
    st.header("Input")
    with st.form("feature_form"):
        concept_value = st.text_input("English concept", key="concept_input")
        aliases_value = st.text_area("Extra aliases", key="aliases_input", height=90)
        match_mode_value = st.selectbox(
            "Matching",
            ["Contains cleaned concept", "Exact cleaned concept"],
            key="match_mode_input",
        )
        top_n_value = st.slider("Top standard lemma variants", 3, 15, key="top_n_input")
        submit_col, clear_col = st.columns(2)
        generate_clicked = submit_col.form_submit_button("Generate", type="primary")
        clear_clicked = clear_col.form_submit_button("Clear", on_click=reset_inputs)
    st.header("Bundled data")
    st.caption(f"Corpus: `{DEFAULT_WORD_ANALYSIS.name}`")
    st.caption(f"Place IDs: `{DEFAULT_P_PL.name}` + `{DEFAULT_PLACES.name}`")
    st.caption(f"Areas: `apps/assets/{DEFAULT_AREAS.name}`")

if clear_clicked:
    st.info("Inputs cleared. Enter an English concept and click Generate.")
    st.stop()

if generate_clicked:
    st.session_state["active_params"] = {
        "concept": concept_value,
        "aliases_text": aliases_value,
        "match_mode": match_mode_value,
        "top_n": top_n_value,
    }

if "active_params" not in st.session_state:
    st.info("Enter an English concept in the left panel and click Generate.")
    st.stop()

active = st.session_state["active_params"]
concept = active["concept"]
aliases_text = active["aliases_text"]
match_mode = active["match_mode"]
top_n = active["top_n"]

try:
    progress = st.progress(0, text="Starting lexical variation pipeline...")
    aliases = concept_aliases(concept, aliases_text)
    matcher = concept_matcher(aliases, match_mode)
    progress.progress(8, text="Loading bundled area geometry...")
    areas, area_aliases, bounds = load_areas(str(DEFAULT_AREAS))

    with st.spinner("Running pipeline..."):
        progress.progress(20, text="Reading bundled word_analysis.parquet...")
        data = read_local_parquet(str(DEFAULT_WORD_ANALYSIS))
        progress.progress(42, text="Normalizing English translations...")
        data["english_key"] = data["word_in_english"].map(norm_key)
        progress.progress(58, text="Filtering tokens by semantic concept...")
        matched = data[data["word_in_english"].map(matcher)].copy()
        progress.progress(72, text="Attaching place IDs and area polygons...")
        matched = add_area(matched, area_aliases, str(DEFAULT_P_PL), str(DEFAULT_PLACES))
        matched = matched.drop_duplicates([col for col in ["p_id", "v_id", "v_pos", "w_pos"] if col in matched.columns])
        progress.progress(86, text="Grouping standard lemma variants...")
        matched, labels = categorize_variants(matched, top_n)
        progress.progress(94, text="Aggregating regional variation...")
        area_counts, summary = aggregate(matched)
        stats = variation_stats(area_counts)
        progress.progress(100, text="Done.")
except Exception as exc:
    st.error(str(exc))
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Mentions", f"{stats['mentions']:,}")
metric_cols[1].metric("Areas", f"{stats['areas']:,}")
metric_cols[2].metric("Variants", f"{stats['variant_count']:,}")
metric_cols[3].metric("Cramer's V", stats["cramers_v"] or "n/a")

st.write("Matched aliases:", ", ".join(f"`{alias}`" for alias in aliases))

map_tab, summary_tab, counts_tab, tokens_tab = st.tabs(["Map", "Variant summary", "Area counts", "Matched tokens"])

with map_tab:
    if area_counts.empty:
        st.warning("No regional matches found.")
    else:
        st.components.v1.html(render_svg_map(areas, bounds, area_counts, labels), height=760, scrolling=False)

with summary_tab:
    st.dataframe(summary, use_container_width=True)
    st.download_button("Download variant summary CSV", csv_bytes(summary), f"{norm_key(concept) or 'feature'}_variant_summary.csv")

with counts_tab:
    st.dataframe(area_counts, use_container_width=True)
    st.download_button("Download area counts CSV", csv_bytes(area_counts), f"{norm_key(concept) or 'feature'}_area_counts.csv")

with tokens_tab:
    token_cols = [col for col in REQUIRED_COLUMNS + ["place_name", "county", "area_name", "variant", "english_key"] if col in matched.columns]
    st.dataframe(matched[token_cols], use_container_width=True)
    st.download_button("Download matched tokens CSV", csv_bytes(matched[token_cols]), f"{norm_key(concept) or 'feature'}_matched_tokens.csv")
