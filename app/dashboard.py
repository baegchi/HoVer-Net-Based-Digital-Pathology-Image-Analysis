"""Interactive dashboard for the digital pathology analyzer.

    streamlit run app/dashboard.py

Not a medical device - research/educational prototype only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from hovernet.checkpoint import CheckpointError, find_checkpoint  # noqa: E402
from hovernet.engine import Analyzer  # noqa: E402
from hovernet.overlay import colorize_types, type_legend  # noqa: E402
from hovernet.readers import ArrayReader, SlideReader, open_reader  # noqa: E402
from pipeline import report as report_mod  # noqa: E402
from pipeline import stats as stats_mod  # noqa: E402

st.set_page_config(
    page_title="Digital Pathology Analyzer",
    page_icon="🔬",
    layout="wide",
)

DISCLAIMER = (
    "**Research and educational prototype.** Not a medical device and not "
    "validated for clinical or diagnostic use."
)

CSS = """
<style>
.block-container { padding-top: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.7rem; }
.legend-row { display: flex; gap: 1.2rem; flex-wrap: wrap; align-items: center;
              font-size: 0.85rem; margin: 0.25rem 0 0.5rem 0; }
.legend-swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block;
                 margin-right: 5px; vertical-align: middle; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- session state
@st.cache_resource(show_spinner="Loading HoVer-Net checkpoint ...")
def get_analyzer(checkpoint: str, device: str, batch_size: int, min_area: int, stride: int):
    """One model instance is shared across reruns - loading it is expensive."""
    return Analyzer(
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        min_area=min_area,
        stride=stride,
    )


@st.cache_data(show_spinner=False)
def analyze_cached(path: str, config: tuple, tissue_only: bool = True):
    """Cache analyses by file content hash so reruns are instant."""
    analyzer = get_analyzer(*config)
    result = analyzer.analyze(path, stride=config[-1], tissue_only=tissue_only)
    statistics = stats_mod.compute(result)
    return result, statistics


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def upload_signature(uploaded) -> str:
    return fingerprint(uploaded.getvalue()) if uploaded is not None else "none"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    try:
        default_ckpt = str(find_checkpoint())
    except CheckpointError:
        default_ckpt = ""
    checkpoint = st.text_input("Checkpoint", value=default_ckpt, help="Path to a .pth checkpoint")
    device = st.selectbox("Device", ["auto", "cuda", "cpu"])
    batch_size = st.select_slider("Batch size", [1, 2, 4, 8, 16], value=8)
    stride = st.slider("Tile stride (px)", 32, 128, 64, 8, help="Lower = more overlap, slower")
    min_area = st.slider("Minimum nucleus area (px)", 5, 200, 20, 5)
    tissue_only = st.checkbox(
        "Skip background on slides",
        value=True,
        help="Skip windows that contain no tissue. Large speed-up on whole-slide images.",
    )

    st.header("Input")
    source = st.radio("Source", ["Upload", "Sample images", "Sample slide"], horizontal=True)

    uploaded = None
    chosen_path: str | None = None
    if source == "Upload":
        uploaded = st.file_uploader(
            "Pathology image or slide",
            type=["png", "jpg", "jpeg", "tif", "tiff", "svs", "ndpi", "scn"],
        )
    else:
        sub = "slides" if source == "Sample slide" else "images"
        base = ROOT / "samples" / sub
        options = sorted(p for p in base.glob("*") if p.is_file())
        if options:
            chosen = st.selectbox("File", [p.name for p in options])
            chosen_path = str(base / chosen)
        else:
            st.info(f"No sample files in {base}. Run `python scripts/fetch_assets.py`.")
    st.divider()
    st.caption(DISCLAIMER)


# ------------------------------------------------------------------- materialise
@st.cache_data(show_spinner=False)
def load_upload(name: str, data: bytes) -> str:
    """Write an uploaded file to the OS temp dir once and return its path."""
    import tempfile

    path = Path(tempfile.gettempdir()) / "pathology_uploads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(data):
        path.write_bytes(data)
    return str(path)


input_path = chosen_path
if uploaded is not None:
    input_path = load_upload(uploaded.name, uploaded.getvalue())

if not input_path:
    st.title("Digital Pathology Analyzer")
    st.markdown(DISCLAIMER)
    st.info(
        "Upload a histopathology image (PNG/TIFF) or a whole-slide image (SVS/NDPI) "
        "in the sidebar to start. Nuclei are segmented with HoVer-Net and classified "
        "into neoplastic, inflammatory, connective and dead cells."
    )
    st.stop()

# ------------------------------------------------------------------ run analysis
st.title("Digital Pathology Analyzer")

try:
    config = (checkpoint, device, batch_size, min_area, stride)
    with st.spinner("Segmenting nuclei ..."):
        result, statistics = analyze_cached(input_path, config, tissue_only)
except CheckpointError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Analysis failed: {exc}")
    st.stop()

# ------------------------------------------------------------------- rendering
try:
    reader = open_reader(input_path)
    is_slide = isinstance(reader, SlideReader)
    if is_slide:
        image = reader.thumbnail(max_side=min(result.width, 2048))
        reader.close()
        sy = image.shape[0] / result.height
        sx = image.shape[1] / result.width
        for n in result.nuclei:
            n.centroid_y *= sy
            n.centroid_x *= sx
            n.contour = np.round(n.contour * np.array([sx, sy])).astype(np.int32)
    else:
        image = reader.array
    overlay = colorize_types(image, result.nuclei)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not render the overlay: {exc}")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Original")
    st.image(image, use_container_width=True)
with right:
    st.subheader("Segmentation overlay")
    st.image(overlay, use_container_width=True)

# legend
legend_html = "".join(
    f'<span class="legend-row"><span class="legend-swatch" style="background:#{c[0]:02X}{c[1]:02X}{c[2]:02X}"></span>'
    f"{name} <b>{count}</b></span>"
    for name, c, count in type_legend(result)
)
st.markdown(legend_html, unsafe_allow_html=True)
st.caption("Outlines mark instance boundaries; fill colour is the predicted cell type.")

# ------------------------------------------------------------------- metrics
st.subheader("Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nuclei detected", f"{statistics['nuclei_total']:,}")
c2.metric("Density", f"{statistics['nuclei_per_mm2']:,.0f}", help="nuclei per mm²")
c3.metric("Field area", f"{statistics['field_area_mm2']:.2f}", help="mm²")
c4.metric("Mean nucleus area", f"{statistics['area_px']['mean']:.0f}", help="pixels")
c5.metric("Runtime", f"{result.inference_seconds + result.postproc_seconds:.1f}s")

with st.expander("Method details", expanded=False):
    st.json(
        {
            "image": result.summary()["image_size"],
            "tiles": result.summary()["tiles"],
            "parameters": result.summary()["parameters"],
            "timing": {
                "tile_read_s": round(result.read_seconds, 2),
            "slide_tissue_fraction": round(result.tissue_fraction, 4),
                "gpu_forward_s": round(result.inference_seconds, 2),
                "post_processing_s": round(result.postproc_seconds, 2),
            },
            "microns_per_pixel": {
                "value": statistics["microns_per_pixel"],
                "source": statistics["microns_per_pixel_source"],
            },
            "model": result.summary()["model"],
            "unclassified_nuclei": statistics["unclassified"],
        }
    )

# ------------------------------------------------------------------- charts
chart_tab, table_tab, files_tab = st.tabs(["Cell types", "Nuclei table", "Downloads"])

with chart_tab:
    rows = [(name, row) for name, row in statistics["by_type"].items() if row["count"] > 0]
    if rows:
        frame = pd.DataFrame(
            [{"type": name, "count": row["count"], "share_%": row["fraction"] * 100} for name, row in rows]
        )
        left_c, right_c = st.columns(2)
        with left_c:
            st.bar_chart(frame.set_index("type")[["count"]], color="#2E86C1", height=280)
        with right_c:
            st.bar_chart(frame.set_index("type")[["share_%"]], color="#E67E22", height=280)
    else:
        st.info("No nuclei above the minimum area threshold - try lowering it in the sidebar.")

    st.markdown("**Nucleus size distribution**")
    sizes = [n.area for n in result.nuclei]
    if sizes:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.hist(sizes, bins=min(30, max(5, len(sizes) // 3)), color="#2E86C1", edgecolor="white")
        ax.set_xlabel("nucleus area (px)")
        ax.set_ylabel("count")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)

with table_tab:
    frame = pd.DataFrame(
        [
            {
                "id": n.id,
                "type": n.type_name,
                "confidence": round(n.type_prob, 3),
                "area_px": n.area,
                "centroid_y": round(n.centroid_y, 1),
                "centroid_x": round(n.centroid_x, 1),
                "bbox": f"{n.bbox_y0},{n.bbox_x0} - {n.bbox_y1},{n.bbox_x1}",
            }
            for n in result.nuclei
        ]
    )
    if frame.empty:
        st.info("No nuclei detected.")
    else:
        type_filter = st.multiselect(
            "Filter by cell type", sorted(frame["type"].unique()), default=sorted(frame["type"].unique())
        )
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        view = frame[frame["type"].isin(type_filter) & (frame["confidence"] >= min_conf)]
        st.caption(f"{len(view):,} of {len(frame):,} nuclei shown")
        st.dataframe(view, use_container_width=True, height=420)

with files_tab:
    stem = Path(input_path).stem
    json_text = json.dumps(report_mod.to_json(result, statistics), indent=2)
    csv_bytes = report_mod.write_csv(result, Path(st.session_state.setdefault("_tmp", "/tmp")) / f"{stem}.csv")
    md_text = report_mod.markdown_report(result, statistics)

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("JSON", json_text, f"{stem}_analysis.json", "application/json")
    d2.download_button("CSV", Path(csv_bytes).read_bytes(), f"{stem}_nuclei.csv", "text/csv")
    d3.download_button("Report (MD)", md_text, f"{stem}_report.md", "text/markdown")
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    d4.download_button("Overlay (PNG)", buf.getvalue(), f"{stem}_overlay.png", "image/png")

    st.markdown("**Report preview**")
    st.markdown(md_text)