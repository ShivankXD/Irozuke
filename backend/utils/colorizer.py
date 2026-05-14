"""
Colorizer dispatch layer for Irozuke AI.
Supports: manga_v2 (MangaColorization v2 GAN), stylize (AnimeColorDeOldify Sketch), mock
"""

from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

MODELS_DIR   = Path(__file__).parent.parent / "models"
MANGA_V2_DIR = Path(__file__).parent.parent / "manga_colorization_v2"
_loaded_models = {}


# ── Mock colorizer ────────────────────────────────────────────────────────────

def _colorize_pil(img):
    img = img.convert("RGB")
    r, g, b = img.split()
    r = r.point(lambda i: min(255, int(i * 1.10)))
    g = g.point(lambda i: min(255, int(i * 0.98)))
    b = b.point(lambda i: min(255, int(i * 0.82)))
    return ImageEnhance.Sharpness(Image.merge("RGB", (r, g, b))).enhance(1.3)

def _mock_image(input_path, output_path, **_):
    _colorize_pil(Image.open(input_path)).save(output_path, format="PNG")

def _mock_pdf(input_path, output_path, **_):
    import pypdfium2 as pdfium
    pdf   = pdfium.PdfDocument(str(input_path))
    pages = [_colorize_pil(page.render(scale=2).to_pil()) for page in pdf]
    if pages:
        pages[0].save(output_path, format="PDF", save_all=True, append_images=pages[1:])


# ── AnimeColorDeOldify (Sketch model) ────────────────────────────────────────

DEOLDIFY_DIR        = Path(__file__).parent.parent / "anime_deoldify"
DEOLDIFY_WEIGHTS    = MODELS_DIR / "deoldify_sketch.pth"
DEOLDIFY_WEIGHTS_NAME = "deoldify_sketch"   # stem only — fastai appends nothing

_deoldify_visualizer = None


def _load_deoldify_sketch():
    """Lazy-load and cache the AnimeColorDeOldify Sketch ModelImageVisualizer."""
    global _deoldify_visualizer
    if _deoldify_visualizer is not None:
        return _deoldify_visualizer

    import sys
    deoldify_module = str(DEOLDIFY_DIR)
    if deoldify_module not in sys.path:
        sys.path.insert(0, deoldify_module)

    if not DEOLDIFY_WEIGHTS.exists():
        raise FileNotFoundError(
            f"AnimeColorDeOldify Sketch weights not found at {DEOLDIFY_WEIGHTS}\n"
            "Download from: https://www.dropbox.com/s/lykykhvpy9byb7u/"
            "JG5yF2bRBdpEJweytyvBSz3Qu6jcg8cfZ5kcFYGY.pth?dl=1"
        )

    print("[Irozuke] Loading AnimeColorDeOldify Sketch model...")
    # This AnimeColorDeOldify fork does not ship device.py / device_id.py.
    # fastai 1.x picks up the GPU automatically — no explicit device setup needed.

    from deoldify.visualize import get_artistic_image_colorizer
    # root_folder must contain a 'models/' subfolder with the .pth file
    vis = get_artistic_image_colorizer(
        root_folder=MODELS_DIR.parent,          # backend/ — models/ is a child
        weights_name=DEOLDIFY_WEIGHTS_NAME,
        results_dir=str(MODELS_DIR.parent / "outputs" / "deoldify_tmp"),
        render_factor=30,                       # 30 × 16 = 480 px — good balance
    )
    _deoldify_visualizer = vis
    print("[Irozuke] AnimeColorDeOldify Sketch ready ✓")
    return _deoldify_visualizer


def _deoldify_sketch_colorize_pil(pil_img: Image.Image) -> Image.Image:
    """Colorize a single PIL image with the AnimeColorDeOldify Sketch model.

    Strategy:
    - Composite RGBA/palette onto white background (same as manga_v2 fix).
    - Save to a temp file so DeOldify's file-based API can read it.
    - Run inference via get_transformed_image (watermarked=False).
    - Upscale result to original page dimensions with LANCZOS.
    """
    import tempfile, os

    orig_w, orig_h = pil_img.size

    # Composite transparency onto white
    if pil_img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        src = pil_img.convert("RGBA") if pil_img.mode == "P" else pil_img
        bg.paste(src, mask=src.split()[-1] if src.mode in ("RGBA", "LA") else None)
        pil_rgb = bg
    else:
        pil_rgb = pil_img.convert("RGB")

    vis = _load_deoldify_sketch()

    # Write to temp PNG — DeOldify visualizer reads from path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pil_rgb.save(tmp_path, format="PNG")
        from pathlib import Path as _P
        result_pil = vis.get_transformed_image(
            _P(tmp_path), render_factor=30, post_process=True, watermarked=False
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Restore original page resolution
    if result_pil.size != (orig_w, orig_h):
        result_pil = result_pil.resize((orig_w, orig_h), Image.LANCZOS)

    return result_pil


def _deoldify_sketch_image(input_path: Path, output_path: Path, **_):
    pil_in = Image.open(input_path)
    _deoldify_sketch_colorize_pil(pil_in).save(output_path, format="PNG")
    print(f"[Irozuke] deoldify_sketch saved → {output_path.name}")


def _deoldify_sketch_pdf(input_path: Path, output_path: Path, **_):
    """Colorize every page of a PDF with AnimeColorDeOldify Sketch,
    preserving original page dimensions via pypdfium2 (same approach as manga_v2)."""
    import pypdfium2 as pdfium, tempfile, shutil, io

    src_pdf  = pdfium.PdfDocument(str(input_path))
    out_pdf  = pdfium.PdfDocument.new()
    tmp_dir  = Path(tempfile.mkdtemp())

    SCALE = 2
    try:
        for i, src_page in enumerate(src_pdf):
            bitmap   = src_page.render(scale=SCALE)
            pil_page = bitmap.to_pil()

            page_w_pt = src_page.get_width()
            page_h_pt = src_page.get_height()

            colored = _deoldify_sketch_colorize_pil(pil_page)
            print(f"[Irozuke] deoldify_sketch PDF page {i+1}/{len(src_pdf)} done")

            out_page = out_pdf.new_page(page_w_pt, page_h_pt)

            jpeg_buf = io.BytesIO()
            colored.convert("RGB").save(jpeg_buf, format="JPEG", quality=95, subsampling=0)
            jpeg_buf.seek(0)

            pdf_img = pdfium.PdfImage.new(out_pdf)
            pdf_img.load_jpeg(jpeg_buf)
            matrix  = pdfium.PdfMatrix().scale(page_w_pt, page_h_pt)
            pdf_img.set_matrix(matrix)
            out_page.insert_obj(pdf_img)
            out_page.gen_content()

        out_pdf.save(str(output_path))
        print(f"[Irozuke] deoldify_sketch PDF saved → {output_path.name}")
    finally:
        out_pdf.close()
        src_pdf.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Manga Colorization v2 (GAN) ───────────────────────────────────────────────

_manga_v2_instance = None

def _load_manga_v2():
    """Lazy-load and cache the MangaColorizator instance."""
    global _manga_v2_instance
    if _manga_v2_instance is not None:
        return _manga_v2_instance

    import sys, torch, inspect
    repo = str(MANGA_V2_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # Import the denoiser FIRST and fix its hardcoded relative path to absolute
    import denoising.denoiser as _denoiser_mod
    _abs_weights_dir = str(MANGA_V2_DIR / "denoising" / "models") + "/"
    # Patch default arg on FFDNetDenoiser.__init__ so MangaColorizator picks it up
    _orig_init = _denoiser_mod.FFDNetDenoiser.__init__
    def _patched_init(self, _device, _sigma=25,
                      _weights_dir=_abs_weights_dir, _in_ch=3):
        _orig_init(self, _device, _sigma, _weights_dir, _in_ch)
    _denoiser_mod.FFDNetDenoiser.__init__ = _patched_init

    from colorizator import MangaColorizator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator_path = str(MANGA_V2_DIR / "networks" / "generator.zip")

    if not Path(generator_path).exists():
        raise FileNotFoundError(
            f"MangaColorization v2 generator not found at {generator_path}\n"
            "Download from: https://drive.google.com/file/d/1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t/"
        )

    denoiser_path = MANGA_V2_DIR / "denoising" / "models" / "net_rgb.pth"
    if not denoiser_path.exists():
        raise FileNotFoundError(
            f"FFDNet denoiser weights not found at {denoiser_path}\n"
            "Download from: https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/"
        )

    print(f"[Irozuke] Loading MangaColorization v2 on {device}...")
    # MangaColorizator only takes generator_path; extractor_path arg doesn't exist
    _manga_v2_instance = MangaColorizator(device, generator_path)
    print("[Irozuke] MangaColorization v2 ready ✓")
    return _manga_v2_instance


def _manga_v2_colorize_pil(pil_img: Image.Image) -> Image.Image:
    """Run MangaColorizator on a single PIL image, return coloured PIL image.

    Root-cause fixes:
    1. RGBA composited onto white (prevents transparency → algae-green tint).
    2. Pre-scale to ≤ DENOISER_MAX_DIM before denoising.  The FFDNet denoiser
       internally resizes images larger than 1200 px and returns BGR uint8.
       That makes resize_pad() extract the Blue channel instead of luminance,
       which is what causes the orange/sepia tint on alternating pages.
       By pre-scaling ourselves we keep the image ≤ 1200 px so the denoiser
       never triggers its internal downscale, always returning consistent data.
    3. After denoising the returned array is BGR uint8 — we convert it to a
       proper float32 grayscale before handing it to set_image's internal path.
    4. Output is upscaled back to the original page size with Lanczos so the
       PDF is never compressed.
    """
    import cv2
    import numpy as np

    orig_w, orig_h = pil_img.size          # remember original pixel dimensions

    # ── 1. Composite RGBA / palette onto white background ────────────────────
    if pil_img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        src = pil_img.convert("RGBA") if pil_img.mode == "P" else pil_img
        bg.paste(src, mask=src.split()[-1] if src.mode in ("RGBA", "LA") else None)
        pil_rgb = bg
    else:
        pil_rgb = pil_img.convert("RGB")

    # ── 2. Pre-scale to ≤ DENOISER_MAX_DIM so the denoiser never internally
    #       resizes and channel-swaps the image (its internal path returns BGR). ─
    DENOISER_MAX_DIM = 1190          # keep safely under the 1200 px threshold
    img_np = np.array(pil_rgb).astype(np.float32) / 255.0   # RGB float32 [0,1]
    h_np, w_np = img_np.shape[:2]
    if max(h_np, w_np) > DENOISER_MAX_DIM:
        scale_d = DENOISER_MAX_DIM / max(h_np, w_np)
        img_np = cv2.resize(
            img_np,
            (int(w_np * scale_d), int(h_np * scale_d)),
            interpolation=cv2.INTER_AREA,
        )

    colorizer = _load_manga_v2()

    # ── 3. Denoise manually so we can fix the channel order before resize_pad.
    #       The denoiser returns BGR uint8; convert back to RGB float32. ────────
    denoised_bgr_u8 = colorizer.denoiser.get_denoised_image(img_np, sigma=25)
    # denoised_bgr_u8 is uint8 BGR  (see denoising/utils.py variable_to_cv2_image)
    denoised_rgb_f32 = (
        cv2.cvtColor(denoised_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    )

    # ── 4. Run the colourizer with apply_denoise=False (we already denoised). ─
    colorizer.set_image(denoised_rgb_f32, size=576, apply_denoise=False)
    result   = colorizer.colorize()                  # float32 numpy [0,1] RGB
    result_u8 = (result.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    colored  = Image.fromarray(result_u8)

    # ── 5. Restore original page resolution ──────────────────────────────────
    if colored.size != (orig_w, orig_h):
        colored = colored.resize((orig_w, orig_h), Image.LANCZOS)

    return colored


def _manga_v2_image(input_path: Path, output_path: Path, **_):
    pil_in = Image.open(input_path)
    _manga_v2_colorize_pil(pil_in).save(output_path, format="PNG")
    print(f"[Irozuke] manga_v2 saved → {output_path.name}")


def _manga_v2_pdf(input_path: Path, output_path: Path, **_):
    """Colorize every page of a PDF and write the result as a new PDF.

    Uses pypdfium2 to render at 2× scale (≈144 DPI for standard manga) and
    writes the output via pypdfium2 as well, so page dimensions are preserved
    exactly.  PIL's built-in PDF writer is intentionally avoided because it
    silently rescales images to 72 DPI, causing the 'compressed pages' issue.
    """
    import pypdfium2 as pdfium, tempfile, shutil
    import io

    src_pdf  = pdfium.PdfDocument(str(input_path))
    out_pdf  = pdfium.PdfDocument.new()
    tmp_dir  = Path(tempfile.mkdtemp())

    SCALE    = 2          # render resolution multiplier (2 ≈ 144 DPI)
    try:
        for i, src_page in enumerate(src_pdf):
            # ── Render source page ────────────────────────────────────────────
            bitmap   = src_page.render(scale=SCALE)
            pil_page = bitmap.to_pil()          # may be RGBA

            # Page dimensions in PDF points (1 pt = 1/72 inch)
            page_w_pt = src_page.get_width()
            page_h_pt = src_page.get_height()

            # ── Colorize ──────────────────────────────────────────────────────
            colored = _manga_v2_colorize_pil(pil_page)   # restored to pil_page.size
            print(f"[Irozuke] manga_v2 PDF page {i+1}/{len(src_pdf)} done")

            # ── Insert into output PDF at the original point dimensions ────────
            out_page = out_pdf.new_page(page_w_pt, page_h_pt)

            # Encode as JPEG (95 quality, no chroma subsampling) → pypdfium2
            jpeg_buf = io.BytesIO()
            colored.convert("RGB").save(jpeg_buf, format="JPEG", quality=95, subsampling=0)
            jpeg_buf.seek(0)

            pdf_img = pdfium.PdfImage.new(out_pdf)
            pdf_img.load_jpeg(jpeg_buf)
            # Scale image to fill the page exactly (points coordinate space)
            matrix  = pdfium.PdfMatrix().scale(page_w_pt, page_h_pt)
            pdf_img.set_matrix(matrix)
            out_page.insert_obj(pdf_img)
            out_page.gen_content()

        out_pdf.save(str(output_path))
        print(f"[Irozuke] manga_v2 PDF saved → {output_path.name}")
    finally:
        out_pdf.close()
        src_pdf.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mock": {
        "image": _mock_image,
        "pdf":   _mock_pdf,
    },
    "manga_v2": {
        "image": _manga_v2_image,
        "pdf":   _manga_v2_pdf,
    },
    "stylize": {
        "image": _deoldify_sketch_image,
        "pdf":   _deoldify_sketch_pdf,
    },
}

def _resolve(model):
    if model != "auto":
        return model
    # Prefer manga_v2 if weights exist
    if (MANGA_V2_DIR / "networks" / "generator.zip").exists():
        return "manga_v2"
    if DEOLDIFY_WEIGHTS.exists():
        return "stylize"
    return "mock"


# ── Public API ────────────────────────────────────────────────────────────────

def run_colorizer(input_path, output_dir, job_id, model="auto", file_kind="image"):
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _resolve(model)
    if resolved not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{resolved}'. Available: {list(MODEL_REGISTRY)}")
    ext         = ".pdf" if file_kind == "pdf" else ".png"
    output_path = output_dir / f"{job_id}_output{ext}"
    print(f"[Irozuke] Job {job_id} | model={resolved} | kind={file_kind}")
    MODEL_REGISTRY[resolved][file_kind](input_path, output_path)
    return output_path
