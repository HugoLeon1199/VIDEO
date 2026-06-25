"""Download and cache icon PNG files for video overlay use.

Icons are saved to assets/icons/{name}.png (200x200, white on transparent).
Source: Google Fonts Material Icons CDN (SVG) → converted via cairosvg, or
fallback to a pre-built PNG CDN if cairosvg is not available.
"""

import sys
import subprocess
from pathlib import Path

ICONS_DIR = Path("assets/icons")

# Material Icon name → Google Fonts CDN SVG path
ICON_MAP = {
    "email":     "email",
    "calendar":  "event",
    "clock":     "schedule",
    "phone":     "phone",
    "laptop":    "laptop",
    "checkmark": "check_circle",
    "x-mark":    "cancel",
    "fire":      "local_fire_department",
    "leaf":      "eco",
    "wheat":     "grass",
    "skull":     "skull",
    "heart":     "favorite",
    "star":      "star",
    "sun":       "wb_sunny",
    "moon":      "nightlight_round",
    "mountain":  "terrain",
    "river":     "water",
    "tree":      "park",
    "person":    "person",
    "group":     "group",
}

SVG_URL_TEMPLATE = (
    "https://fonts.gstatic.com/s/i/materialicons/{name}/v1/24px.svg"
)

PNG_SIZE = 200


def _svg_to_png(svg_bytes: bytes, out_path: Path) -> bool:
    """Convert SVG bytes to white-on-transparent PNG via cairosvg."""
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=str(out_path),
            output_width=PNG_SIZE,
            output_height=PNG_SIZE,
        )
        return True
    except ImportError:
        return False


def _svg_to_png_inkscape(svg_path: Path, out_path: Path) -> bool:
    """Fallback: convert SVG to PNG via Inkscape CLI."""
    result = subprocess.run(
        ["inkscape", "--export-type=png", f"--export-filename={out_path}",
         f"--export-width={PNG_SIZE}", f"--export-height={PNG_SIZE}", str(svg_path)],
        capture_output=True,
    )
    return result.returncode == 0 and out_path.exists()


def _download_svg(icon_name: str) -> bytes | None:
    """Download SVG from Google Fonts CDN."""
    import urllib.request
    url = SVG_URL_TEMPLATE.format(name=icon_name)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read()
    except Exception as e:
        print(f"  Download failed ({url}): {e}")
        return None


def _make_white_svg(svg_bytes: bytes) -> bytes:
    """Replace fill colors with white so icons render white on transparent bg."""
    svg_text = svg_bytes.decode("utf-8")
    # Material icons use fill="currentColor" or no fill — set fill to white
    if 'fill="white"' not in svg_text:
        svg_text = svg_text.replace("<svg ", '<svg fill="white" ', 1)
    return svg_text.encode("utf-8")


def download_all(force: bool = False) -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    failed = []

    for alias, material_name in ICON_MAP.items():
        out_path = ICONS_DIR / f"{alias}.png"
        if out_path.exists() and not force:
            skipped += 1
            continue

        print(f"  Downloading: {alias} ({material_name})...")
        svg_bytes = _download_svg(material_name)
        if svg_bytes is None:
            failed.append(alias)
            continue

        svg_bytes = _make_white_svg(svg_bytes)

        if _svg_to_png(svg_bytes, out_path):
            print(f"    Saved: {out_path}")
            ok += 1
        else:
            # Fallback: write SVG to temp file, try inkscape
            tmp_svg = ICONS_DIR / f"{alias}_tmp.svg"
            tmp_svg.write_bytes(svg_bytes)
            if _svg_to_png_inkscape(tmp_svg, out_path):
                tmp_svg.unlink(missing_ok=True)
                print(f"    Saved (inkscape): {out_path}")
                ok += 1
            else:
                tmp_svg.unlink(missing_ok=True)
                print(f"    FAILED (no cairosvg or inkscape): {alias}")
                failed.append(alias)

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {len(failed)} failed")
    if failed:
        print(f"Failed icons: {failed}")
        print("Install cairosvg:  pip install cairosvg")
        print("Or inkscape:       winget install inkscape")


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_all(force=force)
