"""
VGA — Google Images Scraper
============================
Scrapes Google Images via Selenium for baby skin photos in three classes:
healthy | eczema | chickenpox

Usage (from project root)
-------------------------
    python -m src.pipelines.vga.scrape
    python -m src.pipelines.vga.scrape --class eczema
    python -m src.pipelines.vga.scrape --no-headless
    python -m src.pipelines.vga.scrape --dry-run
"""

import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DIR = Path(__file__).parent
_PROJECT_ROOT = _DIR.parent.parent.parent
_DEFAULT_CONFIG = _DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Google Images backend (Selenium)
# ---------------------------------------------------------------------------

def _make_driver(headless: bool):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


_SKIP_DOMAINS = (
    "gstatic.com", "google.com", "googleapis.com", "ggpht.com",
    "googleusercontent.com", "googlevideo.com",
)


def _extract_image_urls(page_source: str) -> list[str]:
    import html

    raw = re.findall(
        r'https?://[^\s\'"<>\\]+\.(?:jpg|jpeg|png|webp|JPG|JPEG|PNG|WEBP)[^\s\'"<>\\]*',
        page_source,
    )
    seen = set()
    result = []
    for u in raw:
        u = html.unescape(u)
        try:
            u = u.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
        m = re.match(r'(https?://[^\s\'"<>]+\.(?:jpg|jpeg|png|webp))', u, re.IGNORECASE)
        if not m:
            continue
        u = m.group(1)
        if u in seen:
            continue
        if any(d in u for d in _SKIP_DOMAINS):
            continue
        seen.add(u)
        result.append(u)
    return result


def _download_url(url: str, dest: Path, timeout: int = 10) -> bool:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return False
        dest.write_bytes(resp.content)
        if dest.stat().st_size < 5 * 1024:
            dest.unlink()
            return False
        return True
    except Exception:
        if dest.exists():
            dest.unlink()
        return False


def scrape_google(query: str, out_dir: Path, n: int, headless: bool, scroll_pause: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    q_enc = quote_plus(query)

    driver = _make_driver(headless)
    all_urls: list[str] = []
    try:
        start = 0
        step = 100
        stale = 0
        while len(all_urls) < n and stale < 3:
            page_url = (
                f"https://www.google.com/search?q={q_enc}"
                f"&tbm=isch&tbs=isz:m&start={start}"
            )
            driver.get(page_url)
            time.sleep(scroll_pause)

            if start == 0:
                for btn_text in ["Accept all", "Reject all", "I agree"]:
                    try:
                        btns = driver.find_elements(
                            "xpath", f"//button[contains(., '{btn_text}')]"
                        )
                        if btns:
                            btns[0].click()
                            time.sleep(1)
                            break
                    except Exception:
                        pass

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)

            page_urls = _extract_image_urls(driver.page_source)
            before = len(all_urls)
            seen = set(all_urls)
            for u in page_urls:
                if u not in seen:
                    all_urls.append(u)
                    seen.add(u)

            gained = len(all_urls) - before
            if gained < 5:
                stale += 1
            else:
                stale = 0
            start += step
    finally:
        driver.quit()

    log.info("    Found %d candidate URLs across %d page(s)", len(all_urls), start // step)

    downloaded = 0
    existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))

    for url in all_urls[:n]:
        ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        dest = out_dir / f"{existing + downloaded:06d}.{ext}"
        if _download_url(url, dest):
            downloaded += 1
            if downloaded % 50 == 0:
                log.info("      %d / %d downloaded...", downloaded, n)

    return downloaded


# ---------------------------------------------------------------------------
# DuckDuckGo fallback
# ---------------------------------------------------------------------------

def scrape_duckduckgo(query: str, out_dir: Path, n: int) -> int:
    from duckduckgo_search import DDGS

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))

    with DDGS() as ddgs:
        results = list(ddgs.images(keywords=query, max_results=n, size="Medium") or [])

    for r in results:
        url = r.get("image", "")
        if not url:
            continue
        ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        dest = out_dir / f"{existing + downloaded:06d}.{ext}"
        if _download_url(url, dest):
            downloaded += 1

    return downloaded


# ---------------------------------------------------------------------------
# Per-class orchestration
# ---------------------------------------------------------------------------

def scrape_class(class_name: str, class_cfg: dict, cfg: dict, backend: str):
    output_root = _PROJECT_ROOT / cfg["output_dir"]
    raw_dir = output_root / class_cfg["folder"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    queries = class_cfg["queries"]
    per_query = cfg["scraper"]["images_per_query"]
    headless = cfg["scraper"].get("headless", True)
    scroll_pause = cfg["scraper"].get("scroll_pause", 2.0)
    sleep_s = cfg["scraper"].get("sleep_between_queries", 5)

    log.info(
        "Class '%s': %d queries x %d imgs = %d requested",
        class_name, len(queries), per_query, len(queries) * per_query,
    )

    for i, query in enumerate(queries, 1):
        log.info("  [%d/%d] '%s'", i, len(queries), query)
        query_dir = raw_dir / f"q{i:02d}"

        if backend == "duckduckgo":
            count = scrape_duckduckgo(query, query_dir, per_query)
        else:
            count = scrape_google(query, query_dir, per_query, headless, scroll_pause)

        log.info("         -> downloaded %d images", count)
        if i < len(queries):
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VGA skin image scraper")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG))
    p.add_argument(
        "--backend", choices=["google", "duckduckgo"], default=None,
        help="Override backend from config",
    )
    p.add_argument(
        "--class", dest="only_class", default=None,
        choices=["healthy", "eczema", "chickenpox"],
    )
    p.add_argument("--no-headless", action="store_true", help="Show Chrome window")
    p.add_argument("--dry-run", action="store_true", help="Print queries, no download")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    backend = args.backend or cfg["scraper"]["backend"]

    if args.no_headless:
        cfg["scraper"]["headless"] = False

    log.info("Backend: %s | headless: %s", backend, cfg["scraper"].get("headless", True))

    classes = cfg["classes"]
    if args.only_class:
        classes = {args.only_class: classes[args.only_class]}

    for class_name, class_cfg in classes.items():
        log.info("=" * 60)
        log.info("Scraping class: %s", class_name.upper())
        log.info("=" * 60)

        if args.dry_run:
            for q in class_cfg["queries"]:
                print(f"  [{class_name}] {q}")
            continue

        scrape_class(class_name, class_cfg, cfg, backend)

    if not args.dry_run:
        log.info("Done. Run deduplicate next: python -m src.pipelines.vga.deduplicate")


if __name__ == "__main__":
    main()
