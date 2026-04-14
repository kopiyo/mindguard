"""
scraper_worker.py
=================
Runs Facebook and Twitter Playwright scraping in a completely separate
Python process to avoid the asyncio conflict with Streamlit on Windows.

Called by the main app via subprocess.run() and communicates via JSON stdout.

Usage (internal):
    python scraper_worker.py facebook https://www.facebook.com/zuck 3
    python scraper_worker.py twitter  https://x.com/elonmusk 3
"""

import sys
import json
import time
import datetime

def scrape_facebook(url: str, months: int) -> list:
    from playwright.sync_api import sync_playwright

    posts = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox","--disable-dev-shm-usage",
                  "--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(4)

            # Dismiss popups
            for selector in [
                '[aria-label="Allow all cookies"]',
                '[data-testid="cookie-policy-manage-dialog-accept-button"]',
                'button:has-text("Accept All")',
                '[aria-label="Close"]',
            ]:
                try:
                    btn = page.locator(selector).first
                    if btn.is_visible(timeout=1500):
                        btn.click()
                        time.sleep(1)
                except Exception:
                    pass

            # Click Posts tab if present
            try:
                tab = page.locator('a:has-text("Posts")').first
                if tab.is_visible(timeout=3000):
                    tab.click()
                    time.sleep(3)
            except Exception:
                pass

            seen = set()
            for scroll_i in range(15):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(2.5)

                extracted = page.evaluate("""() => {
                    const results = [];
                    const seen = new Set();
                    const selectors = [
                        '[data-ad-preview="message"]',
                        '[data-testid="post_message"]',
                        '.userContent',
                        'div[dir="auto"] > span[dir="auto"]',
                        'div[data-ad-comet-preview="message"]',
                    ];
                    for (const sel of selectors) {
                        document.querySelectorAll(sel).forEach(el => {
                            const txt = el.innerText.trim();
                            if (txt.length > 15 && !seen.has(txt)) {
                                seen.add(txt);
                                results.push(txt);
                            }
                        });
                    }
                    return results;
                }""")

                for text in extracted:
                    clean = text.strip()
                    if len(clean) > 15 and clean not in seen:
                        seen.add(clean)
                        dt = (datetime.datetime.utcnow() -
                              datetime.timedelta(days=scroll_i * 5)).isoformat()
                        posts.append({"text": clean, "date": dt, "url": url})

                if len(posts) >= 100:
                    break
        finally:
            browser.close()
    return posts


def scrape_twitter(url: str, months: int) -> list:
    from playwright.sync_api import sync_playwright

    posts = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox","--disable-dev-shm-usage",
                  "--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)

            seen = set()
            for scroll_i in range(20):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(2)

                extracted = page.evaluate("""() => {
                    const results = [];
                    document.querySelectorAll('[data-testid="tweetText"]').forEach(el => {
                        const txt = el.innerText.trim();
                        if (txt.length > 5) results.push(txt);
                    });
                    return [...new Set(results)];
                }""")

                dates = page.evaluate("""() => {
                    const times = [];
                    document.querySelectorAll('time').forEach(el => {
                        times.push(el.getAttribute('datetime') || '');
                    });
                    return times;
                }""")

                for i, text in enumerate(extracted):
                    if text not in seen and len(text) > 5:
                        seen.add(text)
                        dt = (datetime.datetime.utcnow() -
                              datetime.timedelta(days=scroll_i * 2 + i)).isoformat()
                        if i < len(dates) and dates[i]:
                            try:
                                dt = dates[i].replace("Z", "")
                            except Exception:
                                pass
                        posts.append({"text": text, "date": dt, "url": url})

                if len(posts) >= 100:
                    break
        finally:
            browser.close()
    return posts


if __name__ == "__main__":
    platform = sys.argv[1]  # "facebook" or "twitter"
    url      = sys.argv[2]
    months   = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    try:
        if platform == "facebook":
            posts = scrape_facebook(url, months)
        elif platform == "twitter":
            posts = scrape_twitter(url, months)
        else:
            posts = []
        print(json.dumps({"ok": True, "posts": posts}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
