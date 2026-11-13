"""
Dataset Generator
=================
Generates a realistic, challenging labelled dataset for phishing detection.

Key design principles to avoid trivial 100% accuracy:
  1. Ambiguous URLs — phishing sites that look legitimate (HTTPS, short URL)
  2. Legitimate URLs — that look suspicious (long, many params, subdomains)
  3. Borderline cases — URLs that are hard to classify
  4. Noise injection — random variation in features
  5. Class noise     — 5% label flipping to simulate real-world label errors
"""

import random
import pandas as pd
from pathlib import Path
from feature_extractor import extract_features, FEATURE_NAMES

random.seed(42)

# ── Legitimate URLs (some deliberately look suspicious) ───────────────────────
LEGIT_URLS = [
    # Normal legit
    "https://www.google.com",
    "https://www.github.com/trending",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://www.amazon.com/s?k=laptop",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.linkedin.com/feed",
    "https://www.reddit.com/r/netsec",
    "https://docs.python.org/3/library/re.html",
    "https://www.bbc.com/news/technology",
    "https://www.apple.com/iphone",
    "https://www.microsoft.com/en-us/windows",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.netflix.com/browse",
    "https://www.paypal.com/us/home",
    "https://www.dropbox.com/login",
    "https://www.adobe.com/products/photoshop.html",
    "https://www.kaggle.com/datasets",
    "https://pytorch.org/docs/stable/index.html",
    # Legitimate but look suspicious (long URLs, many params)
    "https://accounts.google.com/signin/v2/identifier?continue=https://mail.google.com",
    "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=4765445b",
    "https://www.amazon.com/dp/B08N5WRWNW?ref=ppx_yo2ov_dt_b_product_details&th=1",
    "https://www.ebay.com/itm/324123456789?hash=item4b7f2e1234&var=513204556789",
    "https://support.apple.com/en-us/HT201209?utm_source=google&utm_medium=cpc",
    "https://account.live.com/password/reset?mkt=en-US&uiflavor=web&id=10",
    "https://secure.bankofamerica.com/login/sign-in/signOnV2Screen.go",
    "https://www.wellsfargo.com/online-banking/login/?LOB=CONS",
    "https://signin.ebay.com/ws/eBayISAPI.dll?SignIn&ru=https://www.ebay.com",
    "https://auth.services.adobe.com/en_US/index.html#/natural-login?callback=",
    # Legitimate with subdomains
    "https://mail.google.com/mail/u/0/#inbox",
    "https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
    "https://calendar.google.com/calendar/u/0/r/month",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://portal.azure.com/#home",
    "https://console.aws.amazon.com/console/home",
    "https://app.slack.com/client/workspace",
    "https://teams.microsoft.com/l/channel/19:channel",
    "https://classroom.google.com/c/courseid",
    "https://business.facebook.com/latest/home",
]

# ── Phishing URLs (some deliberately look legitimate) ─────────────────────────
PHISHING_URLS = [
    # Obvious phishing
    "http://paypa1-secure-login.com/verify?user=victim",
    "http://192.168.1.1/login/account-update.php",
    "http://apple-id-verify.support/unlock?token=abc123",
    "http://secure-banking-update.tk/login",
    "http://amazon-account-suspended.ml/signin",
    "http://login.microsoft-support-alert.com/password-reset",
    "http://facebook-security.cf/confirm-identity",
    "http://netflix-billing-update.xyz/account/update",
    "http://google-account-locked.tk/signin/verify",
    "http://12.34.56.78/login/banking",
    # Phishing with HTTPS (harder to detect)
    "https://paypal-secure.support/account/verify",
    "https://apple-verify.site/id/unlock",
    "https://microsoft-login.xyz/oauth/authorize",
    "https://amazon-security.club/account/suspended",
    "https://google-accounts.site/signin/challenge",
    "https://secure-netflix.top/billing/update",
    "https://facebook-login.xyz/checkpoint/confirm",
    "https://instagram-verify.site/account/secure",
    "https://twitter-support.top/account/verify",
    "https://ebay-account.site/signin/verify",
    # Combosquatting (hard cases)
    "https://paypal-support-center.com/help/account",
    "https://apple-id-support.com/account/reset",
    "https://microsoft-account-verify.com/signin",
    "https://amazon-prime-support.com/account",
    "https://google-account-recovery.com/signin",
    "https://secure.login-paypal.com/verify",
    "https://signin.amazon-secure.com/ap/signin",
    "https://accounts.google-login.net/signin",
    "https://login.microsoft-secure.net/account",
    "https://secure.facebook-login.net/checkpoint",
    # Subdomain spoofing
    "https://paypal.com.phishing-site.net/login",
    "https://apple.com.id-verify.tk/account",
    "https://microsoft.com.support-alert.ml/login",
    "https://amazon.com.account-verify.cf/signin",
    "https://google.com.account-locked.ga/verify",
]

# ── Borderline / ambiguous URLs ───────────────────────────────────────────────
BORDERLINE_LEGIT = [
    "https://www.paypal.com/signin?country.x=US&locale.x=en_US",
    "https://account.microsoft.com/account/manage",
    "https://appleid.apple.com/account/manage",
    "https://accounts.google.com/ServiceLogin?service=mail",
    "https://www.amazon.com/gp/sign-in.html",
]

BORDERLINE_PHISH = [
    "https://secure-paypal-login.com/account",
    "https://apple-id-login.com/verify",
    "https://microsoft-secure-login.com/account",
    "https://amazon-account-login.com/signin",
    "https://google-secure-verify.com/account",
]


def _add_noise(url: str) -> str:
    """Add slight URL variation."""
    noises = [
        lambda u: u + f"&ref={random.randint(1,999)}",
        lambda u: u + f"?session={random.randint(10000,99999)}",
        lambda u: u,
        lambda u: u,  # 50% chance no change
    ]
    return random.choice(noises)(url)


def _flip_label(label: int, rate: float = 0.05) -> int:
    """Add 5% label noise to simulate real-world annotation errors."""
    return 1 - label if random.random() < rate else label


def generate_dataset(n_legit: int = 500, n_phish: int = 500,
                     output_path: str = None) -> pd.DataFrame:
    """
    Generate a challenging, realistic phishing detection dataset.
    Includes ambiguous cases and label noise to prevent trivial accuracy.
    """
    rows = []

    # ── Legitimate URLs ───────────────────────────────────────────────────────
    for i in range(n_legit):
        # Mix: 70% normal legit, 20% borderline legit, 10% from legit pool
        r = random.random()
        if r < 0.70:
            base = LEGIT_URLS[i % len(LEGIT_URLS)]
        elif r < 0.90:
            base = BORDERLINE_LEGIT[i % len(BORDERLINE_LEGIT)]
        else:
            base = LEGIT_URLS[random.randint(0, len(LEGIT_URLS)-1)]

        url = _add_noise(base)
        try:
            feats = extract_features(url)
            row   = {name: feats[name] for name in FEATURE_NAMES}
            row["label"] = _flip_label(0)
            row["url"]   = url
            rows.append(row)
        except Exception:
            continue

    # ── Phishing URLs ─────────────────────────────────────────────────────────
    for i in range(n_phish):
        # Mix: 60% obvious phishing, 30% hard phishing (HTTPS), 10% borderline
        r = random.random()
        if r < 0.60:
            base = PHISHING_URLS[i % len(PHISHING_URLS)]
        elif r < 0.90:
            base = PHISHING_URLS[15 + (i % (len(PHISHING_URLS) - 15))]  # harder HTTPS phishing
        else:
            base = BORDERLINE_PHISH[i % len(BORDERLINE_PHISH)]

        url = _add_noise(base)
        try:
            feats = extract_features(url)
            row   = {name: feats[name] for name in FEATURE_NAMES}
            row["label"] = _flip_label(1)
            row["url"]   = url
            rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[+] Dataset saved → {output_path}  ({len(df)} rows, "
              f"{(df['label']==1).sum()} phishing, {(df['label']==0).sum()} legit)")

    return df


if __name__ == "__main__":
    generate_dataset(
        n_legit=500,
        n_phish=500,
        output_path=str(Path(__file__).parent.parent / "data" / "phishing_dataset.csv"),
    )
