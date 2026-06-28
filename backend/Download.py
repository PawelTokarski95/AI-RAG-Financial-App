import re
from pathlib import Path
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from backend.Tickers import tickers

BASE_PATH = Path("sec-edgar-filings")
SAVE_PATH = Path("processed-filings")

SAVE_PATH.mkdir(exist_ok=True)



def download_filings():
    dl = Downloader("sec_filings", email_address="paweltokarski95@gmail.com")

    for t in tickers:
        dl.get("10-K", t, limit=1)
        print(f"downloaded {t}")



def extract_10k_text(content: str) -> str:


    match = re.search(
        r"<DOCUMENT>\s*<TYPE>10-K.*?<TEXT>(.*?)</TEXT>",
        content,
        re.S | re.I
    )

    return match.group(1) if match else content



def clean_sec_html(soup: BeautifulSoup) -> str:
    for tag in soup.find_all("table", class_="authRefData"):
        tag.decompose()

    for tag in soup.select('[style*="display: none"]'):
        tag.decompose()

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)

    text = " ".join(text.split())

    return text



def process_filings(ticker: str):
    base = BASE_PATH / ticker / "10-K"

    if not base.exists():
        print(f"Missing folder: {base}")
        return

    filing_folder = next(base.iterdir(), None)
    if filing_folder is None:
        print(f"No filings for: {ticker}")
        return

    file_path = next(filing_folder.glob("*.txt"), None)
    if file_path is None:
        print(f"No txt file for: {ticker}")
        return

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    content = extract_10k_text(content)

    soup = BeautifulSoup(content, "html.parser")

    text = clean_sec_html(soup)

    output_path = SAVE_PATH / f"{ticker}_clean.txt"

    with open(output_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)

    print(f"Saved: {output_path}")



for ticker in tickers:
    process_filings(ticker)