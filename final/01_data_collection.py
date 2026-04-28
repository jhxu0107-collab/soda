import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

# ── 1. Coveo API ──────────────────────────────────────────────────
COVEO_URL = "https://commonwealthofpennsylvaniaproductiono8jd9ckm.org.coveo.com/rest/search/v2"
COVEO_ORG = "commonwealthofpennsylvaniaproductiono8jd9ckm"
COVEO_TOKEN = "Bearer xx4e57cda9-3464-437d-9375-b947ca6b72c8"

# ── 2. keywords ─────────────────────────────────────────────────────
keywords = [
    "data center",
    "power plant",
    "AI infrastructure"
]

# ── 3. search：only Governor's Office ──────────────────────────────
def search_governor(keyword, first_result=0, number=50):
    headers = {
        "Authorization": COVEO_TOKEN,
        "Content-Type": "application/json",
        "Origin": "https://www.pa.gov",
        "Referer": "https://www.pa.gov/",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "q": keyword,
        "numberOfResults": number,
        "firstResult": first_result,
        "sortCriteria": "relevancy",
        "searchHub": "All PWP",
        "tab": "default",
        "locale": "en",
        "referrer": "https://www.pa.gov/governor/newsroom",
        "timezone": "America/New_York",
        "aq": "@copapwpagency==\"Governor's Office\""  # 只搜Governor's Office
    }
    params = {"organizationId": COVEO_ORG}
    response = requests.post(COVEO_URL, headers=headers, json=payload, params=params)
    return response.json()

# ── search：all newsroom ───────────────────────────────
def search_all_newsroom(keyword, first_result=0, number=50):
    headers = {
        "Authorization": COVEO_TOKEN,
        "Content-Type": "application/json",
        "Origin": "https://www.pa.gov",
        "Referer": "https://www.pa.gov/",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "q": keyword,
        "numberOfResults": number,
        "firstResult": first_result,
        "sortCriteria": "relevancy",
        "searchHub": "All PWP",
        "tab": "default",
        "locale": "en",
        "referrer": "https://www.pa.gov/governor/newsroom",
        "timezone": "America/New_York"
    }
    params = {"organizationId": COVEO_ORG}
    response = requests.post(COVEO_URL, headers=headers, json=payload, params=params)
    return response.json()

# ── 5 extract full text ───────────────────────────────────────────────────
def get_full_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        contents = soup.find_all("div", class_="text")
        if contents:
            main_content = max(contents, key=lambda x: len(x.get_text()))
            return main_content.get_text(separator=" ", strip=True)
        else:
            paragraphs = soup.find_all("p")
            return " ".join([p.get_text(strip=True) for p in paragraphs])
    except Exception as e:
        print(f"  fail: {url} - {e}")
        return ""

# ── 6. main ─────────────────────────────────────────────────────────
all_results = []

for kw in keywords:

    # ── search：Governor's Office ──────────────────────────────────────
    print(f"\n[Governor's Office] search: {kw}")
    for page in range(4):
        first = page * 50
        data = search_governor(kw, first_result=first, number=50)
        results = data.get("results", [])
        if not results:
            print(f"  {page+1}no results，stop。")
            break
        for item in results:
            title = item.get("title", "")
            link = item.get("clickUri", "")
            excerpt = item.get("excerpt", "")
            if ".pdf" not in link and ".doc" not in link:
                all_results.append({
                    "source": "Governor's Office",
                    "keyword": kw,
                    "title": title,
                    "link": link,
                    "excerpt": excerpt
                })
                print(f"  ✓ {title[:60]}")
        time.sleep(0.5)

    # ── search B：all newsroom ───────────────────────────────────────
    print(f"\n[All Newsroom] search: {kw}")
    for page in range(4):
        first = page * 50
        data = search_all_newsroom(kw, first_result=first, number=50)
        results = data.get("results", [])
        if not results:
            print(f"  {page+1}no results，stop")
            break
        for item in results:
            title = item.get("title", "")
            link = item.get("clickUri", "")
            excerpt = item.get("excerpt", "")
            if "newsroom" in link and ".pdf" not in link and ".doc" not in link:
                all_results.append({
                    "source": "All Newsroom",
                    "keyword": kw,
                    "title": title,
                    "link": link,
                    "excerpt": excerpt
                })
                print(f"  ✓ {title[:60]}")
        time.sleep(0.5)

# ── 7. duplicate & merge ───────────────────────────────────────────────────────
df = pd.DataFrame(all_results)

if df.empty:
    print("\nnot found，chech COVEO_TOKEN。")
else:
    df.drop_duplicates(subset="link", inplace=True)
    print(f"\n find {len(df)} non-duplication，start extracting")

    # ── 8. textract ───────────────────────────────────────────
    full_texts = []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  ({i+1}/{len(df)}) {row['title'][:50]}")
        full_texts.append(get_full_text(row["link"]))
        time.sleep(0.5)

    df["full_text"] = full_texts

    # ── 9. save to Excel ────────────────────────────────────────────────
    df.to_excel("pa_gov_statements.xlsx", index=False)
    print(f"\nfinish！save {len(df)} to pa_gov_statements.xlsx")
    gov_count = len(df[df["source"] == "Governor's Office"])
    news_count = len(df[df["source"] == "All Newsroom"])
    print(f"  - Governor's Office: {gov_count} ")
    print(f"  - All Newsroom: {news_count} ")
