"""
Bangladesh IT Job Scraper
Scrapes IT job listings from LinkedIn (public) and exports to CSV for application tracking.
"""

import csv
import time
import random
import logging
from datetime import datetime
from dataclasses import dataclass, fields
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

OUTPUT_FILE = "bangladesh_it_jobs.csv"

IT_KEYWORDS = [
    "software engineer",
    "web developer",
    "mobile developer",
    "android developer",
    "iOS developer",
    "flutter developer",
    "react developer",
    "node.js developer",
    "python developer",
    "java developer",
    "php developer",
    "DevOps engineer",
    "data scientist",
    "machine learning engineer",
    "data analyst",
    "UI UX designer",
    "QA engineer",
    "network engineer",
    "system administrator",
    "cloud engineer",
    "cybersecurity",
    "database administrator",
    "IT manager",
    "project manager IT",
    "backend developer",
    "frontend developer",
    "fullstack developer",
]


@dataclass
class Job:
    title: str
    company: str
    location: str
    posted_date: str
    apply_url: str
    source: str
    category: str
    application_status: str = "Not Applied"
    notes: str = ""
    date_applied: str = ""
    scraped_at: str = datetime.now().strftime("%Y-%m-%d")


def fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def scrape_linkedin(keyword: str, pages: int = 3) -> list[Job]:
    jobs = []
    for page in range(pages):
        start = page * 25
        url = (
            f"https://www.linkedin.com/jobs/search/"
            f"?keywords={keyword.replace(' ', '%20')}"
            f"&location=Bangladesh&start={start}"
        )
        log.info(f"LinkedIn [{keyword}] page {page + 1}")
        soup = fetch(url)
        if not soup:
            break

        cards = soup.select(".base-card")
        if not cards:
            break

        for card in cards:
            try:
                title_el = card.select_one("h3.base-search-card__title")
                company_el = card.select_one("h4.base-search-card__subtitle")
                location_el = card.select_one("span.job-search-card__location")
                time_el = card.select_one("time")
                link_el = card.select_one("a.base-card__full-link")

                if not title_el:
                    continue

                jobs.append(Job(
                    title=title_el.get_text(strip=True),
                    company=company_el.get_text(strip=True) if company_el else "N/A",
                    location=location_el.get_text(strip=True) if location_el else "Bangladesh",
                    posted_date=time_el["datetime"] if time_el and time_el.get("datetime") else "N/A",
                    apply_url=link_el["href"].split("?")[0] if link_el else url,
                    source="linkedin.com",
                    category=keyword,
                ))
            except Exception as e:
                log.debug(f"Card parse error: {e}")

        time.sleep(random.uniform(2.0, 4.0))

    return jobs


def deduplicate(jobs: list[Job]) -> list[Job]:
    seen = set()
    unique = []
    for job in jobs:
        key = (job.title.lower().strip(), job.company.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def save_csv(jobs: list[Job], filename: str) -> None:
    field_names = [f.name for f in fields(Job)]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for job in jobs:
            writer.writerow({f.name: getattr(job, f.name) for f in fields(Job)})
    log.info(f"Saved {len(jobs)} jobs to {filename}")


def main():
    all_jobs: list[Job] = []

    for keyword in IT_KEYWORDS:
        jobs = scrape_linkedin(keyword, pages=3)
        log.info(f"  '{keyword}': {len(jobs)} jobs")
        all_jobs.extend(jobs)

    all_jobs = deduplicate(all_jobs)
    log.info(f"Total unique jobs: {len(all_jobs)}")

    save_csv(all_jobs, OUTPUT_FILE)
    print(f"\nDone. {len(all_jobs)} unique IT job listings saved to '{OUTPUT_FILE}'")
    print("Open in Excel or Google Sheets to track your applications.")
    print("Update 'application_status': Not Applied / Applied / Interview / Offer / Rejected")


if __name__ == "__main__":
    main()
