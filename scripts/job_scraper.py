"""
Bangladesh IT Job Scraper
Scrapes job listings from Bdjobs.com and builds a CSV database for tracking applications.
"""

import csv
import time
import random
import logging
import re
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

IT_CATEGORIES = [
    "Software-Engineer",
    "Web-Developer",
    "Mobile-Application-Developer",
    "Network-Engineer",
    "Database-Administrator",
    "IT-Manager",
    "System-Administrator",
    "DevOps-Engineer",
    "Data-Scientist",
    "Cyber-Security",
    "UI-UX-Designer",
    "Project-Manager-IT",
    "QA-Engineer",
    "Machine-Learning-Engineer",
    "Cloud-Engineer",
]


@dataclass
class Job:
    title: str
    company: str
    location: str
    job_type: str
    deadline: str
    posted_date: str
    apply_url: str
    source: str
    category: str
    description_snippet: str
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
            log.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def scrape_bdjobs_category(category: str, max_pages: int = 5) -> list[Job]:
    jobs = []
    base_url = f"https://jobs.bdjobs.com/jobsearch.asp?txtsearch=&fcatId=9&fAreaId=0&fDeadline=0&iPage="

    for page in range(1, max_pages + 1):
        url = f"https://jobs.bdjobs.com/jobsearch.asp?txtsearch={category}&fcatId=9&iPage={page}"
        log.info(f"Scraping bdjobs: {category} page {page}")
        soup = fetch(url)
        if not soup:
            break

        rows = soup.select("div.job-list-item, tr.job_list_row, .norm-jobs-panel")
        if not rows:
            # fallback selector
            rows = soup.find_all("div", class_=re.compile(r"job|listing", re.I))

        for row in rows:
            try:
                title_el = row.select_one("a.jobtitle, .job-title a, h3 a, .title a")
                company_el = row.select_one(".comp-name, .company-name, .org-name")
                location_el = row.select_one(".location, .job-location, .loc")
                deadline_el = row.select_one(".deadline, .apply-deadline, .date")
                link_el = row.select_one("a[href*='jobdetails'], a.jobtitle, .job-title a")

                if not title_el:
                    continue

                apply_url = ""
                if link_el and link_el.get("href"):
                    href = link_el["href"]
                    apply_url = href if href.startswith("http") else f"https://jobs.bdjobs.com/{href}"

                jobs.append(Job(
                    title=title_el.get_text(strip=True),
                    company=company_el.get_text(strip=True) if company_el else "N/A",
                    location=location_el.get_text(strip=True) if location_el else "Bangladesh",
                    job_type="Full Time",
                    deadline=deadline_el.get_text(strip=True) if deadline_el else "N/A",
                    posted_date="N/A",
                    apply_url=apply_url,
                    source="bdjobs.com",
                    category=category,
                    description_snippet="",
                ))
            except Exception as e:
                log.debug(f"Row parse error: {e}")
                continue

        time.sleep(random.uniform(1.5, 3.0))

    return jobs


def scrape_chakri() -> list[Job]:
    jobs = []
    pages = 10
    for page in range(1, pages + 1):
        url = f"https://www.chakri.com/jobs?category=it-telecom&page={page}"
        log.info(f"Scraping chakri.com page {page}")
        soup = fetch(url)
        if not soup:
            break

        cards = soup.select(".job-card, .job-item, article.job, .listing-item")
        for card in cards:
            try:
                title_el = card.select_one("h2 a, h3 a, .job-title a, .title")
                company_el = card.select_one(".company, .employer, .org")
                location_el = card.select_one(".location, .area")
                deadline_el = card.select_one(".deadline, .expire, .date")
                link_el = card.select_one("a[href]")

                if not title_el:
                    continue

                href = link_el["href"] if link_el else ""
                apply_url = href if href.startswith("http") else f"https://www.chakri.com{href}"

                jobs.append(Job(
                    title=title_el.get_text(strip=True),
                    company=company_el.get_text(strip=True) if company_el else "N/A",
                    location=location_el.get_text(strip=True) if location_el else "Bangladesh",
                    job_type="Full Time",
                    deadline=deadline_el.get_text(strip=True) if deadline_el else "N/A",
                    posted_date="N/A",
                    apply_url=apply_url,
                    source="chakri.com",
                    category="IT",
                    description_snippet="",
                ))
            except Exception as e:
                log.debug(f"Chakri row error: {e}")

        time.sleep(random.uniform(1.5, 3.0))

    return jobs


def scrape_linkedin_bd() -> list[Job]:
    """Scrapes public LinkedIn job listings for Bangladesh IT roles."""
    jobs = []
    keywords = ["software engineer", "developer", "IT", "DevOps", "data scientist"]

    for keyword in keywords:
        url = (
            f"https://www.linkedin.com/jobs/search/?keywords={keyword.replace(' ', '%20')}"
            f"&location=Bangladesh&f_TP=1%2C2&f_I=96"
        )
        log.info(f"Scraping LinkedIn: {keyword}")
        soup = fetch(url)
        if not soup:
            continue

        cards = soup.select(".job-search-card, .base-card, .jobs-search__results-list li")
        for card in cards:
            try:
                title_el = card.select_one("h3.base-search-card__title, .job-search-card__title, h3")
                company_el = card.select_one("h4.base-search-card__subtitle, .job-search-card__company-name, h4")
                location_el = card.select_one(".job-search-card__location, .base-search-card__metadata span")
                link_el = card.select_one("a.base-card__full-link, a[href*='linkedin.com/jobs']")

                if not title_el:
                    continue

                apply_url = link_el["href"] if link_el else url

                jobs.append(Job(
                    title=title_el.get_text(strip=True),
                    company=company_el.get_text(strip=True) if company_el else "N/A",
                    location=location_el.get_text(strip=True) if location_el else "Bangladesh",
                    job_type="Full Time",
                    deadline="N/A",
                    posted_date="N/A",
                    apply_url=apply_url,
                    source="linkedin.com",
                    category=keyword,
                    description_snippet="",
                ))
            except Exception as e:
                log.debug(f"LinkedIn row error: {e}")

        time.sleep(random.uniform(2.0, 4.0))

    return jobs


def deduplicate(jobs: list[Job]) -> list[Job]:
    seen = set()
    unique = []
    for job in jobs:
        key = (job.title.lower(), job.company.lower())
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

    # Scrape bdjobs across IT categories
    for category in IT_CATEGORIES:
        jobs = scrape_bdjobs_category(category, max_pages=4)
        log.info(f"  {category}: {len(jobs)} jobs")
        all_jobs.extend(jobs)

    # Scrape chakri.com
    chakri_jobs = scrape_chakri()
    log.info(f"Chakri: {len(chakri_jobs)} jobs")
    all_jobs.extend(chakri_jobs)

    # Scrape LinkedIn public listings
    linkedin_jobs = scrape_linkedin_bd()
    log.info(f"LinkedIn: {len(linkedin_jobs)} jobs")
    all_jobs.extend(linkedin_jobs)

    # Clean up
    all_jobs = deduplicate(all_jobs)
    log.info(f"Total unique jobs: {len(all_jobs)}")

    save_csv(all_jobs, OUTPUT_FILE)
    print(f"\nDone. {len(all_jobs)} unique IT job listings saved to '{OUTPUT_FILE}'")
    print("Open the CSV in Excel or Google Sheets to track your applications.")
    print("Update the 'application_status' column as you apply (Not Applied / Applied / Interview / Offer / Rejected)")


if __name__ == "__main__":
    main()
