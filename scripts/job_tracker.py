"""
Job Application Tracker
Update application status in the CSV database from the command line.
Usage:
    python job_tracker.py list                        # show all jobs
    python job_tracker.py list --status "Not Applied" # filter by status
    python job_tracker.py apply <row_number>          # mark as Applied
    python job_tracker.py status <row_number> <status> --notes "Had interview"
"""

import csv
import sys
import argparse
from pathlib import Path

CSV_FILE = "bangladesh_it_jobs.csv"
STATUSES = ["Not Applied", "Applied", "Interview", "Offer", "Rejected"]


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def list_jobs(rows: list[dict], status_filter: str | None) -> None:
    filtered = [r for r in rows if not status_filter or r["application_status"] == status_filter]
    print(f"\n{'#':<5} {'Title':<40} {'Company':<30} {'Status':<15} {'Deadline'}")
    print("-" * 105)
    for i, row in enumerate(filtered, 1):
        print(f"{i:<5} {row['title'][:38]:<40} {row['company'][:28]:<30} {row['application_status']:<15} {row['deadline']}")
    print(f"\nTotal: {len(filtered)} jobs")


def update_status(rows: list[dict], row_num: int, status: str, notes: str) -> list[dict]:
    if row_num < 1 or row_num > len(rows):
        print(f"Row {row_num} not found. Valid range: 1-{len(rows)}")
        sys.exit(1)
    if status not in STATUSES:
        print(f"Invalid status. Choose from: {', '.join(STATUSES)}")
        sys.exit(1)

    from datetime import date
    rows[row_num - 1]["application_status"] = status
    if notes:
        rows[row_num - 1]["notes"] = notes
    if status == "Applied":
        rows[row_num - 1]["date_applied"] = date.today().isoformat()

    print(f"Updated row {row_num}: {rows[row_num - 1]['title']} @ {rows[row_num - 1]['company']} -> {status}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Bangladesh IT Job Application Tracker")
    sub = parser.add_subparsers(dest="command")

    list_p = sub.add_parser("list", help="List jobs")
    list_p.add_argument("--status", choices=STATUSES, help="Filter by status")
    list_p.add_argument("--file", default=CSV_FILE)

    apply_p = sub.add_parser("apply", help="Mark a job as Applied")
    apply_p.add_argument("row", type=int, help="Row number from list output")
    apply_p.add_argument("--notes", default="", help="Optional notes")
    apply_p.add_argument("--file", default=CSV_FILE)

    status_p = sub.add_parser("status", help="Update status of a job")
    status_p.add_argument("row", type=int, help="Row number from list output")
    status_p.add_argument("status", choices=STATUSES)
    status_p.add_argument("--notes", default="", help="Optional notes")
    status_p.add_argument("--file", default=CSV_FILE)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    csv_path = getattr(args, "file", CSV_FILE)
    if not Path(csv_path).exists():
        print(f"Database not found: {csv_path}")
        print("Run 'python job_scraper.py' first to build the database.")
        sys.exit(1)

    rows = load(csv_path)

    if args.command == "list":
        list_jobs(rows, getattr(args, "status", None))

    elif args.command == "apply":
        rows = update_status(rows, args.row, "Applied", args.notes)
        save(csv_path, rows)

    elif args.command == "status":
        rows = update_status(rows, args.row, args.status, args.notes)
        save(csv_path, rows)


if __name__ == "__main__":
    main()
