"""Synthetic stand-in for bl_full_data.csv: same header quirks and category strings, no real people."""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

HEADER = ["", "session_dt", "conversion_dt", "register_date", "session_id", "campaign_id", "page", "auto_city",
          "auto_country", "auto_state", "device_type", "sub1", "sub2", "sub3", "vertical", "business_name",
          "business_type", "credit_score", "industry", "loan_amount", "loan_reason", "monthly_revenue",
          "time_in_business", "business_address", "business_name", "fname", "lname", "cellphone", "address",
          "client_id", "client_name", "payout", "disposition", "disposition_note", "disposition_source"]

PAGES = ["top10us.com/app/business-loans-v6", "top10us.com/app/business-loans-v2", "top10us.com/app/business-loans-v6c"]
DEVICES = ["mobile", "desktop", "tablet"]
BUSINESS_TYPES = ["LLC", "Sole Proprietorship", "S Corporation", "C Corporation", "Other"]
CREDIT = ["Very Poor - Under 550", "Poor (550-599)", "Fair (600-649)", "Good (650-719)", "Excellent (720+)"]
INDUSTRIES = ["construction", "retail_trade", "other", "transport_warehousing", "healthcare_social"]
LOAN = ["$10,000 - $24,999", "$25,000 - $49,999", "$50,000 - $74,999", "$75,000 - $99,999", "$100,000+"]
REASON = ["Business expansion", "Equipment purchase", "Debt consolidation", "Other"]
REVENUE = ["$0 - $9,999", "$10,000 - $19,999", "$20,000 - $49,999", "$50,000 - $99,999", "$100,000 +"]
TIB = ["2+ years", "Less than 6 months", "1-2 years", "6-12 months"]
STATES = [("Houston", "Texas"), ("Miami", "Florida"), ("Atlanta", "Georgia"), ("Fresno", "California")]
FNAMES = ["James", "Maria", "Robert", "Linda", "Alex", "Taylor", "Chris", "Pat"]
LNAMES = ["Smith", "Garcia", "Johnson", "Lee", "Brown"]
# brand → (disposition_source, payout, lead probability)
BRANDS = {"fundera / nerdwallet": ("Fundera API", 65.0, 0.55), "businessloans.com": ("businessloans.com", 112.0, 0.4),
          "xlt": ("xlt", 216.0, 0.35), "lendzi": ("Lendzi API", 60.0, 0.2)}


def make_synthetic_csv(path: Path, days: int = 21, sessions_per_day: int = 150, seed: int = 0) -> Path:
    rng = random.Random(seed)
    start = datetime(2026, 1, 1)
    rows, idx = [], 0
    for day in range(days):
        for s in range(sessions_per_day):
            t0 = start + timedelta(days=day, seconds=rng.randint(0, 86_000))
            registered = rng.random() < 0.5
            sid = f"sess-{day:03d}-{s:04d}"
            base = {
                "session_dt": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "conversion_dt": (t0 + timedelta(seconds=130)).strftime("%Y-%m-%d %H:%M:%S") if registered else "",
                "register_date": (t0 + timedelta(seconds=rng.randint(40, 400))).strftime("%Y-%m-%d %H:%M:%S")
                if registered else "",
                "session_id": sid, "campaign_id": str(120240633562020306 + rng.randint(0, 3) * 1000),
                "page": rng.choice(PAGES), "auto_country": "United States", "device_type": rng.choice(DEVICES),
                "sub1": f"0112{rng.randint(2000, 2020):04d}", "sub2": f"0112{rng.randint(2000, 2005):04d} Ad set",
                "sub3": str(1511143042 + rng.randint(0, 5)), "vertical": "Business Loans",
                "business_name": f"Biz {sid}", "business_type": rng.choice(BUSINESS_TYPES),
                "credit_score": rng.choice(CREDIT), "industry": rng.choice(INDUSTRIES), "loan_amount": rng.choice(LOAN),
                "loan_reason": rng.choice(REASON), "monthly_revenue": rng.choice(REVENUE),
                "time_in_business": rng.choice(TIB), "business_address": "1 Main St, Suite 2",
                "fname": rng.choice(FNAMES), "lname": rng.choice(LNAMES),
                "cellphone": str(rng.choice([305, 713, 404, 559]) * 10_000_000 + rng.randint(0, 9_999_999)),
                "address": "1 Main St",
            }
            city, state = rng.choice(STATES)
            base.update(auto_city=city, auto_state=state)
            brands = rng.sample(list(BRANDS), rng.choice([1, 1, 2])) if registered else [None]
            for brand in brands:
                row = dict(base, client_id="", client_name="", payout="", disposition="", disposition_note="",
                           disposition_source="")
                if brand:
                    source, payout, p_lead = BRANDS[brand]
                    lead = rng.random() < p_lead
                    row.update(client_id=str(700 + list(BRANDS).index(brand)), client_name=brand,
                               payout=f"{payout + rng.choice([0, 0, 5, -5]):.1f}" if lead else "",
                               disposition="Lead" if lead else "Rejected",
                               disposition_source=source if lead else "")
                rows.append([str(idx)] + [row[c] for c in HEADER[1:15]] + [row["business_name"]]
                            + [row[c] for c in HEADER[16:24]] + [row["business_name"]]
                            + [row[c] for c in HEADER[25:]])
                idx += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path
