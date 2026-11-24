"""
make_synthetic_data.py
=======================
Regenerates the three raw workbooks so that the SAME synthetic people
appear across files with realistic partial overlap (mirroring how real
membership data is scattered across systems of record maintained by
different people). This is what makes the downstream record-linkage,
feature engineering, and renewal-prediction steps meaningful.

Design:
  - A master pool of 523 "active" people is created first (full identity:
    name, address, city/state/zip, phone, email).
  - The dues roster ("2026 -2027" tab) and the EV-subscription rosters are
    built by SAMPLING from that same master pool (plus a smaller number of
    brand-new people never in the active roster — e.g. someone who mailed
    in a check before their record was added to the roster, or a bounced
    linkage due to a name typo), so cross-file joins recover real overlap.
  - Deceased members and Special Cases are separate identity pools (as in
    the real data, these rarely re-appear in the current dues roster).

Run once:  python -m src.make_synthetic_data
Overwrites data/raw/*.xlsx in place.
"""
from __future__ import annotations

import random
import datetime
from pathlib import Path

from faker import Faker
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

fake = Faker()
Faker.seed(7)
random.seed(7)

AZ_CITIES = ["Tempe", "Phoenix", "Scottsdale", "Mesa", "Chandler", "Gilbert",
             "Fountain Hills", "Paradise Valley", "Glendale", "Peoria",
             "Sun Lakes", "Queen Creek", "Sedona", "Prescott", "Flagstaff"]
OUT_OF_STATE = [("San Diego", "CA"), ("Tucson", "AZ"), ("Portland", "OR"),
                ("Santa Fe", "NM"), ("Denver", "CO"), ("Seattle", "WA")]
DISCIPLINES = [
    "Physics", "Chemistry", "Biology", "Mathematics", "English",
    "History", "Psychology", "Sociology", "Electrical Engineering",
    "Mechanical Engineering", "Civil Engineering", "Computer Science",
    "Economics", "Political Science", "Anthropology", "Nursing",
    "Education", "Music", "Art History", "Theater and Film",
    "Geography", "Philosophy", "Journalism", "Public Health",
    "Social Work", "Biomedical Informatics", "Astronomy", "Geology",
]
STATUS_NOTES = ["Has requested removal from all email lists", None, None, None, None]

HEADER_FILL = PatternFill(start_color="8C1D40", end_color="8C1D40", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def random_date(start_year, end_year):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return start + datetime.timedelta(days=random.randint(0, delta))


def asu_id():
    return 1000000000 + random.randint(100000, 999999)


def phone():
    return f"{random.choice(['480', '602', '623'])}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"


def city_state_zip():
    if random.random() < 0.90:
        city = random.choice(AZ_CITIES)
        state = "AZ"
        zip_code = f"85{random.randint(200, 399)}"
    else:
        city, state = random.choice(OUT_OF_STATE)
        zip_code = fake.zipcode()
    if random.random() < 0.15:
        zip_code = f"{zip_code}-{random.randint(1000, 9999)}"
    return city, state, zip_code


def email_for(first, last, domain_weight=0.85):
    if random.random() < domain_weight:
        local = f"{first.lower()}.{last.lower()}".replace(" ", "")
        return f"{local}@asu.edu"
    return fake.free_email()


def write_header(ws, headers):
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=i, value=h)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="center")


def autosize(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def weighted_sample_without_replacement(population, weights, k):
    """Efraimidis-Spirakis weighted reservoir sampling without replacement."""
    k = min(k, len(population))
    keyed = [(random.random() ** (1.0 / max(w, 1e-6)), item) for item, w in zip(population, weights)]
    keyed.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in keyed[:k]]


def make_person():
    first = fake.first_name()
    last = fake.last_name()
    city, state, zip_code = city_state_zip()
    return {
        "asu_id": asu_id(),
        "first_name": first,
        "last_name": last,
        "address": fake.street_address(),
        "city": city,
        "state": state,
        "zip": zip_code,
        "phone": phone(),
        "email": email_for(first, last),
    }


# ---------------------------------------------------------------------------
# Master identity pool for "active" members — this is the backbone every
# other file samples from, so names/zips really do join across files.
# ---------------------------------------------------------------------------
N_ACTIVE = 523
active_pool = [make_person() for _ in range(N_ACTIVE)]
REFERENCE_YEAR = 2026
STICKY_DISCIPLINES = {"Physics", "Chemistry", "Biology", "Mathematics", "Computer Science",
                      "Electrical Engineering", "Mechanical Engineering"}
for p in active_pool:
    retirement = random_date(1998, 2026)
    joined = retirement + datetime.timedelta(days=random.randint(0, 900))
    p["discipline"] = random.choice(DISCIPLINES)
    p["retirement_date"] = retirement
    p["joined_ec"] = joined
    p["note"] = random.choice(STATUS_NOTES)

    # --- Latent renewal propensity (0-1): drives who shows up in the dues /
    # EV-subscription rosters below, so the synthetic data carries a REAL,
    # learnable relationship between member attributes and renewal, instead
    # of being pure noise. Recently retired, in-state, STEM/sticky-discipline
    # members renew more; long-retired, out-of-state, opted-out members less.
    years_since_retirement = (datetime.date(REFERENCE_YEAR, 11, 24) - retirement).days / 365.25
    score = 0.6
    score -= 0.018 * min(years_since_retirement, 25)        # recency effect
    score += 0.15 if p["state"] == "AZ" else -0.10           # in-state effect
    score += 0.10 if p["discipline"] in STICKY_DISCIPLINES else 0.0
    score -= 0.20 if p["note"] else 0.0                      # opted-out members rarely renew
    score += random.gauss(0, 0.12)                           # irreducible noise
    p["renewal_propensity"] = min(max(score, 0.03), 0.97)

# Separate, non-overlapping identity pools
N_DECEASED = 231
deceased_pool = [make_person() for _ in range(N_DECEASED)]
N_SPECIAL = 440
special_pool = [make_person() for _ in range(N_SPECIAL)]

# =================================================================
# FILE 1: Regular_Members_and_Historical_List_SYNTHETIC.xlsx
# =================================================================
wb1 = Workbook()
wb1.remove(wb1.active)

ws = wb1.create_sheet("Active Regular Members")
headers = ["#", "ASU ID", "First Name", "Last Name", "Address", "City", "State",
           "Zip", "Phone", "Email", "Discipline", "Joined EC", "Retirement Date", "Updated"]
write_header(ws, headers)
for p in active_pool:
    note = p["note"]
    row = [
        "=ROW()-ROW($A$1)",
        p["asu_id"], p["first_name"], p["last_name"], p["address"], p["city"], p["state"], p["zip"],
        p["phone"], p["email"],
        note if note else p["discipline"],
        note if note else p["joined_ec"],
        None if note else p["retirement_date"],
        None if note else random_date(2024, 2026),
    ]
    ws.append(row)
autosize(ws, [4, 12, 12, 14, 24, 14, 6, 11, 13, 26, 24, 12, 14, 12])

ws = wb1.create_sheet("Deceased Regular Members")
headers = ["ASU ID", "Deceased Date", "First Name", "Last Name", "Address",
           "City", "State", "Zip", "E-mail", "Department", "Updated"]
write_header(ws, headers)
for p in deceased_pool:
    deceased = random_date(2005, 2026)
    id_val = p["asu_id"] if random.random() > 0.05 else "Unknown"
    row = [id_val, deceased, p["first_name"], p["last_name"], p["address"], p["city"], p["state"],
           p["zip"], p["email"], random.choice(DISCIPLINES), None]
    ws.append(row)
autosize(ws, [12, 14, 12, 14, 24, 14, 6, 11, 26, 24, 10])

ws = wb1.create_sheet("Special Cases")
headers = ["#", "ASU ID", "First Name", "Last Name", "Address", "City", "State", "Zip", "Phone", "Email"]
write_header(ws, headers)
for i, p in enumerate(special_pool, start=1):
    row = [i, p["asu_id"], p["first_name"], p["last_name"], p["address"], p["city"], p["state"],
           p["zip"], p["phone"], p["email"]]
    ws.append(row)
autosize(ws, [4, 12, 12, 14, 24, 14, 6, 11, 13, 26])

wb1.save(RAW_DIR / "Regular_Members_and_Historical_List_SYNTHETIC.xlsx")
print("Saved File 1 (Regular Members) with a stable active-member identity pool.")

# =================================================================
# FILE 2: Membership_Dues_-_EV_Subscriptions_SYNTHETIC.xlsx
# =================================================================
wb2 = Workbook()
wb2.remove(wb2.active)

# Tab "2026-2027": simple roster, ~95% drawn from the active pool (the vast
# majority of people on the dues roster are already active members), plus a
# few brand-new people (just-joined, not yet folded into Active Regular Members).
ws = wb2.create_sheet("2026-2027")
headers = ["ASU ID", "First Name", "Last Name", "Address", "City", "State", "Zip", "Phone", "Email", "Updated"]
write_header(ws, headers)
N1 = 519
n_from_pool1 = min(495, N1)
weights1 = [p["renewal_propensity"] + 0.3 for p in active_pool]  # most people appear on SOME roster
sample1 = weighted_sample_without_replacement(active_pool, weights1, n_from_pool1)
extra1 = [make_person() for _ in range(N1 - len(sample1))]
for p in sample1 + extra1:
    row = [p["asu_id"], p["first_name"], p["last_name"], p["address"], p["city"], p["state"], p["zip"],
           p["phone"], p["email"], random_date(2024, 2026)]
    ws.append(row)
autosize(ws, [12, 12, 14, 24, 14, 6, 11, 13, 26, 12])

# Tab "2026 -2027" (note trailing space quirk preserved from the real file):
# this is the one with dues + EV subscription status — the key renewal signal.
# ~90% of these rows are active-pool members (people who DID pay this cycle),
# the rest are new/unmatched people.
ws = wb2.create_sheet("2026 -2027")
headers = ["ASU ID", "First Name", "Last Name", "Address", "City", "State", "Zip",
           "Phone", "Email", "Membership Fee ", "Date Paid ", "EV Subscription ", "Date Paid", "Updated"]
write_header(ws, headers)
N2 = 521
n_renewing = min(380, N2)
weights2 = [p["renewal_propensity"] for p in active_pool]
renewing_members = weighted_sample_without_replacement(active_pool, weights2, n_renewing)
extra2 = [make_person() for _ in range(N2 - len(renewing_members))]
for p in renewing_members + extra2:
    propensity = p.get("renewal_propensity", 0.5)
    fee = 25 if random.random() < (0.5 + 0.4 * propensity) else 0
    fee_paid = random_date(2025, 2026) if fee else None
    ev_sub = "Yes" if random.random() < (0.25 + 0.35 * propensity) else "No"
    ev_paid = random_date(2025, 2026) if ev_sub == "Yes" else None
    row = [p["asu_id"], p["first_name"], p["last_name"], p["address"], p["city"], p["state"], p["zip"],
           p["phone"], p["email"], fee, fee_paid, ev_sub, ev_paid, random_date(2024, 2026)]
    ws.append(row)
autosize(ws, [12, 12, 14, 24, 14, 6, 11, 13, 26, 14, 12, 14, 12, 12])

wb2.save(RAW_DIR / "Membership_Dues_-_EV_Subscriptions_SYNTHETIC.xlsx")
print("Saved File 2 (Membership Dues / EV Subscriptions roster) with active-pool overlap.")

# =================================================================
# FILE 3: EV_Subscriptions_-_CURRENT_SYNTHETIC.xlsx
# =================================================================
wb3 = Workbook()
wb3.remove(wb3.active)


def ev_sheet(ws, people, issue_labels):
    headers = ["First Name", "Last Name", "Address", "City", "State", "Zip",
               " Date Paid", " Amount Paid", "Check No.", "Subscription Details",
               "Status"] + issue_labels + ["Mailed/Picked Up"]
    write_header(ws, headers)
    for p in people:
        amount = random.choice([20, 25, 30])
        status = random.choice(["Active", "Active", "Active", "Expired"])
        check_no = random.randint(1000, 9999) if random.random() > 0.3 else "Cash"
        row = [p["first_name"], p["last_name"], p["address"], p["city"], p["state"], p["zip"],
               random_date(2025, 2026), amount, check_no,
               f"{len(issue_labels)}-issue subscription", status]
        row += [random.choice(["X", "X", "", ""]) for _ in issue_labels]
        row.append(random.choice(["Mailed", "Picked Up"]))
        ws.append(row)
    widths = [12, 14, 24, 14, 6, 11, 11, 12, 10, 20, 9] + [5] * len(issue_labels) + [13]
    autosize(ws, widths)


# Current EV subscribers are mostly a subset of the most-engaged active members.
weights3 = [p["renewal_propensity"] for p in active_pool]
current_subs = weighted_sample_without_replacement(active_pool, weights3, 50) + [make_person() for _ in range(7)]
ws = wb3.create_sheet("Current EV Subscriptions")
ev_sheet(ws, current_subs, ["EV40", "EV39", "EV38", "EV37", "EV36", "EV35", "EV34"])

new_cycle_subs = weighted_sample_without_replacement(active_pool, weights3, 15) + [make_person() for _ in range(3)]
ws = wb3.create_sheet("EV Subscriptions (2026-27)")
ev_sheet(ws, new_cycle_subs, ["EV41", "EV40", "EV39", "EV38", "EV37", "EV36", "EV35"])

wb3.save(RAW_DIR / "EV_Subscriptions_-_CURRENT_SYNTHETIC.xlsx")
print("Saved File 3 (EV Subscriptions) with active-pool overlap.")

print("\nAll synthetic files regenerated with realistic cross-file identity overlap.")
