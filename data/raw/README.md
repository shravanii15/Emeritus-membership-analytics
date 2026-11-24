# Raw data — synthetic, not real

Every file in `data/raw/` is **100% synthetically generated** by
`src/make_synthetic_data.py` using the `Faker` library with a fixed random
seed. No real person's name, address, email, phone number, or membership
record appears anywhere in this repository.

The generator was built to mirror the *structure* and realistic
data-quality issues of a real university emeritus-association membership
database (inconsistent IDs, mixed zip formats, free-text notes mixed into
data fields, partial cross-file identity overlap) so that the data
engineering, modeling, and analysis in this project are a faithful
simulation of the real problem — without exposing any private information.

## Files & data dictionary

### `Regular_Members_and_Historical_List_SYNTHETIC.xlsx`

| Sheet | Rows | Key columns |
|---|---|---|
| `Active Regular Members` | 523 | ASU ID, First/Last Name, Address, City, State, Zip, Phone, Email, Discipline, Joined EC, Retirement Date, Updated |
| `Deceased Regular Members` | 231 | ASU ID (sometimes `"Unknown"`), Deceased Date, First/Last Name, Address, City, State, Zip, E-mail, Department |
| `Special Cases` | 440 | #, ASU ID, First/Last Name, Address, City, State, Zip, Phone, Email |

### `Membership_Dues_-_EV_Subscriptions_SYNTHETIC.xlsx`

| Sheet | Rows | Key columns |
|---|---|---|
| `2026-2027` | 519 | ASU ID, First/Last Name, Address, City, State, Zip, Phone, Email, Updated |
| `2026 -2027` *(note the trailing-space quirk, preserved from the source system)* | 521 | ASU ID, …, Membership Fee, Date Paid, EV Subscription, Date Paid, Updated |

### `EV_Subscriptions_-_CURRENT_SYNTHETIC.xlsx`

| Sheet | Rows | Key columns |
|---|---|---|
| `Current EV Subscriptions` | 57 | First/Last Name, Address, City, State, Zip, Date Paid, Amount Paid, Check No., Status, per-issue flags (EV34–EV40), Mailed/Picked Up |
| `EV Subscriptions (2026-27)` | 18 | Same structure, issues EV35–EV41 |

## Known, intentional data-quality issues (used to demonstrate data engineering)

- **ASU ID is not a reliable key.** It is sometimes the literal string
  `"Unknown"` and carries no relationship across files.
- **Zip codes mix formats** — plain 5-digit and ZIP+4 (`85287-1234`).
- **Administrative notes sometimes sit in data columns** — e.g. a
  "Discipline" cell that actually contains the free-text note *"Has
  requested removal from all email lists"*.
- **Cross-file identity overlap is partial, not exact** — the dues and
  EV-subscription rosters are weighted samples of the active-member pool
  plus some brand-new people, exactly as a real multi-year, multi-office
  spreadsheet workflow would produce.

See `src/data_cleaning.py` for how each of these is detected and resolved.
