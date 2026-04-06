import pandas as pd

# ─────────────────────────────────────────────
# LOAD DATA
# Change this path to your exported CSV file
# To export: Power BI Desktop → Transform Data
#            → Home → Close & Apply
#            OR just export the table to CSV
# ─────────────────────────────────────────────
df = pd.read_csv("job_salary_prediction_dataset.csv")

# Rename columns to be easier to work with
df.columns = [
    "job_title", "experience_years", "education_level",
    "skills_count", "industry", "company_size",
    "location", "remote_work", "certifications", "salary"
]

print("=" * 60)
print("  DATASET OVERVIEW")
print("=" * 60)
print(f"  Total rows       : {len(df):,}")
print(f"  Unique job titles: {df['job_title'].nunique()}")
print(f"  Columns          : {list(df.columns)}")
print()

# ─────────────────────────────────────────────
# DEFINE ROLE GROUPS
# ─────────────────────────────────────────────
DATA_AI_ROLES = [
    "AI Engineer",
    "Data Scientist",
    "Data Analyst",
    "Machine Learning Engineer"
]

ENG_ROLES = [
    "Cloud Engineer",
    "DevOps Engineer",
    "Software Engineer",
    "Frontend Developer"
]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def print_section(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_measure(name, value, fmt="$"):
    if fmt == "$":
        print(f"  {name:<35} ${value:>12,.0f}")
    elif fmt == "#":
        print(f"  {name:<35} {value:>13,.0f}")

def get_group_df(roles):
    return df[df["job_title"].isin(roles)].copy()

def total_jobs(group_df):
    return len(group_df)

def avg_salary(group_df):
    return group_df["salary"].mean()

def top_salary(group_df):
    """Max of avg salary per job title — same as DAX MAXX over VALUES(job_title)"""
    return group_df.groupby("job_title")["salary"].mean().max()

def salary_gap(group_df):
    """Gap between highest and lowest avg salary per job title"""
    per_role = group_df.groupby("job_title")["salary"].mean()
    return per_role.max() - per_role.min()

def experience_uplift(group_df):
    """Avg salary for experience >= 15 minus avg salary for experience <= 2"""
    senior = group_df[group_df["experience_years"] >= 15]["salary"].mean()
    entry  = group_df[group_df["experience_years"] <= 2]["salary"].mean()
    return senior - entry

def education_uplift(group_df):
    """Avg salary PhD minus avg salary High School"""
    phd  = group_df[group_df["education_level"] == "PhD"]["salary"].mean()
    hs   = group_df[group_df["education_level"] == "High School"]["salary"].mean()
    return phd - hs

def remote_premium(group_df):
    """Avg salary Remote(Yes) minus avg salary On-Site(No)"""
    remote  = group_df[group_df["remote_work"] == "Yes"]["salary"].mean()
    onsite  = group_df[group_df["remote_work"] == "No"]["salary"].mean()
    return remote - onsite


# ─────────────────────────────────────────────
# DATA & AI DASHBOARD MEASURES
# ─────────────────────────────────────────────
print_section("DATA & AI DASHBOARD MEASURES")

dai = get_group_df(DATA_AI_ROLES)

print_measure("Data & AI - Total Jobs",        total_jobs(dai),        fmt="#")
print_measure("Data & AI - Avg Salary",        avg_salary(dai))
print_measure("Data & AI - Top Salary",        top_salary(dai))
print_measure("Data & AI - Salary Gap",        salary_gap(dai))
print_measure("Data & AI - Experience Uplift", experience_uplift(dai))
print_measure("Data & AI - Education Uplift",  education_uplift(dai))
print_measure("Data & AI - Remote Premium",    remote_premium(dai))

print()
print("  --- Per Role Breakdown ---")
per_role_dai = dai.groupby("job_title")["salary"].agg(["mean", "count"]).rename(
    columns={"mean": "Avg Salary", "count": "Total Jobs"}
).sort_values("Avg Salary", ascending=False)
per_role_dai["Avg Salary"] = per_role_dai["Avg Salary"].map("${:,.0f}".format)
print(per_role_dai.to_string())

print()

# ─────────────────────────────────────────────
# ENGINEERING DASHBOARD MEASURES
# ─────────────────────────────────────────────
print_section("ENGINEERING DASHBOARD MEASURES")

eng = get_group_df(ENG_ROLES)

print_measure("Eng - Total Jobs",        total_jobs(eng),        fmt="#")
print_measure("Eng - Avg Salary",        avg_salary(eng))
print_measure("Eng - Top Salary",        top_salary(eng))
print_measure("Eng - Salary Gap",        salary_gap(eng))
print_measure("Eng - Experience Uplift", experience_uplift(eng))
print_measure("Eng - Education Uplift",  education_uplift(eng))
print_measure("Eng - Remote Premium",    remote_premium(eng))

print()
print("  --- Per Role Breakdown ---")
per_role_eng = eng.groupby("job_title")["salary"].agg(["mean", "count"]).rename(
    columns={"mean": "Avg Salary", "count": "Total Jobs"}
).sort_values("Avg Salary", ascending=False)
per_role_eng["Avg Salary"] = per_role_eng["Avg Salary"].map("${:,.0f}".format)
print(per_role_eng.to_string())

print()

# ─────────────────────────────────────────────
# EXPECTED VALUES (from DAX queries we ran)
# Used to auto-validate Python results
# ─────────────────────────────────────────────
print_section("AUTO VALIDATION")

EXPECTED = {
    # Data & AI
    "dai_total_jobs"        : 83234,
    "dai_avg_salary"        : 151436,   # approx
    "dai_top_salary"        : 173498,   # AI Engineer
    "dai_salary_gap"        : 53607,    # AI Eng - Data Analyst
    # Engineering
    "eng_total_jobs"        : 83084,
    "eng_avg_salary"        : 144000,   # approx
    "eng_top_salary"        : 152103,   # Cloud Engineer
    "eng_salary_gap"        : 19450,    # Cloud - Frontend
}

def check(label, actual, expected, tolerance=0.01):
    diff_pct = abs(actual - expected) / expected
    status = "✅ PASS" if diff_pct <= tolerance else "❌ FAIL"
    print(f"  {status}  {label:<35} actual={actual:>10,.0f}  expected≈{expected:>10,}")

check("Data & AI - Total Jobs",    total_jobs(dai),  EXPECTED["dai_total_jobs"],  tolerance=0.001)
check("Data & AI - Top Salary",    top_salary(dai),  EXPECTED["dai_top_salary"])
check("Data & AI - Salary Gap",    salary_gap(dai),  EXPECTED["dai_salary_gap"])
check("Eng - Total Jobs",          total_jobs(eng),  EXPECTED["eng_total_jobs"],  tolerance=0.001)
check("Eng - Top Salary",          top_salary(eng),  EXPECTED["eng_top_salary"])
check("Eng - Salary Gap",          salary_gap(eng),  EXPECTED["eng_salary_gap"])

print()
print("=" * 60)
print("  HOW TO EXPORT THE CSV FROM POWER BI")
print("=" * 60)
print("""
  Option 1 — From Power BI Desktop:
    1. Go to Home tab → Transform Data
    2. In Power Query Editor, right-click the table
    3. Select 'Export' → Save as CSV
    4. Place the CSV next to this script and run it

  Option 2 — From a visual:
    1. Right-click any table visual in your report
    2. Export data → Underlying data
    3. Save as CSV and rename to:
       job_salary_prediction_dataset.csv
""")