import pandas as pd
from pathlib import Path
from typing import Literal

DATA_DIR = Path(__file__).parent.parent / "dataset" / "company_data"

_CSV_PATHS = {
    # Engineering
    "api_dependencies":  DATA_DIR / "engineering" / "api_dependencies.csv",
    "on_call_aug_2023":  DATA_DIR / "engineering" / "on_call_schedule_aug2023.csv",
    "on_call_q4_2023":   DATA_DIR / "engineering" / "on_call_schedule_q4_2023.csv",

    # Finance
    "budget_2023":       DATA_DIR / "finance" / "budget_allocation_2023.csv",
    "revenue_2022":      DATA_DIR / "finance" / "revenue_by_product_2022.csv",
    "revenue_2023":      DATA_DIR / "finance" / "revenue_by_product_2023.csv",
    "vendor_contracts":  DATA_DIR / "finance" / "vendor_contracts_summary.csv",

    # HR
    "employees":         DATA_DIR / "hr" / "employee_directory.csv",
    "offboarding_2023":  DATA_DIR / "hr" / "offboarding_records_2023.csv",

    # Sales
    "csm_history":       DATA_DIR / "sales" / "csm_account_history.csv",
    "customer_health":   DATA_DIR / "sales" / "customer_health_scores_2023.csv",
    "customers":         DATA_DIR / "sales" / "customer_list.csv",
    "deals_q3_2023":     DATA_DIR / "sales" / "deal_pipeline_q3_2023.csv",
    "deals_q4_2023":     DATA_DIR / "sales" / "deal_pipeline_q4_2023.csv",
}

def csv_info(file_name:str):
    '''
    given a file name it returns the info about that csv file
    '''
    df = pd.read_csv(_CSV_PATHS[file_name])
    lines = [
        f"Rows: {len(df)}",
        f"Columns: {list(df.columns)}",
        f"Dtypes: {df.dtypes.astype(str).to_dict()}",
        f"Sample rows: {df.head(3).to_dict(orient='records')}",
    ]
    return "\n".join(lines)


def list_csvs() -> dict[str, str]:
    """
    Returns the available CSV tables and a one-line description of each.
    Use this when you need to know what data is available before querying.
    """
    return {
        # Engineering
        "api_dependencies":  "Internal service-to-service API dependencies — which service depends on which, criticality, purpose.",
        "on_call_aug_2023":  "On-call rotation schedule for August 2023 — week, team, on-call engineer.",
        "on_call_q4_2023":   "On-call rotation schedule for Q4 2023 — week, team, on-call engineer.",

        # Finance
        "budget_2023":       "2023 budget allocation by department — annual budget USD, headcount, owner.",
        "revenue_2022":      "2022 monthly revenue broken out by product (NexusFlow, InsightLens, PulseConnect) plus total.",
        "revenue_2023":      "2023 monthly revenue broken out by product (NexusFlow, InsightLens, PulseConnect) plus total.",
        "vendor_contracts":  "Vendor contracts summary — vendor, category, annual value USD, renewal date, owner.",

        # HR
        "employees":         "Employee directory — name, department, role, manager, start date, status (active/departed_<year>), office location.",
        "offboarding_2023":  "2023 employee offboarding records — name, department, last day, departure type.",

        # Sales
        "csm_history":       "Customer Success Manager assignment history — which CSM was assigned to which customer and when.",
        "customer_health":   "2023 customer health scores — health score, NPS, open tickets, renewal risk, renewal date.",
        "customers":         "Vertexia customer list — company, industry, ARR USD, products, CSM, segment, contract start.",
        "deals_q3_2023":     "Q3 2023 sales pipeline — deal, company, ARR, stage (Closed-Won/Lost/InProgress), products, owner.",
        "deals_q4_2023":     "Q4 2023 sales pipeline — deal, company, ARR, stage (Closed-Won/Lost/InProgress), products, owner.",
    }

def query_csv(
    file_key: str,
    filters: list[dict] | None = None,
    operation: Literal["count", "sum", "mean","rows"] = "count",
    operation_column: str | None = None,
):
    """
    Filter a CSV and aggregate. Call csv_info first to learn the columns and values.

    Args:
        file_key: as returned by list_csvs.
        filters: list of {column, op, value}. op ∈ {"eq", "ne", "gt", "lt", "contains"}.
        operation: "count", "sum" or "mean" a numeric column. also available "rows" which return max 5 rows for that dataframe
        operation_column: required when operation is "sum" or "mean".
    """
    if file_key not in _CSV_PATHS:
        raise ValueError(
            f"unknown file_key '{file_key}'; "
            f"valid keys: {list(_CSV_PATHS.keys())}"
        )
    df = pd.read_csv(_CSV_PATHS[file_key])
    for f in filters or []:
        col = f["column"]
        op  = f["op"]
        val = f["value"]

        if col not in df.columns:
            raise ValueError(
                f"unknown column '{col}' in {file_key}; "
                f"valid columns: {list(df.columns)}"
            )

        if op == "eq":
            df = df[df[col] == val]
        elif op == "ne":
            df = df[df[col] != val]
        elif op == "gt":
            df = df[df[col] > val]
        elif op == "lt":
            df = df[df[col] < val]
        elif op == "contains":
            df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
        else:
            raise ValueError(f"unknown op '{op}'; valid: eq, ne, gt, lt, contains")
    
    if operation == "count":
        return float(len(df))

    if operation == "rows":
        return df.head(5).to_dict(orient="records") 

    if operation in {"sum", "mean"}:
        if operation_column is None:
            raise ValueError(f"operation_column required for operation='{operation}'")
        if operation_column not in df.columns:
            raise ValueError(
                f"unknown operation_column '{operation_column}'; "
                f"valid columns: {list(df.columns)}"
            )
        if operation == "sum":
            return float(df[operation_column].sum())
        
        else:  # mean
            return float(df[operation_column].mean())
