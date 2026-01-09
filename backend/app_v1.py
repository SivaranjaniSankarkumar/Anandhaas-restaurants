from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import boto3
import io
import io
import os
import requests
import tempfile
import base64
from matplotlib.backends.backend_pdf import PdfPages
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os


load_dotenv()  
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))  # Backend directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))  # Root directory

app = Flask(__name__)
CORS(app)



BEDROCK_MODEL_ID = "amazon.nova-pro-v1:0"
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"


print(f"SARVAM_API_KEY loaded: {'Yes' if SARVAM_API_KEY else 'No'}")
if SARVAM_API_KEY:
    print(f"API Key starts with: {SARVAM_API_KEY[:10]}...")
else:
    print("API Key is None or empty")

anandhaas_data = None  
last_pdf_data = {"data": None, "title": "", "insights": "", "filename": ""}  


def load_anandhaas_data() -> pd.DataFrame | None:
    """Load data from S3 parquet file"""
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # S3 Configuration
        S3_BUCKET = "anandhaas"
        S3_KEY = "output/part-00000-0012309b-cf19-46df-b074-950f1ec8c616-c000.snappy.parquet"
        
        print(f"Loading parquet file: s3://{S3_BUCKET}/{S3_KEY}")
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        parquet_data = response['Body'].read()
        
        # Read parquet data
        df = pd.read_parquet(io.BytesIO(parquet_data))
        print(f" Loaded {len(df)} records from S3 parquet")
        print(f"Available columns: {list(df.columns)}")
        
        # Convert data types for processing
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Row Total"] = pd.to_numeric(df["Row Total"], errors="coerce")
        
        print(f"Final dataset: {len(df)} records (no rows dropped)")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Sample branches: {df['Branch Name'].unique()[:5]}")
        
        return df
        
    except Exception as e:
        print(f"Cannot load data from S3: {e}")
        return None

def analyze_anandhaas_structure(data: pd.DataFrame) -> dict:
    if data is None or data.empty:
        return {}
    clean_branches = [b for b in data["Branch Name"].dropna().unique()]
    clean_groups = [g for g in data["Group Name"].dropna().unique()]
    clean_categories = [c for c in data["Category"].dropna().unique()]
    
    analysis = {
        "total_records": len(data),
        "branches": clean_branches,
        "groups": clean_groups,
        "categories": clean_categories,
        "date_range": {
            "start": data["Date"].min(),
            "end": data["Date"].max(),
        },
        "revenue_stats": {
            "total": float(data["Row Total"].sum()),
            "avg": float(data["Row Total"].mean()),
            "max": float(data["Row Total"].max()),
            "min": float(data["Row Total"].min()),
        },
    }
    
    # Add optional fields if columns exist
    if "Customer/Vendor Name" in data.columns:
        clean_customers = [c for c in data["Customer/Vendor Name"].dropna().unique()]
        analysis["customers"] = clean_customers
    
    if "SubGroup" in data.columns:
        clean_subgroups = [s for s in data["SubGroup"].dropna().unique()]
        analysis["subgroups"] = clean_subgroups
    
    if "Item/Service Description" in data.columns:
        clean_items = [i for i in data["Item/Service Description"].dropna().unique()]
        analysis["items"] = clean_items
    
    if "Quantity" in data.columns:
        analysis["quantity_stats"] = {
            "total": float(data["Quantity"].sum()),
            "avg": float(data["Quantity"].mean()),
            "max": float(data["Quantity"].max()),
            "min": float(data["Quantity"].min()),
        }
    
    return analysis

def get_ai_plan(query: str, data_analysis: dict) -> dict:
    branches = data_analysis.get("branches", [])
    categories = data_analysis.get("categories", [])
    groups = data_analysis.get("groups", [])
    customers = data_analysis.get("customers", [])
    subgroups = data_analysis.get("subgroups", [])
    items = data_analysis.get("items", [])

    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

        prompt = f"""
Analyze this business query about restaurant sales and create a visualization plan.

Query: "{query}"

Available Data:
- Branches: {branches}
- Categories: {categories}
- Items: {items[:50] if items else []}
- Groups: {groups}
- Customers: {customers[:20] if customers else []}
- SubGroups: {subgroups}

IMPORTANT: There are TWO different columns:
1. "Category" - Food categories like "roast", "biriyani varieties", "coffee", etc.
2. "Item/Service Description" - Specific food items like individual roast items, biriyani items, etc.

Return ONLY valid JSON in this exact format:
{{
  "chart_type": "bar|pie|line|dual_bar",
  "x_axis": "Branch Name|Group Name|Category|Item/Service Description|Customer/Vendor Name|SubGroup|Month|Date",
  "y_axis": "Row Total|Quantity|count|dual",
  "y_axis_secondary": null or "Row Total|Quantity|count",
  "aggregation": "sum|mean|count",
  "aggregation_secondary": null or "sum|mean|count",
  "category_filters": null or [string, ...],
  "item_filters": null or [string, ...],
  "branch_filters": null or [string, ...],
  "group_filters": null or [string, ...],
  "customer_filters": null or [string, ...],
  "subgroup_filters": null or [string, ...],
  "month_filter": null or month_number or [month_numbers],
  "date_filter": null or "YYYY-MM-DD" or ["YYYY-MM-DD", "YYYY-MM-DD"],
  "year_filter": null or year_number or [year_numbers],
  "limit": null or number,
  "title": "chart title",
  "dual_metrics": false or true,
  "comparison_type": null or "monthly" or "metric"
}}

CRITICAL TITLE GENERATION RULES:
- NEVER use "Vivi" or any made-up names in titles
- Use ACTUAL branch names from the available data: {branches}
- If query mentions specific branches, include those EXACT branch names in title
- For branch analysis, use format: "[Branch Name] Revenue Analysis" or "Revenue Analysis by Branch"
- For category analysis, use format: "[Category Name] Analysis" or "Analysis by Category"
- For general queries, use descriptive titles like "Restaurant Revenue Analysis", "Sales Performance Analysis"
- ALWAYS use real data values, never fictional names

CRITICAL DISTINCTION RULES:
- When user says "rava roast item" or "roast item" or "[anything] item" → ALWAYS use item_filters
- When user says "rava roast category" or "roast category" or "[anything] category" → ALWAYS use category_filters
- When user just says "rava roast" or "roast" without specifying item/category:
  * If "rava roast" exists in categories, use category_filters: ["rava roast"]
  * If "rava roast" exists in items, use item_filters: ["rava roast"]
  * If both exist, prefer Category over Item
- CRITICAL: If user explicitly mentions "item" anywhere in the query, ALWAYS use item_filters, NOT category_filters
- CRITICAL: When user says "rava roast item", use item_filters: ["rava roast"] NOT ["Parcel Rava Roast"]
- CRITICAL: Don't add prefixes like "Parcel" unless user specifically mentions them
- CRITICAL: Use the EXACT term the user mentioned - if they say "rava roast", use "rava roast", not "parcel rava roast"

Rules:
- CRITICAL: TOP N QUERIES - Extract limit numbers from queries:
  * "top 5", "top 10", "top 7", "first 5", "best 10" → extract as limit: 5, 10, 7, 5, 10
  * "highest 3", "lowest 5", "bottom 10" → extract as limit: 3, 5, 10
  * "show me 5", "give me 10", "list 7" → extract as limit: 5, 10, 7
  * "Top 10 Roast Items" → extract as limit: 10
  * "Top 5 branches" → extract as limit: 5
  * ANY query with "top [NUMBER]" or "[NUMBER] top" → ALWAYS extract the NUMBER as limit
  * When user specifies a number, ALWAYS extract it as limit
  * If no number specified, set limit: null to show all results
- Extract ALL filters from the query: categories, branches, groups, customers, subgroups, months, dates, and years
- For months: january=1, february=2, march=3, april=4, may=5, june=6, july=7, august=8, september=9, october=10, november=11, december=12
- For dates: extract specific dates ("2024-01-15") or date ranges (["2024-01-01", "2024-01-31"])
- For years: extract year numbers (2023, 2024, etc.)
- Groups are service types: "Parcel" (takeaway/delivery), "Line AC" (dine-in AC), "Line Non AC" (dine-in non-AC)
- Categories are food items like "Biriyani Varieties", "Coffee", "Chappathi Single", etc.
- Branches are locations: "VV", "SPM", "AVR", "RSP", "LMJ", "BRK", "GPM", "SBC", "GKNM"
- CRITICAL: Extract ALL branch names mentioned in query, including variations:
  * "VV branch" or "VV" → "VV"
  * "SPM branch" or "SPM" → "SPM"
  * "SBC branch" or "SBC" → "SBC"
  * "GKNM branch" or "GKNM" → "GKNM"
  * "AVR branch" or "AVR" → "AVR"
  * Look for patterns like "X branch, Y branch, Z branch" or "X and Y and Z branches"
  * Parse comma-separated lists: "VV, SPM, SBC, GKNM" should extract all four branches
- When user mentions "parcel", "takeaway", "delivery" → use group_filters: ["Parcel"]
- When user mentions "dine in AC", "AC hall" → use group_filters: ["Line AC"]
- When user mentions "dine in", "hall", "non AC" → use group_filters: ["Line Non AC"]
- When user mentions "customer", "vendor" → use x_axis: "Customer/Vendor Name" or customer_filters
- When user mentions "subgroup" → use x_axis: "SubGroup" or subgroup_filters
- If user asks "how many" or "count" with quantity → use y_axis: "Quantity" and aggregation: "sum"
- If user asks "how many" or "count" without quantity → use y_axis: "count" and aggregation: "count"
- For "each branch" or "by branch", use x_axis: "Branch Name"
- For "each customer" or "by customer", use x_axis: "Customer/Vendor Name"
- For "each subgroup" or "by subgroup", use x_axis: "SubGroup"
- IMPORTANT: For "distribution", "breakdown", "share", "split", "proportion" → ALWAYS use chart_type: "pie"
- For "comparison", "compare", "vs" → use bar chart
- CRITICAL: When user mentions "by months", "monthly", "month wise", "each month" → use x_axis: "Month" NOT "Posting Date"
- For time/trend with specific dates, use line chart with Posting Date
- For monthly analysis, ALWAYS use x_axis: "Month" and chart_type: "bar"
- For daily analysis over short periods, use x_axis: "Posting Date" and chart_type: "line"
- For yearly comparisons, extract year_filter and use appropriate grouping
- CRITICAL: DYNAMIC DUAL METRICS - Detect ANY combination of two metrics:
  * "revenue and quantity" → y_axis: "Row Total", y_axis_secondary: "Quantity"
  * "sales and count" → y_axis: "Row Total", y_axis_secondary: "count"
  * "quantity and revenue" → y_axis: "Quantity", y_axis_secondary: "Row Total"
  * "average revenue and total quantity" → y_axis: "Row Total", aggregation: "mean", y_axis_secondary: "Quantity", aggregation_secondary: "sum"
  * "total sales and average quantity" → y_axis: "Row Total", aggregation: "sum", y_axis_secondary: "Quantity", aggregation_secondary: "mean"
  * ANY query with "and" between two metrics → set dual_metrics: true, extract both metrics
  * For comparison queries like "February vs March", use dual_metrics: true, comparison_type: "monthly"
- CRITICAL: When user asks for roast count or any category count, include quantity in analysis by using y_axis: "Quantity" if available
- CRITICAL: For category filtering, distinguish between similar items: "roast" should NOT include "rava roast", "dosa" should NOT include "masala dosa" unless specifically mentioned
- Use exact matching first, then word boundary matching to avoid substring confusion
- CRITICAL: Date filtering rules:
  * "today", "yesterday" → extract current/previous date
  * "last week", "this week" → extract date range
  * "January 2024", "Jan 2024" → extract month and year filters
  * "2024" → extract year filter
  * "February 12th", "Feb 12", "12th February" → extract as "02-12" (MM-DD format)
  * "February 12th, 2024" → extract as "2024-02-12"
  * "from Jan 1 to Jan 31", "between 2024-01-01 and 2024-01-31" → extract date range
  * "last 7 days", "past month" → calculate and extract date range
- Match user terms intelligently to available data
- IMPORTANT: When no year is specified in dates, assume current year (2025)
"""
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"temperature": 0.1},
            }
        )
        response = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
        raw = response["body"].read()
        result = json.loads(raw)
        ai_text = result["output"]["message"]["content"][0]["text"].strip()
        
        # Debug: Show what AI returned
        print(f"DEBUG: AI Response: {ai_text}")

        if "{" in ai_text and "}" in ai_text:
            start = ai_text.find("{")
            end = ai_text.rfind("}") + 1
            json_str = ai_text[start:end]
            plan = json.loads(json_str)
        else:
            raise ValueError("Model did not return JSON")
        
        # Debug: Show parsed plan
        print(f"DEBUG: Parsed AI Plan: {plan}")

        plan.setdefault("chart_type", "bar")
        plan.setdefault("x_axis", "Branch Name")
        plan.setdefault("y_axis", "Row Total")
        plan.setdefault("aggregation", "sum")
        plan.setdefault("title", "Anandhaas Revenue Analysis")
        plan.setdefault("dual_metrics", False)
        plan.setdefault("y_axis_secondary", None)
        plan.setdefault("aggregation_secondary", None)
        plan.setdefault("comparison_type", None)

        filters = []

        if plan.get("item_filters"):
            if len(plan["item_filters"]) == 1:
                filters.append(("Item/Service Description", plan["item_filters"][0]))
            else:
                filters.append(("Item/Service Description_in", plan["item_filters"]))

        if plan.get("category_filters"):
            if len(plan["category_filters"]) == 1:
                filters.append(("Category", plan["category_filters"][0]))
            else:
                filters.append(("Category_in", plan["category_filters"]))

        if plan.get("branch_filters"):
            if len(plan["branch_filters"]) == 1:
                filters.append(("Branch Name", plan["branch_filters"][0]))
            else:
                filters.append(("Branch_in", plan["branch_filters"]))

        if plan.get("group_filters"):
            if len(plan["group_filters"]) == 1:
                filters.append(("Group Name", plan["group_filters"][0]))
            else:
                filters.append(("Group_in", plan["group_filters"]))

        if plan.get("customer_filters"):
            if len(plan["customer_filters"]) == 1:
                filters.append(("Customer/Vendor Name", plan["customer_filters"][0]))
            else:
                filters.append(("Customer_in", plan["customer_filters"]))

        if plan.get("subgroup_filters"):
            if len(plan["subgroup_filters"]) == 1:
                filters.append(("SubGroup", plan["subgroup_filters"][0]))
            else:
                filters.append(("SubGroup_in", plan["subgroup_filters"]))

        if plan.get("month_filter"):
            month_val = plan["month_filter"]
            if isinstance(month_val, list):
                filters.append(("date_month_in", month_val))
            else:
                filters.append(("date_month", month_val))

        if plan.get("date_filter"):
            date_val = plan["date_filter"]
            if isinstance(date_val, list) and len(date_val) == 2:
                filters.append(("date_range", date_val))
            else:
                filters.append(("date_specific", date_val))

        if plan.get("year_filter"):
            year_val = plan["year_filter"]
            if isinstance(year_val, list):
                filters.append(("date_year_in", year_val))
            else:
                filters.append(("date_year", year_val))

        plan["filters"] = filters
        return plan

    except Exception as e:
    
        print(f"AI model failed to process query: {str(e)}")
        raise

def create_anandhaas_visualization(data: pd.DataFrame, ai_plan: dict):
    dual_metrics = ai_plan.get("dual_metrics", False) or ai_plan.get("y_axis") == "dual"
    comparison_type = ai_plan.get("comparison_type", "metric")
    
    if dual_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    else:
        fig, ax = plt.subplots(figsize=(20, 12))
    
    filtered_data = data.copy()
    filters = ai_plan.get("filters", [])

    for filter_type, filter_value in filters:
        if filter_type == "date_month":
            filtered_data = filtered_data[
                filtered_data["Date"].dt.month == int(filter_value)
            ]
        elif filter_type == "date_month_in":
            month_list = [int(m) for m in filter_value]
            filtered_data = filtered_data[
                filtered_data["Date"].dt.month.isin(month_list)
            ]
        elif filter_type == "date_specific":
            try:
                # Handle various date formats and add current year if missing
                if len(filter_value.split('-')) == 2:  # MM-DD format
                    current_year = pd.Timestamp.now().year
                    filter_value = f"{current_year}-{filter_value}"
                target_date = pd.to_datetime(filter_value).date()
                filtered_data = filtered_data[
                    filtered_data["Date"].dt.date == target_date
                ]
                print(f"DEBUG: Date filter '{target_date}' resulted in {len(filtered_data)} records")
            except Exception as e:
                print(f"DEBUG: Date parsing error for '{filter_value}': {e}")
                continue
        elif filter_type == "date_range":
            start_date = pd.to_datetime(filter_value[0])
            end_date = pd.to_datetime(filter_value[1])
            filtered_data = filtered_data[
                (filtered_data["Date"] >= start_date) & 
                (filtered_data["Date"] <= end_date)
            ]
        elif filter_type == "date_year":
            filtered_data = filtered_data[
                filtered_data["Date"].dt.year == int(filter_value)
            ]
        elif filter_type == "date_year_in":
            year_list = [int(y) for y in filter_value]
            filtered_data = filtered_data[
                filtered_data["Date"].dt.year.isin(year_list)
            ]
        elif filter_type in ["Category", "Item/Service Description", "Branch Name", "Group Name", "Customer/Vendor Name", "SubGroup"]:
            filter_value_str = str(filter_value).lower().strip()
            
            # Debug: Show what we're looking for and what's available
            available_items = filtered_data[filter_type].astype(str).str.lower().unique()
            print(f"DEBUG: Looking for '{filter_value_str}' in {filter_type}")
            print(f"DEBUG: Available items: {sorted(available_items)}")
            
            # First try exact match
            exact_match = filtered_data[filtered_data[filter_type].astype(str).str.lower().str.strip() == filter_value_str]
            
            if not exact_match.empty:
                filtered_data = exact_match
                print(f"DEBUG: Found exact match for '{filter_value_str}': {len(exact_match)} records")
            else:
                # Dynamic intelligent matching to avoid substring confusion
                # Get all unique values in the column to check for conflicts
                all_values = filtered_data[filter_type].astype(str).str.lower().unique()
                
                # Check if there are longer items that contain our search term
                conflicting_items = [item for item in all_values if filter_value_str in item and item != filter_value_str]
                print(f"DEBUG: Conflicting items found: {conflicting_items}")
                
                if conflicting_items:
                    # Use word boundary matching to avoid substring matches
                    pattern = r'\b' + filter_value_str.replace(' ', r'\s+') + r'\b'
                    mask = filtered_data[filter_type].astype(str).str.contains(pattern, case=False, na=False, regex=True)
                    
                    # Additional exclusion for items that are clearly different
                    for conflicting_item in conflicting_items:
                        if len(conflicting_item.split()) > len(filter_value_str.split()):
                            # Exclude longer compound items
                            mask = mask & ~filtered_data[filter_type].astype(str).str.lower().str.contains(conflicting_item, case=False, na=False)
                            print(f"DEBUG: Excluding conflicting item: '{conflicting_item}'")
                else:
                    # No conflicts, use simple word boundary matching
                    pattern = r'\b' + filter_value_str.replace(' ', r'\s+') + r'\b'
                    mask = filtered_data[filter_type].astype(str).str.contains(pattern, case=False, na=False, regex=True)
                
                tmp = filtered_data[mask]
                if tmp.empty:
                    # Fallback to simple contains if word boundary fails
                    mask = filtered_data[filter_type].astype(str).str.contains(filter_value_str, case=False, na=False)
                    tmp = filtered_data[mask]
                
                # Debug: Show what items were actually matched
                matched_items = tmp[filter_type].astype(str).str.lower().unique()
                print(f"DEBUG: Items matched for '{filter_value_str}': {sorted(matched_items)}")
                
                filtered_data = tmp
                print(f"DEBUG: Filter '{filter_type}={filter_value}' resulted in {len(filtered_data)} records")
                
                # Debug: Show total revenue for this filter
                if len(filtered_data) > 0:
                    total_revenue = filtered_data['Row Total'].sum()
                    print(f"DEBUG: Total revenue after filtering: {total_revenue}")
        elif filter_type in ["Category_in", "Item/Service Description_in", "Branch_in", "Group_in", "Customer_in", "SubGroup_in"]:
            col_map = {
                "Category_in": "Category",
                "Item/Service Description_in": "Item/Service Description",
                "Branch_in": "Branch Name",
                "Group_in": "Group Name",
                "Customer_in": "Customer/Vendor Name",
                "SubGroup_in": "SubGroup",
            }
            col = col_map[filter_type]
            values = [str(v) for v in filter_value]
            clean_data = filtered_data
            tmp = clean_data[clean_data[col].astype(str).isin(values)]
            filtered_data = tmp

    if filtered_data.empty:
        # Debug information for troubleshooting
        print(f"DEBUG: Applied filters: {filters}")
        print(f"DEBUG: Original data shape: {data.shape}")
        
        # Check what data is available for each filter
        debug_info = []
        temp_data = data.copy()
        
        for filter_type, filter_value in filters:
            if filter_type in ["Category", "Branch Name", "Group Name", "Customer/Vendor Name", "SubGroup"]:
                available_values = temp_data[filter_type].dropna().unique().tolist()
                debug_info.append(f"{filter_type}: looking for '{filter_value}', available: {available_values[:10]}")
            elif filter_type == "date_specific":
                date_range = f"{temp_data['Date'].min()} to {temp_data['Date'].max()}"
                debug_info.append(f"Date: looking for '{filter_value}', available range: {date_range}")
        
        print("DEBUG: Filter analysis:")
        for info in debug_info:
            print(f"  {info}")
        
        raise ValueError(f"No data found after applying filters. Check filter values against available data.")

    x_col = ai_plan.get("x_axis", "Branch Name")
    
    # Handle month-wise grouping
    if x_col == "Month":
        filtered_data = filtered_data.copy()
        filtered_data["Month"] = filtered_data["Date"].dt.strftime("%B %Y")
        filtered_data["MonthSort"] = filtered_data["Date"].dt.to_period("M")
    
    if dual_metrics:
        x_col = ai_plan.get("x_axis", "Branch Name")
        y_col_1 = ai_plan.get("y_axis", "Row Total")
        y_col_2 = ai_plan.get("y_axis_secondary", "Quantity")
        agg_1 = ai_plan.get("aggregation", "sum")
        agg_2 = ai_plan.get("aggregation_secondary", "sum")
        limit = ai_plan.get("limit")
        
        if comparison_type == "monthly" and ai_plan.get("month_filter"):
            # Month comparison logic
            month_list = [int(m) for m in ai_plan.get("month_filter", [])]
            month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                          7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
            
            # Get top items first
            if y_col_1 == "count":
                top_items = filtered_data.groupby(x_col).size().sort_values(ascending=False)
            else:
                top_items = filtered_data.groupby(x_col)[y_col_1].agg(agg_1).sort_values(ascending=False)
            
            if limit and isinstance(limit, int) and limit > 0:
                top_items = top_items.head(limit)
            
            filtered_data = filtered_data[filtered_data[x_col].isin(top_items.index)]
            
            # Create data for each month
            metric1_data = {}
            for month in month_list:
                month_data = filtered_data[filtered_data["Date"].dt.month == month]
                if y_col_1 == "count":
                    month_metric = month_data.groupby(x_col).size()
                else:
                    month_metric = month_data.groupby(x_col)[y_col_1].agg(agg_1)
                metric1_data[month_names.get(month, f"Month {month}")] = month_metric.reindex(top_items.index, fill_value=0)
            
            # Create side-by-side bars
            items = list(top_items.index)
            x_pos = range(len(items))
            width = 0.35
            months = list(metric1_data.keys())
            colors = ['#1e40af', '#059669', '#d97706', '#dc2626']
            
            for i, month in enumerate(months):
                values = [metric1_data[month].get(item, 0) for item in items]
                bars = ax1.bar([x + width*i for x in x_pos], values, width, 
                              label=month, color=colors[i % len(colors)], alpha=0.95, edgecolor='white', linewidth=1.5)
                
                for bar in bars:
                    height = bar.get_height()
                    label = f'₹{height:,.0f}' if y_col_1 == 'Row Total' else f'{height:.0f}'
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01, 
                            label, ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            ax1.set_xticks([x + width/2 for x in x_pos])
            ax1.set_xticklabels(items, rotation=45, ha='right', fontsize=10)
            ax1.set_xlabel(x_col, fontsize=12, fontweight="bold")
            ax1.set_ylabel(f"{y_col_1} ({agg_1})", fontsize=12, fontweight="bold")
            ax1.set_title(f"{' vs '.join(months)} {y_col_1} Comparison", fontsize=14, fontweight="bold")
            ax1.legend()
            
            # Second chart shows percentage comparison
            total_by_item = {item: sum(metric1_data[month].get(item, 0) for month in months) for item in items}
            percentages = {}
            for month in months:
                percentages[month] = [(metric1_data[month].get(item, 0) / total_by_item[item] * 100) if total_by_item[item] > 0 else 0 for item in items]
            
            for i, month in enumerate(months):
                bars = ax2.bar([x + width*i for x in x_pos], percentages[month], width, 
                              label=month, color=colors[i % len(colors)], alpha=0.95)
                
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%',
                            ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            ax2.set_xticks([x + width/2 for x in x_pos])
            ax2.set_xticklabels(items, rotation=45, ha='right', fontsize=10)
            ax2.set_xlabel(x_col, fontsize=12, fontweight="bold")
            ax2.set_ylabel("Percentage Share", fontsize=12, fontweight="bold")
            ax2.set_title("Percentage Share Comparison", fontsize=14, fontweight="bold")
            ax2.legend()
            
            chart_data = []
            for item in items:
                item_data = {"name": str(item)}
                for month in months:
                    item_data[month.lower()] = float(metric1_data[month].get(item, 0))
                chart_data.append(item_data)
        
        else:
            # Regular dual metrics (two different metrics)
            if y_col_1 == "count":
                metric1_data = filtered_data.groupby(x_col).size().sort_values(ascending=False)
            else:
                metric1_data = filtered_data.groupby(x_col)[y_col_1].agg(agg_1).sort_values(ascending=False)
            
            if limit and isinstance(limit, int) and limit > 0:
                metric1_data = metric1_data.head(limit)
            
            if y_col_2 == "count":
                metric2_data = filtered_data.groupby(x_col).size().reindex(metric1_data.index, fill_value=0)
            else:
                metric2_data = filtered_data.groupby(x_col)[y_col_2].agg(agg_2).reindex(metric1_data.index, fill_value=0)
            
            # First metric chart
            bars1 = ax1.bar(range(len(metric1_data)), metric1_data.values, color='#1e40af', alpha=0.95, edgecolor='white', linewidth=1.5)
            ax1.set_xticks(range(len(metric1_data)))
            ax1.set_xticklabels(metric1_data.index, rotation=0 if len(metric1_data) <= 5 else 45, ha='center' if len(metric1_data) <= 5 else 'right', fontsize=11)
            ax1.set_xlabel(x_col, fontsize=12, fontweight="bold")
            ax1.set_ylabel(f"{y_col_1} ({agg_1})", fontsize=12, fontweight="bold")
            ax1.set_title(f"{y_col_1} Analysis", fontsize=14, fontweight="bold")
            
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                label = f'₹{height:,.0f}' if y_col_1 == 'Row Total' else f'{height:.0f}'
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01, label,
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Second metric chart
            bars2 = ax2.bar(range(len(metric2_data)), metric2_data.values, color='#059669', alpha=0.95, edgecolor='white', linewidth=1.5)
            ax2.set_xticks(range(len(metric2_data)))
            ax2.set_xticklabels(metric2_data.index, rotation=0 if len(metric2_data) <= 5 else 45, ha='center' if len(metric2_data) <= 5 else 'right', fontsize=11)
            ax2.set_xlabel(x_col, fontsize=12, fontweight="bold")
            ax2.set_ylabel(f"{y_col_2} ({agg_2})", fontsize=12, fontweight="bold")
            ax2.set_title(f"{y_col_2} Analysis", fontsize=14, fontweight="bold")
            
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                label = f'₹{height:,.0f}' if y_col_2 == 'Row Total' else f'{height:.0f}'
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01, label,
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            chart_data = []
            for item in metric1_data.index:
                chart_data.append({
                    "name": str(item),
                    "metric1": float(metric1_data.get(item, 0)),
                    "metric2": float(metric2_data.get(item, 0)),
                    "metric1_name": y_col_1,
                    "metric2_name": y_col_2
                })
        
    else:
        y_col = ai_plan.get("y_axis", "Row Total")
        agg_method = ai_plan.get("aggregation", "sum")

        if y_col == "count":
            if x_col == "Month":
                grouped_data = filtered_data.groupby(["MonthSort", "Month"]).size().reset_index(name="count")
                grouped_data = grouped_data.set_index("Month")["count"].sort_index()
            else:
                grouped_data = filtered_data[x_col].value_counts().sort_values(ascending=False)
        elif y_col == "Quantity" and "Quantity" in filtered_data.columns:
            if x_col == "Month":
                grouped_data = filtered_data.groupby(["MonthSort", "Month"])["Quantity"].agg(agg_method).reset_index()
                grouped_data = grouped_data.set_index("Month")["Quantity"].sort_index()
            else:
                grouped_data = filtered_data.groupby(x_col)["Quantity"].agg(agg_method).sort_values(ascending=False)
        else:
            if x_col == "Month":
                grouped_data = filtered_data.groupby(["MonthSort", "Month"])[y_col].agg(agg_method).reset_index()
                grouped_data = grouped_data.set_index("Month")[y_col].sort_index()
            else:
                grouped_data = filtered_data.groupby(x_col)[y_col].agg(agg_method).sort_values(ascending=False)

        # Apply limit if specified
        limit = ai_plan.get("limit")
        if limit and isinstance(limit, int) and limit > 0:
            grouped_data = grouped_data.head(limit)
            print(f"Applied limit: showing top {limit} results")

        chart_type = ai_plan.get("chart_type", "bar")

        if chart_type == "pie":
            grouped_data = grouped_data.sort_values(ascending=False)
            professional_colors = ['#1e40af', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#65a30d', '#ea580c']
            colors = [professional_colors[i % len(professional_colors)] for i in range(len(grouped_data))]
            
            wedges, texts, autotexts = ax.pie(
                grouped_data.values,
                labels=None,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                colors=colors,
                startangle=90,
                pctdistance=0.85,
                explode=[0.05 if i == 0 else 0 for i in range(len(grouped_data))]
            )
            
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(10)
            
            ax.legend(wedges, [f'{name}: ₹{value:,.0f}' if y_col == 'Row Total' else f'{name}: {value:.0f}' 
                              for name, value in grouped_data.items()], 
                     title=x_col, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
        elif chart_type == "line":
            ax.plot(
                range(len(grouped_data)),
                grouped_data.values,
                marker="o",
                linewidth=3,
                markersize=8,
            )
            ax.set_xticks(range(len(grouped_data)))
            ax.set_xticklabels(grouped_data.index, rotation=0 if len(grouped_data) <= 5 else 45, ha='center' if len(grouped_data) <= 5 else 'right', fontsize=11)
            ax.set_xlabel(x_col, fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{y_col} {'(Lakhs)' if y_col == 'Row Total' else ''}", fontsize=12, fontweight="bold")
            if y_col == "Row Total":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/100000:.0f}'))
            ax.grid(True, alpha=0.3)
        else:
            professional_colors = ['#1e40af', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#65a30d', '#ea580c']
            bar_colors = [professional_colors[i % len(professional_colors)] for i in range(len(grouped_data))]
            bars = ax.bar(
                range(len(grouped_data)),
                grouped_data.values,
                color=bar_colors,
                alpha=0.95,
                edgecolor='white',
                linewidth=1.5
            )
            ax.set_xticks(range(len(grouped_data)))
            ax.set_xticklabels(grouped_data.index, rotation=0 if len(grouped_data) <= 5 else 45, ha='center' if len(grouped_data) <= 5 else 'right', fontsize=11)
            ax.set_xlabel(x_col, fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{y_col} {'(Lakhs)' if y_col == 'Row Total' else ''}", fontsize=12, fontweight="bold")
            if y_col == "Row Total":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/100000:.0f}'))
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if y_col == "Row Total":
                    label = f"₹{height:,.0f}"
                else:
                    label = f"{height:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height*0.01,
                    label,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9
                )
        
        chart_data = [{"name": str(k), "value": float(v)} for k, v in grouped_data.items()]
    
    if not dual_metrics:
        ax.set_title(ai_plan.get("title", "Anandhaas Analysis"), fontsize=16, fontweight="bold", pad=20)
    else:
        fig.suptitle(ai_plan.get("title", "Anandhaas Analysis"), fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    if dual_metrics:
        plt.subplots_adjust(top=0.9)
    return chart_data, fig

def generate_simple_response(ai_plan: dict) -> str:
    chart_desc_map = {"bar": "comparison chart", "pie": "distribution chart", "line": "trend chart"}
    chart_desc = chart_desc_map.get(ai_plan.get("chart_type", "bar"), "chart")
    filters = ai_plan.get("filters", [])
    parts = []
    for filter_type, filter_value in filters:
        if filter_type == "date_month":
            month_names = [
                "",
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ]
            if isinstance(filter_value, int) and 1 <= filter_value <= 12:
                parts.append(f"in {month_names[filter_value]}")
        elif filter_type.endswith("_in"):
            parts.append(f"for {', '.join(map(str, filter_value))}")
        else:
            parts.append(f"for {filter_value}")
    filter_text = ""
    if parts:
        filter_text = " " + " ".join(parts)
    return (
        f"Created a {chart_desc} showing {ai_plan.get('y_axis', 'Row Total')} by "
        f"{ai_plan.get('x_axis', 'Branch Name')}{filter_text}."
    )

def detect_language(text: str) -> str:
    tamil_chars = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
    letters = [c for c in text if c.isalpha()]
    total_chars = len(letters)
    if total_chars == 0:
        return "unknown"
    if tamil_chars / total_chars > 0.3:
        return "tamil"
    return "english"

def translate_tamil_to_english(tamil_text: str) -> str:
    try:
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        }
        data = {
            "input": tamil_text,
            "source_language_code": "ta-IN",
            "target_language_code": "en-IN",
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1",
        }
        response = requests.post(SARVAM_TRANSLATE_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json().get("translated_text", tamil_text)
    except Exception:
        return tamil_text

def transcribe_audio(file_path: str) -> str:
    try:
        headers = {"api-subscription-key": SARVAM_API_KEY}
        with open(file_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            response = requests.post(SARVAM_STT_URL, headers=headers, files=files, timeout=45)
        response.raise_for_status()
        return response.json().get("transcript", "")
    except Exception as e:
        return f"TRANSCRIPTION_ERROR: {e}"

def text_to_speech(text: str, language: str = "hi-IN") -> bytes | None:
    try:
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        }
        data = {"text": text, "target_language_code": language}
        response = requests.post(SARVAM_TTS_URL, headers=headers, json=data, timeout=45)
        response.raise_for_status()
        audios = response.json().get("audios")
        if not audios:
            return None
        return base64.b64decode(audios[0])
    except Exception:
        return None

def generate_pdf_report(fig, title, insights):
    with io.BytesIO() as pdf_buffer:
        with PdfPages(pdf_buffer) as pdf:
            # Save only the chart - no separate insights page
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
        pdf_buffer.seek(0)
        return pdf_buffer.read()


@app.route("/api/dashboard-data", methods=["GET"])
def get_dashboard_data():
    global anandhaas_data
    if anandhaas_data is None:
        anandhaas_data = load_anandhaas_data()

    if anandhaas_data is None:
        return jsonify({"error": "Data not available"}), 404

    analysis = analyze_anandhaas_structure(anandhaas_data)
    if analysis.get("date_range"):
        analysis["date_range"]["start"] = analysis["date_range"]["start"].isoformat()
        analysis["date_range"]["end"] = analysis["date_range"]["end"].isoformat()
    return jsonify(analysis)

@app.route("/api/query", methods=["POST"])
def process_query():
    global anandhaas_data
    try:
        payload = request.get_json(silent=True) or {}
        query = payload.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        if anandhaas_data is None:
            anandhaas_data = load_anandhaas_data()
        if anandhaas_data is None:
            return jsonify({"error": "Data not available. Ensure anandhaas_data.csv exists."}), 404

        detected_lang = detect_language(query)
        english_query = translate_tamil_to_english(query) if detected_lang == "tamil" else query

        data_analysis = analyze_anandhaas_structure(anandhaas_data)
        ai_plan = get_ai_plan(english_query, data_analysis)
        chart_data, fig = create_anandhaas_visualization(anandhaas_data, ai_plan)
        response_text = generate_simple_response(ai_plan)

        try:
            chart_title = ai_plan.get("title", "Anandhaas Revenue Analysis")
            pdf_bytes = generate_pdf_report(fig, chart_title, response_text)
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
         
            global last_pdf_data
            last_pdf_data = {
                'data': pdf_bytes,
                'title': chart_title,
                'insights': response_text,
                'filename': f"{chart_title.replace(' ', '_')}_report.pdf"
            }
            print(f"PDF stored: {chart_title}, size: {len(pdf_bytes)} bytes")
        except Exception as e:
            print(f"PDF generation error: {e}")
            pdf_b64 = None

        plt.close(fig)

        return jsonify({
            "original_query": query,
            "english_query": english_query,
            "chart_type": ai_plan.get("chart_type", "bar"),
            "title": ai_plan.get("title", "Analysis"),
            "data": chart_data,
            "x_axis": ai_plan.get("x_axis", "Branch Name"),
            "y_axis": ai_plan.get("y_axis", "Row Total"),
            "insights": response_text,
            "pdf_base64": pdf_b64,
            "pdf_filename": f"{ai_plan.get('title','report').replace(' ', '_')}.pdf",
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    temp_file_path = None
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400
        
        audio_file = request.files["audio"]
        print(f"Received audio file: {audio_file.filename}, content_type: {audio_file.content_type}")
        
     
        if not SARVAM_API_KEY or len(SARVAM_API_KEY.strip()) < 10:
            return jsonify({"error": "Sarvam API key not configured"}), 500
        
      
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()  
        
   
        audio_file.save(temp_file_path)
        print(f"Saved audio to: {temp_file_path}")
        
    
        transcript = transcribe_audio(temp_file_path)
        print(f"Transcription result: {transcript}")
        
        return jsonify({"transcript": transcript})
        
    except Exception as e:
        print(f"Transcription endpoint error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"Failed to cleanup temp file: {cleanup_error}")

@app.route("/api/tts", methods=["POST"])
def tts_api():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    language = data.get("language", "hi-IN")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    audio_bytes = text_to_speech(text, language)
    if not audio_bytes:
        return jsonify({"error": "TTS failed"}), 500
    return jsonify({"audio": base64.b64encode(audio_bytes).decode("utf-8")})


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID") or "C09UUJZ56QJ"

print(f"DEBUG: SLACK_BOT_TOKEN loaded: {SLACK_BOT_TOKEN[:20]}...")
print(f"DEBUG: SLACK_CHANNEL_ID loaded: {SLACK_CHANNEL_ID}")


try:
    test_client = WebClient(token=SLACK_BOT_TOKEN)
    test_response = test_client.auth_test()
    print(f"DEBUG: Slack auth test successful: {test_response.get('ok')}")
except Exception as e:
    print(f"DEBUG: Slack auth test failed: {e}")

def send_pdf_to_slack(pdf_bytes, filename, title, initial_comment):
    """Send PDF to Slack - exact copy from working Streamlit version"""
    token = SLACK_BOT_TOKEN
    channel = SLACK_CHANNEL_ID
    if not token or not channel:
        return {"success": False, "message": "Slack not configured"}
    try:
        client = WebClient(token=token)
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_file.seek(0)
        response = client.files_upload_v2(
            channel=channel,
            file=pdf_file,
            filename=filename,
            title=title,
            initial_comment=initial_comment
        )
        if response and response.get("ok"):
            return {"success": True, "message": "Successfully sent to Slack"}
        else:
            error_msg = response.get("error", "Unknown error") if response else "Unknown error"
            return {"success": False, "message": f"Failed to send to Slack: {error_msg}"}
    except SlackApiError as e:
        error_msg = str(e.response.get("error", str(e))) if hasattr(e, 'response') else str(e)
        return {"success": False, "message": f"Slack API error: {error_msg}"}
    except Exception as e:
        return {"success": False, "message": f"Error sending to Slack: {str(e)}"}

@app.route("/api/send-to-slack", methods=["POST", "GET"])
def send_to_slack_api():
    print("=== SLACK ENDPOINT CALLED ===")
    try:
        global last_pdf_data
        print(f"PDF available: {bool(last_pdf_data.get('data'))}")
        print(f"PDF size: {len(last_pdf_data.get('data', b''))} bytes")
        
        if not last_pdf_data.get('data'):
            return jsonify({"success": False, "message": "No PDF available. Generate a chart first."}), 400
        
        print("Calling send_pdf_to_slack...")
        result = send_pdf_to_slack(
            pdf_bytes=last_pdf_data['data'],
            filename=last_pdf_data['filename'],
            title=last_pdf_data['title'],
            initial_comment=last_pdf_data['insights']
        )
        print(f"Slack result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Slack endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/last-pdf-info", methods=["GET"])
def get_last_pdf_info():
    """Get info about the last generated PDF like Streamlit session_state"""
    global last_pdf_data
    if last_pdf_data.get('data'):
        return jsonify({
            "available": True,
            "filename": last_pdf_data['filename'],
            "title": last_pdf_data['title']
        })
    else:
        return jsonify({"available": False})

if __name__ == "__main__":
    # app.run(debug=True, port=5000)
    app.run(host="0.0.0.0", port=5000, debug=True)