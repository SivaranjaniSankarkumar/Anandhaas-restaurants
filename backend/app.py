from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import boto3
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


def load_anandhaas_data(csv_path: str = "anandhaas_data.csv") -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["Branch Name", "Posting Date", "Group Name", "Category", "Row Total"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return None
        df = df[required_cols].copy()
        df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
        df["Row Total"] = pd.to_numeric(df["Row Total"], errors="coerce")
        df = df.dropna(subset=["Posting Date", "Row Total"])
        print(f"Loaded Anandhaas data with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Cannot load anandhaas_data.csv: {e}")
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
            "start": data["Posting Date"].min(),
            "end": data["Posting Date"].max(),
        },
        "revenue_stats": {
            "total": float(data["Row Total"].sum()),
            "avg": float(data["Row Total"].mean()),
            "max": float(data["Row Total"].max()),
            "min": float(data["Row Total"].min()),
        },
    }
    return analysis

def get_ai_plan(query: str, data_analysis: dict) -> dict:
    branches = data_analysis.get("branches", [])
    categories = data_analysis.get("categories", [])
    groups = data_analysis.get("groups", [])

    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

        prompt = f"""
Analyze this business query about restaurant sales and create a visualization plan.

Query: "{query}"

Available Data:
- Branches: {branches}
- Categories: {categories}
- Groups: {groups}

Return ONLY valid JSON in this exact format:
{{
  "chart_type": "bar|pie|line|dual_bar",
  "x_axis": "Branch Name|Group Name|Category|Posting Date",
  "y_axis": "Row Total|count|dual",
  "aggregation": "sum|mean|count",
  "category_filters": null or [string, ...],
  "branch_filters": null or [string, ...],
  "group_filters": null or [string, ...],
  "month_filter": null or month_number or [month_numbers],
  "title": "chart title",
  "dual_metrics": false or true
}}

Rules:
- Extract ALL filters from the query: categories, branches, groups, and months
- For months: january=1, february=2, march=3, april=4, may=5, june=6, july=7, august=8, september=9, october=10, november=11, december=12
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
- If user asks "how many" or "count", use y_axis: "count" and aggregation: "count"
- For "each branch" or "by branch", use x_axis: "Branch Name"
- IMPORTANT: For "distribution", "breakdown", "share", "split", "proportion" → ALWAYS use chart_type: "pie"
- For "comparison", "compare", "vs" → use bar chart
- For time/trend, use line chart with Posting Date
- CRITICAL: If user asks for BOTH count AND revenue (like "count and revenue", "both count and total"), set dual_metrics: true, chart_type: "dual_bar", y_axis: "dual"
- Match user terms intelligently to available data
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

        if "{" in ai_text and "}" in ai_text:
            start = ai_text.find("{")
            end = ai_text.rfind("}") + 1
            json_str = ai_text[start:end]
            plan = json.loads(json_str)
        else:
            raise ValueError("Model did not return JSON")

        plan.setdefault("chart_type", "bar")
        plan.setdefault("x_axis", "Branch Name")
        plan.setdefault("y_axis", "Row Total")
        plan.setdefault("aggregation", "sum")
        plan.setdefault("title", "Anandhaas Revenue Analysis")
        plan.setdefault("dual_metrics", False)

        filters = []

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

        if plan.get("month_filter"):
            month_val = plan["month_filter"]
            if isinstance(month_val, list):
                filters.append(("date_month_in", month_val))
            else:
                filters.append(("date_month", month_val))

        plan["filters"] = filters
        return plan

    except Exception as e:
    
        print(f"AI model failed to process query: {str(e)}")
        raise

def create_anandhaas_visualization(data: pd.DataFrame, ai_plan: dict):
    dual_metrics = ai_plan.get("dual_metrics", False) or ai_plan.get("y_axis") == "dual"
    
    if dual_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    else:
        fig, ax = plt.subplots(figsize=(20, 12))
    
    filtered_data = data.copy()
    filters = ai_plan.get("filters", [])

    for filter_type, filter_value in filters:
        if filter_type == "date_month":
            filtered_data = filtered_data[
                filtered_data["Posting Date"].dt.month == int(filter_value)
            ]
        elif filter_type == "date_month_in":
            month_list = [int(m) for m in filter_value]
            filtered_data = filtered_data[
                filtered_data["Posting Date"].dt.month.isin(month_list)
            ]
        elif filter_type in ["Category", "Branch Name", "Group Name"]:
            clean_data = filtered_data.dropna(subset=[filter_type])
            mask = clean_data[filter_type].astype(str).str.contains(
                str(filter_value), case=False, na=False
            )
            tmp = clean_data[mask]
            if tmp.empty:
                tmp = clean_data[clean_data[filter_type] == filter_value]
            filtered_data = tmp
        elif filter_type in ["Category_in", "Branch_in", "Group_in"]:
            col_map = {
                "Category_in": "Category",
                "Branch_in": "Branch Name",
                "Group_in": "Group Name",
            }
            col = col_map[filter_type]
            values = [str(v) for v in filter_value]
            clean_data = filtered_data.dropna(subset=[col])
            tmp = clean_data[clean_data[col].astype(str).isin(values)]
            filtered_data = tmp

    if filtered_data.empty:
        raise ValueError("No data found after applying filters.")

    x_col = ai_plan.get("x_axis", "Branch Name")
    
    if dual_metrics:
        revenue_data = filtered_data.groupby(x_col)["Row Total"].sum().sort_values(ascending=False)
        count_data = filtered_data[x_col].value_counts().sort_values(ascending=False)
        
        # Revenue chart 
        bars1 = ax1.bar(range(len(revenue_data)), revenue_data.values, color='#1e40af', alpha=0.95, edgecolor='white', linewidth=1.5)
        ax1.set_xticks(range(len(revenue_data)))
        ax1.set_xticklabels(revenue_data.index, rotation=0 if len(revenue_data) <= 5 else 45, ha='center' if len(revenue_data) <= 5 else 'right', fontsize=11)
        ax1.set_xlabel(x_col, fontsize=12, fontweight="bold")
        ax1.set_ylabel("Revenue in Lakhs", fontsize=12, fontweight="bold")
        ax1.set_title("Revenue Analysis", fontsize=14, fontweight="bold")
        
        # Format Y-axis to show values in lakhs
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/100000:.0f}'))
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'₹{height:,.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Count chart
        bars2 = ax2.bar(range(len(count_data)), count_data.values, color='#059669', alpha=0.95, edgecolor='white', linewidth=1.5)
        ax2.set_xticks(range(len(count_data)))
        ax2.set_xticklabels(count_data.index, rotation=0 if len(count_data) <= 5 else 45, ha='center' if len(count_data) <= 5 else 'right', fontsize=11)
        ax2.set_xlabel(x_col, fontsize=12, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax2.set_title("Transaction Count", fontsize=14, fontweight="bold")
        
        # Format Y-axis for count 
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Create unified data with all branches for frontend
        all_branches = set(revenue_data.index) | set(count_data.index)
        chart_data = []
        for branch in all_branches:
            chart_data.append({
                "name": str(branch),
                "revenue": float(revenue_data.get(branch, 0)),
                "count": int(count_data.get(branch, 0))
            })
        
        # Sort by revenue for consistent display
        chart_data.sort(key=lambda x: x["revenue"], reverse=True)
        
    else:
        y_col = ai_plan.get("y_axis", "Row Total")
        agg_method = ai_plan.get("aggregation", "sum")

        if y_col == "count":
            grouped_data = filtered_data[x_col].value_counts().sort_values(ascending=False)
        else:
            grouped_data = filtered_data.groupby(x_col)[y_col].agg(agg_method).sort_values(ascending=False)

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
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            fig_text, ax_text = plt.subplots(figsize=(6, 4))
            ax_text.text(0.05, 0.95, title, fontsize=12, fontweight="bold", transform=ax_text.transAxes)
            ax_text.text(0.05, 0.85, "Key Insights:", fontsize=10, fontweight="bold", transform=ax_text.transAxes)
            insight_lines = insights.replace(". ", ".\n").split("\n")
            y_pos = 0.75
            for line in insight_lines[:10]:
                if line.strip():
                    ax_text.text(0.05, y_pos, line.strip(), fontsize=9, transform=ax_text.transAxes, wrap=True)
                    y_pos -= 0.08
            ax_text.axis("off")
            pdf.savefig(fig_text, bbox_inches="tight", dpi=150)
            plt.close(fig_text)
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
    app.run(debug=True, port=5000)
