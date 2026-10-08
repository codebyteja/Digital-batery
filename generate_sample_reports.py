"""
=============================================================================
generate_sample_reports.py — Create Sample EV Battery Service Reports (PDF)
=============================================================================
This script uses the fpdf library to create 3 realistic-looking
"EV Service Center Battery Report" PDFs with labeled fields:
  - Voltage, Current, Temperature, Cycle Count, Capacity

These PDFs are used to demo the Upload Report feature of the dashboard.
The /upload_report endpoint uses pdfplumber + regex to extract these values.

Usage:
  python generate_sample_reports.py
=============================================================================
"""

import os
import random
from datetime import datetime, timedelta
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_reports")

# Sample report configurations with realistic battery data
REPORT_CONFIGS = [
    {
        "filename": "battery_report_healthy.pdf",
        "title": "EV Battery Health Assessment Report",
        "battery_id": "B0005",
        "vin": "1HGBH41JXMN109186",
        "vehicle": "Tesla Model 3 - 2024",
        "voltage": 3.82,
        "current": 1.15,
        "temperature": 28.5,
        "cycle_count": 45,
        "capacity": 1.78,
        "status": "HEALTHY",
        "notes": "Battery is in excellent condition. All parameters within normal range. "
                 "No signs of accelerated degradation detected. Recommended next check: 6 months.",
    },
    {
        "filename": "battery_report_moderate.pdf",
        "title": "EV Battery Health Assessment Report",
        "battery_id": "B0006",
        "vin": "5YJSA1E26MF123456",
        "vehicle": "Nissan Leaf - 2023",
        "voltage": 3.54,
        "current": 1.42,
        "temperature": 35.2,
        "cycle_count": 120,
        "capacity": 1.55,
        "status": "MODERATE WEAR",
        "notes": "Battery shows moderate wear consistent with usage. Slight capacity fade "
                 "observed. Recommend avoiding frequent fast charging. Next check: 3 months.",
    },
    {
        "filename": "battery_report_degraded.pdf",
        "title": "EV Battery Health Assessment Report",
        "battery_id": "B0007",
        "vin": "WVWZZZ3CZWE789012",
        "vehicle": "Hyundai Kona Electric - 2022",
        "voltage": 3.21,
        "current": 1.78,
        "temperature": 42.1,
        "cycle_count": 155,
        "capacity": 1.32,
        "status": "SERVICE RECOMMENDED",
        "notes": "Significant capacity degradation detected. Battery operating at elevated "
                 "temperatures. Recommend immediate service evaluation and potential cell "
                 "balancing procedure. Limit fast charging to emergency use only.",
    },
]


def create_report_pdf(config):
    """
    Create a professional-looking battery health report PDF.
    
    The PDF is structured with labeled fields that the /upload_report
    endpoint can parse using regex patterns like "Voltage: 3.82".
    
    Args:
        config: Dict with report data (voltage, current, etc.)
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # -----------------------------------------------------------------------
    # Header / Title
    # -----------------------------------------------------------------------
    pdf.set_fill_color(0, 102, 204)  # Blue header bar
    pdf.rect(0, 0, 210, 35, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_y(8)
    pdf.cell(0, 10, config["title"], new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "EV Service Center - Certified Battery Diagnostics", new_x="LMARGIN", new_y="NEXT", align="C")

    # -----------------------------------------------------------------------
    # Report metadata
    # -----------------------------------------------------------------------
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(42)
    pdf.set_font("Helvetica", "", 10)

    report_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%B %d, %Y")
    report_id = f"RPT-{random.randint(10000, 99999)}"

    pdf.cell(95, 7, f"Report Date: {report_date}")
    pdf.cell(95, 7, f"Report ID: {report_id}", new_x="LMARGIN", new_y="NEXT", align="R")
    pdf.cell(95, 7, f"Vehicle: {config['vehicle']}")
    pdf.cell(95, 7, f"VIN: {config['vin']}", new_x="LMARGIN", new_y="NEXT", align="R")
    pdf.cell(95, 7, f"Battery ID: {config['battery_id']}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # -----------------------------------------------------------------------
    # Divider line
    # -----------------------------------------------------------------------
    pdf.set_draw_color(0, 102, 204)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    # -----------------------------------------------------------------------
    # Battery Parameters Section
    # (These labeled fields are what the upload_report endpoint parses)
    # -----------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "Battery Parameters", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)

    # Each parameter on its own line with the label: value format
    # The regex in upload_report looks for patterns like "Voltage: 3.82"
    params = [
        ("Voltage", f"{config['voltage']} V"),
        ("Current", f"{config['current']} A"),
        ("Temperature", f"{config['temperature']} C"),
        ("Cycle Count", f"{config['cycle_count']}"),
        ("Capacity", f"{config['capacity']} Ah"),
    ]

    for label, value in params:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(60, 9, f"{label}:")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 9, value, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # -----------------------------------------------------------------------
    # Status Section
    # -----------------------------------------------------------------------
    pdf.set_draw_color(0, 102, 204)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "Diagnostic Result", new_x="LMARGIN", new_y="NEXT")

    # Status badge with color coding
    status = config["status"]
    if "HEALTHY" in status:
        pdf.set_fill_color(34, 197, 94)  # Green
    elif "MODERATE" in status:
        pdf.set_fill_color(234, 179, 8)  # Yellow
    else:
        pdf.set_fill_color(239, 68, 68)  # Red

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(60, 10, f"  Status: {status}", new_x="LMARGIN", new_y="NEXT", fill=True)

    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, f"Notes: {config['notes']}")

    pdf.ln(10)

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "This report is generated by EV Battery Digital Twin diagnostic system.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 5, "For questions, contact your certified EV service center.", new_x="LMARGIN", new_y="NEXT", align="C")

    # Save PDF
    output_path = os.path.join(OUTPUT_DIR, config["filename"])
    pdf.output(output_path)
    print(f"[SAVED] {output_path}")


def main():
    """Generate all 3 sample battery report PDFs."""
    print("=" * 60)
    print("  EV Battery Digital Twin — Sample Report Generator")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for config in REPORT_CONFIGS:
        create_report_pdf(config)

    print(f"\n[DONE] {len(REPORT_CONFIGS)} sample reports saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
