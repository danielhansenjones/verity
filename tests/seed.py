"""Seed script - submit sample contracts to the pipeline via the API."""

import io
import os
import sys

API_URL = os.getenv("API_URL", "http://localhost:8000")

SAMPLE_CONTRACTS = [
    (
        "nda_sample.pdf",
        """\
NON-DISCLOSURE AGREEMENT

1. CONFIDENTIALITY OBLIGATIONS
The Receiving Party agrees to keep all Confidential Information strictly
confidential and shall indemnify and hold harmless the Disclosing Party
from any breach of this Agreement.

1.1 The obligations under this section are perpetual and irrevocable.

2. TERM AND TERMINATION
Either party may terminate this Agreement without cause upon reasonable
notice. Automatic renewal applies unless written notice is provided
thirty (30) days prior to the renewal date.

3. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

4. INTELLECTUAL PROPERTY ASSIGNMENT
All work product created in connection with this Agreement shall be
assigned exclusively and irrevocably to the Disclosing Party.

5. LIABILITY LIMITATION
In no event shall either party be liable for unlimited liability arising
from indirect or consequential damages.

6. DISPUTE RESOLUTION
Any disputes shall be resolved through binding arbitration at the sole
discretion of the Disclosing Party.
""",
    ),
    (
        "saas_agreement_sample.pdf",
        """\
SOFTWARE AS A SERVICE AGREEMENT

Article I - PAYMENT TERMS
Customer agrees to pay all fees as invoiced. Liquidated damages of 1.5%
per month shall apply to overdue balances. Payment obligations shall
survive termination of this Agreement.

Article II - WARRANTY
Provider warrants that the Service will perform in substantial conformance
with the documentation using best efforts standards.

Article III - INDEMNIFICATION
Customer shall indemnify and hold harmless Provider against any third-party
claims arising from Customer's use of the Service.

Article IV - FORCE MAJEURE
Neither party shall be liable for delays caused by circumstances beyond
their reasonable control.

Article V - CONFIDENTIALITY
Each party agrees to maintain in confidence all Confidential Information
received from the other party.
""",
    ),
]


def _make_pdf_bytes(text: str) -> bytes:

    lines = text.split("\n")
    pdf_lines = []
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        pdf_lines.append(f"({safe}) Tj T*")

    stream_content = "BT\n/F1 10 Tf\n50 750 Td\n14 TL\n" + "\n".join(pdf_lines) + "\nET"
    stream_bytes = stream_content.encode("latin-1", errors="replace")
    stream_len = len(stream_bytes)

    objects = []

    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R\n"
        b"   /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )

    objects.append(
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )

    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    header = b"%PDF-1.4\n"
    body = b"".join(objects)

    xref_offset = len(header) + len(body)
    offsets = []
    pos = len(header)
    for obj in objects:
        offsets.append(pos)
        pos += len(obj)

    xref = b"xref\n"
    xref += f"0 {len(objects) + 1}\n".encode()
    xref += b"0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return header + body + xref + trailer


def submit(filename: str, pdf_bytes: bytes) -> str:
    import requests

    resp = requests.post(
        f"{API_URL}/jobs",
        files={"file": (filename, io.BytesIO(pdf_bytes), "application/pdf")},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(
        f"  submitted {filename!r} → job_id={data['job_id']}  status={data['status']}"
    )
    return data["job_id"]


def main():
    print(f"Seeding against {API_URL}")
    job_ids = []
    for filename, text in SAMPLE_CONTRACTS:
        pdf_bytes = _make_pdf_bytes(text)
        job_id = submit(filename, pdf_bytes)
        job_ids.append(job_id)

    print("\nSubmitted jobs:")
    for jid in job_ids:
        print(f"  {API_URL}/jobs/{jid}")

    print("\nCheck status:")
    for jid in job_ids:
        import requests

        resp = requests.get(f"{API_URL}/jobs/{jid}", timeout=10)
        data = resp.json()
        print(f"  {jid[:8]}…  status={data['status']}  stage={data['stage']}")


if __name__ == "__main__":
    main()
