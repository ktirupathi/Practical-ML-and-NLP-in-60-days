"""Data ingestion component for parsing raw Enron email files and creating
a labeled dataset using keyword-based heuristic labeling.

Intents: QUESTION, REQUEST, COMPLAINT, MEETING_REQUEST, FOLLOW_UP, INFORM, OTHER
"""

import email
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

INTENT_PATTERNS: Dict[str, List[str]] = {
    "QUESTION": [
        r"\bquestion\b", r"\bwondering\b", r"\bwhat is\b", r"\bhow does\b",
        r"\bdo you know\b", r"\bcan you tell\b", r"\bwhy\b.*\?\s*$",
        r"\bany idea\b", r"\bwhat are\b", r"\bhow do\b", r"\bcurious\b",
    ],
    "MEETING_REQUEST": [
        r"\bschedul\w*\s+(?:a\s+)?meeting\b", r"\bset up a call\b",
        r"\bconference\s*call\b", r"\bbook\s+a\s+meeting\b",
        r"\bavailab\w*\s+for\s+a\s+call\b", r"\bmeeting\s+request\b",
        r"\bcalendar\s+invite\b", r"\blet(?:'s| us)\s+meet\b",
        r"\bschedule\s+time\b", r"\bvirtual\s+meeting\b",
    ],
    "REQUEST": [
        r"\bplease\b", r"\bcould you\b", r"\bcan you\b",
        r"\bwould you\b", r"\bneed you to\b", r"\bkindly\b",
        r"\bi need\b", r"\brequest\w*\b", r"\bsend me\b",
        r"\bprovide\b", r"\bi(?:'d| would) like\b",
    ],
    "COMPLAINT": [
        r"\bissue\b", r"\bproblem\b", r"\bconcerned\b",
        r"\bdisappointed\b", r"\bunacceptable\b", r"\bfrustrat\w*\b",
        r"\bescalat\w*\b", r"\bcomplaint\b", r"\bdissatisf\w*\b",
        r"\bnot happy\b", r"\bterrible\b",
    ],
    "FOLLOW_UP": [
        r"\bfollowing up\b", r"\bfollow\s*up\b", r"\bchecking in\b",
        r"\breminder\b", r"\bstatus update\b", r"\bjust checking\b",
        r"\bany update\b", r"\bcircling back\b", r"\btouch base\b",
    ],
    "INFORM": [
        r"\bfyi\b", r"\bletting you know\b", r"\bwanted to share\b",
        r"\bfor your information\b", r"\bplease note\b", r"\bheads up\b",
        r"\bjust wanted\b", r"\battached\b", r"\bsee below\b",
        r"\bhere is\b", r"\bupdate\b",
    ],
}


class DataIngestion:
    """Parses raw Enron email files and creates a labeled intent dataset.

    Falls back to a built-in synthetic dataset when raw files are absent.
    """

    def __init__(self, raw_data_dir: str = "data/raw/maildir",
                 output_dir: str = "data/ingested",
                 max_emails: Optional[int] = 50000):
        self.raw_data_dir = raw_data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_emails = max_emails

    def _parse_email_file(self, filepath: str) -> Optional[Dict[str, str]]:
        """Parse a single raw email file and extract key fields.

        Args:
            filepath: Path to a raw email text file.

        Returns:
            Dict with message_id, sender, subject, body, or None on failure.
        """
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            msg = email.message_from_string(raw)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="ignore")
                            break
            else:
                payload = msg.get_payload(decode=True)
                body = payload.decode("utf-8", errors="ignore") if payload else (msg.get_payload() or "")
            return {
                "message_id": msg.get("Message-ID", ""),
                "sender": msg.get("From", ""),
                "subject": msg.get("Subject", "") or "",
                "body": body.strip(),
            }
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", filepath, exc)
            return None

    def _assign_intent(self, subject: str, body: str) -> str:
        """Assign intent via keyword heuristics; fallback is OTHER."""
        combined = f"{subject} {body}".lower()
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return intent
        return "OTHER"

    def _collect_email_files(self) -> List[str]:
        files: List[str] = []
        for dirpath, _, fnames in os.walk(self.raw_data_dir):
            for fname in fnames:
                full = os.path.join(dirpath, fname)
                if os.path.isfile(full):
                    files.append(full)
        logger.info("Found %d candidate email files.", len(files))
        return files

    def ingest(self) -> Tuple[str, pd.DataFrame]:
        """Run ingestion pipeline; returns (csv_path, dataframe)."""
        if not os.path.isdir(self.raw_data_dir):
            logger.warning("Raw dir not found: %s — using synthetic dataset.", self.raw_data_dir)
            return self._synthetic_dataset()

        files = self._collect_email_files()
        if not files:
            logger.warning("No email files found — using synthetic dataset.")
            return self._synthetic_dataset()

        if self.max_emails and len(files) > self.max_emails:
            import random; random.seed(42)
            files = random.sample(files, self.max_emails)

        records, seen = [], set()
        for fp in tqdm(files, desc="Parsing emails"):
            parsed = self._parse_email_file(fp)
            if not parsed:
                continue
            mid = parsed["message_id"]
            if mid and mid in seen:
                continue
            if mid:
                seen.add(mid)
            if len(parsed["body"].split()) < 10:
                continue
            parsed["intent"] = self._assign_intent(parsed["subject"], parsed["body"])
            records.append(parsed)

        df = pd.DataFrame(records)
        logger.info("Ingested %d emails.\n%s", len(df), df["intent"].value_counts().to_string())
        out = str(self.output_dir / "emails.csv")
        df.to_csv(out, index=False)
        return out, df

    def _synthetic_dataset(self) -> Tuple[str, pd.DataFrame]:
        """Return a small labelled synthetic dataset for demonstration."""
        samples = [
            ("What is the deadline?", "I was wondering what the submission deadline is for the Q3 report. Can you tell me the exact date?", "QUESTION"),
            ("How does the process work?", "Could you explain how the approval workflow works? I am curious about who needs to sign off on purchase orders.", "QUESTION"),
            ("Please send the report", "Can you send me the updated financial report for last quarter? I need it for the board presentation on Friday.", "REQUEST"),
            ("Kindly review the document", "Please review the attached proposal and provide feedback by end of week. I need your sign-off before sending to the client.", "REQUEST"),
            ("Service is unacceptable", "I am extremely frustrated with the repeated system outages. Three incidents this week alone is completely unacceptable and affecting productivity.", "COMPLAINT"),
            ("Issue with billing", "There is a serious problem with my latest invoice. The charges are significantly higher than the agreed contract rate.", "COMPLAINT"),
            ("Schedule a call?", "Let us set up a meeting to discuss the project roadmap. Are you available for a conference call Thursday at 2pm?", "MEETING_REQUEST"),
            ("Meeting request: Q4 planning", "I would like to schedule a virtual meeting with the full team to kick off Q4 planning. Please check your calendar for Monday availability.", "MEETING_REQUEST"),
            ("Following up on proposal", "I am following up on the proposal I sent two weeks ago. Have you had a chance to review it? We need to finalize before month end.", "FOLLOW_UP"),
            ("Checking in on action items", "Just circling back on the action items from last week's meeting. Could you provide a status update on the vendor evaluation?", "FOLLOW_UP"),
            ("FYI: policy change", "For your information, the company expense policy has been updated effective next month. All international travel now requires VP approval.", "INFORM"),
            ("Update on project", "Just wanted to let you know that Phase 1 of the project has been completed ahead of schedule. We are moving into testing.", "INFORM"),
            ("Random office note", "The coffee machine on floor 3 has been repaired. Also, the parking lot will be closed for maintenance next Saturday morning.", "OTHER"),
            ("Auto-reply: out of office", "I am currently out of office until Monday. For urgent matters please contact my assistant at ext 204.", "OTHER"),
        ]
        rows = [{"message_id": "", "sender": "demo@example.com",
                 "subject": s, "body": b, "intent": i} for s, b, i in samples]
        df = pd.DataFrame(rows)
        out = str(self.output_dir / "emails.csv")
        df.to_csv(out, index=False)
        logger.info("Created synthetic dataset with %d samples at %s", len(df), out)
        return out, df
