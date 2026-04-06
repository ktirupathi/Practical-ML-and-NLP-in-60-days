"""Data ingestion component for parsing raw Enron email files and creating
a labeled dataset using keyword-based heuristic labeling."""

import os
import re
import email
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.config.configuration import DataIngestionConfig
from src.utils.common import create_directories

logger = logging.getLogger(__name__)

# Keyword patterns for heuristic intent labeling.
# Order matters: first match wins, so more specific patterns come first.
INTENT_PATTERNS: Dict[str, List[str]] = {
    "rejection": [
        r"\bdenied\b", r"\bcannot approve\b", r"\bdeclined\b",
        r"\bnot possible\b", r"\bunable to approve\b", r"\breject\b",
        r"\bnot authorized\b", r"\bdo not agree\b", r"\bwill not\b",
        r"\bcannot support\b", r"\bnot feasible\b",
    ],
    "approval": [
        r"\bapproved\b", r"\bgo ahead\b", r"\bgranted\b",
        r"\bauthorized\b", r"\bgreen light\b", r"\byou have my approval\b",
        r"\bsigned off\b", r"\bendorsed\b", r"\bapprove this\b",
        r"\bi agree\b",
    ],
    "complaint": [
        r"\bissue\b", r"\bproblem\b", r"\bconcerned\b",
        r"\bdisappointed\b", r"\bunacceptable\b", r"\bfrustrat\w*\b",
        r"\bescalat\w*\b", r"\bcomplaint\b", r"\bdissatisf\w*\b",
        r"\bnot happy\b", r"\bterrible\b",
    ],
    "schedule": [
        r"\bmeeting\b", r"\bcalendar\b", r"\bschedul\w*\b",
        r"\bavailabl\w*\b", r"\bconference\s*call\b", r"\bappointment\b",
        r"\bbook\s+a\b", r"\breserv\w*\b", r"\b\d{1,2}:\d{2}\b",
        r"\btomorrow\b.*\bmeeting\b", r"\bset up a call\b",
    ],
    "follow_up": [
        r"\bfollowing up\b", r"\bfollow\s*up\b", r"\bchecking in\b",
        r"\breminder\b", r"\bstatus update\b", r"\bjust checking\b",
        r"\bany update\b", r"\bpending\b", r"\bwhere are we\b",
        r"\bcircling back\b", r"\btouch base\b",
    ],
    "request": [
        r"\bplease\b", r"\bcould you\b", r"\bcan you\b",
        r"\bwould you\b", r"\bneed you to\b", r"\bkindly\b",
        r"\bi need\b", r"\brequest\w*\b", r"\bsend me\b",
        r"\bprovide\b", r"\bforward\s+to\s+me\b", r"\bi('d| would) like\b",
        r"\bask(ing)?\s+(you|that)\b",
    ],
    "inquiry": [
        r"\bquestion\b", r"\bwondering\b", r"\bcurious\b",
        r"\bwhat is\b", r"\bhow does\b", r"\bdo you know\b",
        r"\bcan you tell\b", r"\bany idea\b", r"\bis there\b",
        r"\bwhat are\b", r"\bhow do\b", r"\bwhy\b.*\?\s*$",
    ],
    "inform": [
        r"\bfyi\b", r"\bletting you know\b", r"\bupdate\b",
        r"\bwanted to share\b", r"\bfor your information\b",
        r"\bplease note\b", r"\bheads up\b", r"\bjust wanted\b",
        r"\battached\b", r"\bsee below\b", r"\bhere is\b",
    ],
}


class DataIngestion:
    """Parses raw Enron email files, extracts subject and body,
    and creates a labeled dataset using keyword heuristic labeling."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        create_directories([self.config.ingested_data_dir])

    def parse_email_file(self, filepath: str) -> Optional[Dict[str, str]]:
        """Parse a single raw email file and extract key fields.

        Args:
            filepath: Path to a raw email text file.

        Returns:
            Dictionary with message_id, date, sender, to, subject, body
            or None if parsing fails.
        """
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            msg = email.message_from_string(raw_text)

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="ignore")
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="ignore")
                else:
                    body = msg.get_payload() or ""

            return {
                "message_id": msg.get("Message-ID", ""),
                "date": msg.get("Date", ""),
                "sender": msg.get("From", ""),
                "to": msg.get("To", ""),
                "subject": msg.get("Subject", "") or "",
                "body": body.strip(),
            }
        except Exception as e:
            logger.debug("Failed to parse %s: %s", filepath, e)
            return None

    def collect_email_files(self, root_dir: str) -> List[str]:
        """Recursively collect all email file paths under root_dir.

        The Enron maildir structure stores emails as numbered text files
        inside nested folders like: maildir/user/folder/123.

        Args:
            root_dir: Root directory of the extracted Enron maildir.

        Returns:
            List of absolute file paths.
        """
        email_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                if os.path.isfile(full_path):
                    email_files.append(full_path)
        logger.info("Found %d email files under %s", len(email_files), root_dir)
        return email_files

    def assign_intent(self, subject: str, body: str) -> str:
        """Assign an intent label using keyword-based heuristic matching.

        Combines subject and body, checks patterns in priority order.
        Falls back to 'inform' if no pattern matches.

        Args:
            subject: Email subject line.
            body: Email body text.

        Returns:
            Intent label string.
        """
        combined = f"{subject} {body}".lower()

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return intent

        return "inform"

    def ingest(self) -> Tuple[str, pd.DataFrame]:
        """Run the full ingestion pipeline.

        1. Collect all email files from the raw directory.
        2. Parse each file to extract fields.
        3. Deduplicate by Message-ID.
        4. Filter short emails (< 10 words in body).
        5. Assign intent labels via heuristic.
        6. Save to CSV.

        Returns:
            Tuple of (output CSV path, DataFrame).
        """
        raw_dir = self.config.raw_data_dir
        if not os.path.isdir(raw_dir):
            logger.warning(
                "Raw data directory not found: %s. "
                "Creating a small synthetic dataset for demonstration.",
                raw_dir,
            )
            return self._create_synthetic_dataset()

        email_files = self.collect_email_files(raw_dir)
        if not email_files:
            logger.warning("No email files found. Creating synthetic dataset.")
            return self._create_synthetic_dataset()

        max_emails = self.config.max_emails
        if max_emails and len(email_files) > max_emails:
            import random
            random.seed(42)
            email_files = random.sample(email_files, max_emails)
            logger.info("Sampled %d emails for processing.", max_emails)

        records = []
        seen_ids = set()
        for filepath in tqdm(email_files, desc="Parsing emails"):
            parsed = self.parse_email_file(filepath)
            if parsed is None:
                continue

            msg_id = parsed["message_id"]
            if msg_id and msg_id in seen_ids:
                continue
            if msg_id:
                seen_ids.add(msg_id)

            word_count = len(parsed["body"].split())
            if word_count < 10:
                continue

            parsed["intent"] = self.assign_intent(
                parsed["subject"], parsed["body"]
            )
            records.append(parsed)

        df = pd.DataFrame(records)
        logger.info(
            "Ingested %d emails. Intent distribution:\n%s",
            len(df),
            df["intent"].value_counts().to_string(),
        )

        output_path = os.path.join(self.config.ingested_data_dir, "emails.csv")
        df.to_csv(output_path, index=False)
        logger.info("Saved ingested data to %s", output_path)

        return output_path, df

    def _create_synthetic_dataset(self) -> Tuple[str, pd.DataFrame]:
        """Create a synthetic dataset for demonstration and testing
        when the real Enron data is not available."""

        samples = [
            # request
            ("Budget Review Needed", "Please review the Q3 budget report and send me your feedback by end of week. I need your input before the board meeting on Monday.", "request"),
            ("Document Forwarding", "Could you please forward the signed contract to the legal department? They need it for their records as soon as possible.", "request"),
            ("Help with Presentation", "Can you help me prepare the slides for the client presentation next Tuesday? I need the financial charts updated with latest numbers.", "request"),
            ("Action Required: Report Submission", "I need you to submit the monthly compliance report by Friday. Please include all transaction data from the past 30 days.", "request"),
            ("Send Updated Files", "Would you kindly send me the updated project files? The ones I have are from last month and the client is asking for current numbers.", "request"),
            ("Data Export Request", "Please export the customer database records for accounts created in Q2 and share them with the analytics team by Thursday.", "request"),
            ("Approval Needed for Travel", "Can you approve my travel request for the Chicago conference? I need to book flights before the prices go up.", "request"),
            ("Printer Setup", "Could you set up the new printer on the third floor? The old one has been out of service for a week now.", "request"),
            # inform
            ("FYI: Policy Change", "Just letting you know that the company travel policy has been updated effective next month. All international trips now require VP approval.", "inform"),
            ("Update on Project Alpha", "Wanted to share a quick update on Project Alpha. We completed the first phase ahead of schedule and are now moving to testing.", "inform"),
            ("New Team Member", "For your information, Sarah Johnson will be joining our team next Monday as a senior analyst. Please make her feel welcome.", "inform"),
            ("Q3 Results Summary", "Here is a summary of our Q3 financial results. Revenue was up 12 percent year over year and operating costs decreased by 3 percent.", "inform"),
            ("Office Relocation Notice", "Please note that our downtown office will be relocating to the new building on Oak Street starting January 15th. More details to follow.", "inform"),
            ("System Maintenance Notice", "Heads up that the email server will be down for maintenance this Saturday from 10pm to 2am. Please plan accordingly.", "inform"),
            ("Quarterly Newsletter", "Attached is the quarterly newsletter with updates from all departments. Please review and share with your teams.", "inform"),
            ("Policy Update Reminder", "Just wanted to let everyone know that the updated expense policy is now live on the intranet. See below for the key changes.", "inform"),
            # schedule
            ("Meeting Tomorrow at 2pm", "Let us schedule a meeting for tomorrow at 2pm to discuss the quarterly results. Conference room B is available.", "schedule"),
            ("Conference Call Setup", "I would like to set up a conference call with the Tokyo team this Thursday at 9am EST. Please confirm your availability.", "schedule"),
            ("Rescheduling Weekly Sync", "The weekly team sync has been moved from Tuesday to Wednesday at 11am. Please update your calendars accordingly.", "schedule"),
            ("Lunch Meeting", "Are you available for a lunch meeting on Friday? I want to discuss the new hiring plan over at the restaurant across the street.", "schedule"),
            ("Board Meeting Date", "The board meeting is scheduled for March 15th at 10am in the main conference room. Attendance is mandatory for all directors.", "schedule"),
            ("Interview Scheduling", "We need to schedule interviews for the three shortlisted candidates. Are you available next Monday and Tuesday afternoon?", "schedule"),
            ("Training Session", "A training session on the new CRM system has been scheduled for next Wednesday from 2pm to 4pm in the training room.", "schedule"),
            ("Calendar Invite", "Please accept the calendar invite for the project kickoff meeting on Thursday at 3pm. We will be reviewing the project charter.", "schedule"),
            # follow_up
            ("Following Up on Proposal", "I am following up on the proposal I sent last week. Have you had a chance to review it? We need to finalize by end of month.", "follow_up"),
            ("Checking In: Action Items", "Just checking in on the action items from our last meeting. Could you provide a status update on the vendor evaluation?", "follow_up"),
            ("Reminder: Deadline Friday", "This is a reminder that the project deliverables are due this Friday. Please ensure all documents are uploaded to the shared drive.", "follow_up"),
            ("Status Update Request", "Circling back on the infrastructure migration. Where are we on the database transfer? The original deadline was last Tuesday.", "follow_up"),
            ("Pending Invoice", "Following up on invoice 4523 that was submitted three weeks ago. It is still showing as pending in our system. Any update?", "follow_up"),
            ("Re: Contract Review", "Just touching base on the contract review. Has legal had a chance to go through the terms? We need to respond to the vendor.", "follow_up"),
            ("Any Progress on Hiring", "Checking in on the hiring process for the engineering positions. Have we received any strong applications yet?", "follow_up"),
            ("Outstanding Tasks", "This is a reminder about the outstanding tasks from the sprint review. Please update the tracking board with your progress.", "follow_up"),
            # complaint
            ("Service Quality Issue", "I am very disappointed with the service quality from our vendor. The last three deliveries have been late and below standard.", "complaint"),
            ("System Outage Concerns", "The repeated system outages are becoming unacceptable. We have had three incidents this month alone and it is affecting productivity.", "complaint"),
            ("Billing Discrepancy", "I have a serious issue with the latest invoice. The charges are significantly higher than what was agreed upon in our contract.", "complaint"),
            ("Frustrated with Process", "I am extremely frustrated with the current approval process. It takes over two weeks to get a simple purchase order approved.", "complaint"),
            ("Team Communication Problems", "I am concerned about the communication breakdown between engineering and sales. Customers are being promised features we cannot deliver.", "complaint"),
            ("IT Support Complaint", "The IT support response time has been terrible lately. My laptop issue has been open for five days with no resolution in sight.", "complaint"),
            ("Vendor Performance", "The vendor performance this quarter has been very disappointing. Quality has dropped significantly and deadlines are consistently missed.", "complaint"),
            ("Escalation: Unresolved Bug", "I need to escalate the unresolved bug in the billing system. It has been reported three times and is causing incorrect charges to clients.", "complaint"),
            # inquiry
            ("Question About Benefits", "I have a question about the new health benefits package. What is the coverage for dental and vision under the premium plan?", "inquiry"),
            ("How Does the New System Work", "I am wondering how the new expense reporting system works. Do we still need manager approval for amounts under 500 dollars?", "inquiry"),
            ("Curious About Promotion Criteria", "I am curious about the criteria for promotion to senior engineer. Is there a formal rubric or evaluation process?", "inquiry"),
            ("What Is the Policy", "What is the company policy on remote work for employees in the first 90 days? I could not find this in the handbook.", "inquiry"),
            ("Information Needed", "Do you know if there is a budget allocated for team building events this quarter? I would like to organize something for our group.", "inquiry"),
            ("Technical Question", "Can you tell me what version of Python we are standardizing on for new projects? I want to make sure my environment is correct.", "inquiry"),
            ("Clarification Needed", "Is there a minimum notice period for taking vacation days? I want to plan a trip but am not sure about the process.", "inquiry"),
            ("Process Question", "How do I submit a reimbursement claim for the conference I attended last month? I cannot find the form on the intranet.", "inquiry"),
            # approval
            ("Re: Budget Approval", "The budget for Project Beta has been approved. You are authorized to proceed with vendor selection and initial procurement.", "approval"),
            ("Go Ahead on Hiring", "Go ahead and extend the offer to the candidate. I have reviewed the compensation package and it looks appropriate.", "approval"),
            ("Travel Request Approved", "Your travel request to the San Francisco conference has been approved. Please book through the corporate travel portal.", "approval"),
            ("Signed Off on Design", "I have signed off on the new website design. The team can proceed with development. Great work on the mockups.", "approval"),
            ("I Agree with the Plan", "I agree with the proposed timeline and resource allocation. Let us move forward with Phase 2 as outlined in your document.", "approval"),
            ("Expense Report Approved", "Your expense report for the client dinner has been approved and will be reimbursed in the next pay cycle.", "approval"),
            ("Green Light for Launch", "After reviewing the QA results, I am giving the green light for the product launch next Monday. All critical bugs have been resolved.", "approval"),
            ("Authorization Granted", "You are authorized to access the production database for the migration project. Your credentials will be sent separately.", "approval"),
            # rejection
            ("Re: Budget Increase", "Unfortunately, the request for a budget increase has been denied. We need to work within the current allocation for this quarter.", "rejection"),
            ("Cannot Approve Overtime", "I cannot approve the overtime request at this time. We have already exceeded our overtime budget for the month.", "rejection"),
            ("Proposal Declined", "After careful review, the committee has declined the proposal for the new office expansion. The cost analysis does not support it.", "rejection"),
            ("Not Possible This Quarter", "Expanding the team is not possible this quarter due to the hiring freeze. We can revisit this in Q2 when the freeze is lifted.", "rejection"),
            ("Unable to Approve Purchase", "I am unable to approve the software purchase. The vendor does not meet our security compliance requirements.", "rejection"),
            ("Request Denied", "Your request to transfer to the London office has been denied due to current staffing needs. We can discuss alternatives.", "rejection"),
            ("Will Not Proceed", "We will not proceed with the proposed marketing campaign. The projected ROI does not justify the expenditure at this time.", "rejection"),
            ("Not Authorized", "The third party vendor is not authorized to access our customer data. Please find an alternative solution that meets compliance.", "rejection"),
        ]

        records = []
        for subj, body, intent in samples:
            records.append({
                "message_id": "",
                "date": "",
                "sender": "demo@example.com",
                "to": "recipient@example.com",
                "subject": subj,
                "body": body,
                "intent": intent,
            })

        df = pd.DataFrame(records)
        output_path = os.path.join(self.config.ingested_data_dir, "emails.csv")
        df.to_csv(output_path, index=False)
        logger.info(
            "Created synthetic dataset with %d samples at %s",
            len(df), output_path,
        )
        return output_path, df
