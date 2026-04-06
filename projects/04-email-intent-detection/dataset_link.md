# Enron Email Dataset

## Source
- **Name:** Enron Email Dataset
- **Provider:** Carnegie Mellon University (CMU)
- **URL:** https://www.cs.cmu.edu/~enron/
- **Download:** https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

## Overview
The Enron email dataset contains approximately 500,000+ emails from 150 users,
mostly senior management at the Enron Corporation. It was made public by the
Federal Energy Regulatory Commission (FERC) during its investigation of the
Enron scandal and later organized by CMU researchers.

## Schema (Raw Email Fields)
| Field       | Description                              |
|-------------|------------------------------------------|
| Message-ID  | Unique identifier for each email         |
| Date        | Timestamp of when the email was sent     |
| From        | Sender email address                     |
| To          | Recipient email address(es)              |
| Subject     | Email subject line                       |
| X-From      | Sender display name                      |
| X-To        | Recipient display name(s)               |
| Body        | Full email body text                     |

## Why This Dataset Is Useful for Intent Detection
1. **Real-world corporate communication** -- captures authentic business email
   patterns including requests, approvals, scheduling, complaints, and more.
2. **Large scale** -- 500K+ emails provide substantial training data even after
   filtering and deduplication.
3. **Diverse intents** -- corporate emails naturally span many intent categories:
   requesting action, sharing information, scheduling meetings, following up,
   filing complaints, asking questions, granting approvals, and issuing rejections.
4. **Publicly available** -- no access restrictions, widely used in NLP research.

## Intent Labels (Heuristic)
Since the raw dataset has no intent labels, we apply keyword-based heuristic
labeling as an initial bootstrap strategy:

| Intent      | Example Keywords / Patterns                        |
|-------------|----------------------------------------------------|
| request     | please, could you, can you, would you, need you to |
| inform      | FYI, letting you know, update, wanted to share     |
| schedule    | meeting, calendar, schedule, available, conference  |
| follow_up   | following up, checking in, reminder, status update  |
| complaint   | issue, problem, concerned, disappointed, unacceptable |
| inquiry     | question, wondering, curious, what is, how does    |
| approval    | approved, go ahead, granted, authorized, green light|
| rejection   | denied, cannot approve, declined, not possible     |

## Preprocessing Steps
1. **Parse raw email files** -- Extract headers (From, To, Subject, Date) and
   body from each `.txt` file in the maildir structure.
2. **Remove duplicates** -- Deduplicate by Message-ID.
3. **Clean email body** -- Strip forwarded headers, quoted reply chains,
   email signatures, and legal disclaimers.
4. **Filter empty/short emails** -- Remove emails with fewer than 10 words
   in the body after cleaning.
5. **Heuristic labeling** -- Assign intent labels using keyword matching on
   subject + body text. Emails matching no pattern get the `inform` label
   as a default (most common intent in corporate email).
6. **Balance classes** -- Downsample majority classes or upsample minority
   classes to reduce class imbalance.
7. **Train/test split** -- Stratified 80/20 split preserving label distribution.
