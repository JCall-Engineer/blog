---
title: "Designing for Trust in a Hostile Workplace"
tags: [ projects, software ]
draft: true
---

# SecureFeedback: A Two-Way Anonymous Messaging System for Corporate Feedback

## Overview
SecureFeedback is a system designed to enable two-way, anonymous communication between employees and managers within an organization. It preserves anonymity through cryptographic separation of trust domains, air-gapped message transport, randomized delivery, and organization-wide behavioral obfuscation.

## Goals
- Preserve complete anonymity of the sender
- Enable encrypted two-way communication without identity leakage
- Ensure trust in kiosk behavior without revealing timing data
- Normalize behavior to prevent correlation attacks
- Enable escalation while preserving anonymity
- Ensure verifiability and trust in hardware provisioning
- Acknowledge that breach of trust in the system itself may render the organization unrecoverable

---

## Core Components

### 1. SecureFeedback App
- Open-source, stateless, networkless application
- Runs entirely offline
- Used to:
  - Generate cryptographic keypairs
  - Compose encrypted messages
  - Decrypt incoming messages
  - Forward and escalate previous conversations using ZK proofs

### 2. SecureFeedbackKey (USB Device)
- Portable USB device with no identifiable metadata (no MAC address, device ID, or persistent ID)
- Used to:
  - Store public key metadata (manager pubkeys)
  - Store encrypted messages to be uploaded
  - Retrieve encrypted replies from the company kiosk
- **Hardware Trust Model:**
  - Devices are provisioned by the company but must be verifiable by the employee
  - Each SecureFeedbackKey includes a built-in hardware hash or certificate
  - SecureFeedback App includes a feature to verify device authenticity against a public transparency ledger or attested firmware hash
  - Users may optionally purchase their own key if they do not trust corporate provisioning

### 3. Company Kiosk
- Offline terminal connected to company intranet
- Performs the following:
  - Reads encrypted messages from SecureFeedbackKeys
  - Queues messages for randomized delivery
  - Stores encrypted replies for later retrieval
  - Never logs USB metadata or timestamps

---

## Message Flow

### Outbound
1. Employee generates a message in SecureFeedback App
2. Message is encrypted with the manager's public key
3. Message includes an ephemeral return key for reply encryption
4. Message is written to SecureFeedbackKey
5. Employee plugs SecureFeedbackKey into the kiosk
6. Kiosk queues message for randomized delivery

### Inbound
1. Manager replies by encrypting response using the embedded return key
2. Reply is queued to the SecureFeedbackKey associated with the return key
3. Employee later retrieves replies at the kiosk
4. Decrypts replies at home using SecureFeedback App and private key

---

## Escalation via Zero-Knowledge Proofs

### Purpose
To allow anonymous users to escalate issues up the managerial chain **without revealing their identity**, while still **proving** they attempted to resolve the issue through proper channels.

### Process
1. The employee creates a zero-knowledge proof that:
   - A message was sent to a specific manager
   - A reply was received
   - The contents match specific criteria (e.g., dismissiveness, lack of engagement) based on a defined escalation schema
2. The SecureFeedback App uses this ZK proof to forward the complaint and response to a higher-level manager
3. The higher-level manager can verify the **existence and contents** of the thread without learning who the sender is

### Benefits
- Protects anonymity while enabling accountability
- Encourages managerial responsibility and responsiveness
- Prevents misuse by requiring cryptographic evidence of actual message history

---

## Timing Attack Prevention

### Randomized Batch Delivery
- Kiosk flushes queued messages at random intervals (e.g., 30-120 minutes)
- Prevents correlation between upload time and delivery time

### Upload Behavior Obfuscation
- Messages are padded and indistinguishable in format
- Dummy messages may be injected to maintain volume

### Optional Kiosk Receipt
- Kiosk signs a tamper-proof receipt of total messages uploaded (not by whom)

---

## Behavioral Anonymity Policy

### Organization-Wide Cover Protocol
- **Every employee plugs in their SecureFeedbackKey** upon arrival and departure
- Employees are encouraged to plug in during lunch or breaks
- SecureFeedback App can generate **dummy messages** to be uploaded automatically
- Dummy messages are indistinguishable from real ones to the kiosk
- Even management follows this routine to prevent behavioral outliers

---

## Trust and Verifiability

### Open Source Clients
- All cryptographic logic and behavior are auditable

### Verifiable Kiosk Behavior (Future Work)
- Zero-Knowledge Proofs that kiosk:
  - Matches deterministic, auditable source code
  - Does not timestamp messages
  - Obeys randomized delivery logic

### Optional Remote Attestation
- TPM/Secure Boot support for reproducible kiosk binaries

### Hardware Trust Chain
- Employees must be able to verify that their SecureFeedbackKey is:
  - Running officially signed firmware
  - Free from tampering or custom logging mechanisms
  - Cryptographically validated against a published manufacturer keyset or public transparency record
- Tools are provided in the SecureFeedback App to run device integrity checks

### Irreparability of Trust Breach
- If employees discover (and they can) that the kiosk software, SecureFeedbackKey, or delivery protocol has been tampered with, **the organization may irreversibly compromise its internal trust environment**.
- A breach of trust in the very system meant to report breaches indicates a **systemic failure of leadership accountability** and may be viewed by employees as unfixable.
- Any whistleblower system must not only function securely — it must be provably incorruptible, or the fallout may be cultural collapse.

---

## Threat Model Considerations
| Threat                                  | Mitigation                                 |
|-----------------------------------------|--------------------------------------------|
| Kiosk logs USB insertions               | Policy + hardware design (no readback)     |
| Camera observes specific user at kiosk  | Randomized batch delivery obfuscates link  |
| Admin targets based on behavior         | Mandatory all-employee participation       |
| Fake USB device / key spoofing          | Optional SecureFeedbackKey checksum switch |
| Lost private key                        | Employee-managed backup / print QR fallback|
| Tampered key issued by employer         | Device attestation via open verification tool|

---

## Status
**Concept / Proposal Stage** — requires prototype, cryptographic spec, and pilot testing.
