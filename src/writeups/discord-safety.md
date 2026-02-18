---
title: "Rui: The Ongoing Battle Against Scam Campaigns on Discord"
tags: [ projects, software ]
draft: true
---

## Rui's Origin: A personal tool for tracking data

I didn’t set out to build a Discord bot. With all the health challenges I’ve faced, one pattern became clear: improvement was always preceded by data collection. I discovered I had Circadian Rhythm Disorder *because* I started tracking my sleep. Instead of trying to explain my experience to a doctor, I could show them a graph:

![Graph of my sleep prior to CRD diagnosis](@assets/sleep.png)

That graph changed the course of my life! It turned something subjective and hard to describe into something concrete and undeniable. Given the other health challenges I face, I wanted a way to track important data with as little friction as possible. The solution I arrived at was a Discord bot that let me log events with a simple slash command:

- When and what I eat
- When I sleep
- Medication changes
- Mood and symptoms
- Accomplishments

Rui started as a deeply personal tool to make data collection effortless. At the time, Rui had nothing to do with scam detection. That came later.

## The Problem: One Account, Every Channel

I moderate a handful of Discord servers of varying sizes. All of them have experienced periodic disturbances from scammers posting the same message in every channel. Sometimes this would happen while I was sleeping, leaving it to spread unchecked. At the time, Discord’s feature to delete a user’s recent messages when banning them would sometimes fail, making cleanup tedious and time-consuming. I had to go into each channel individually and delete the filth by hand.

After doing this for the third or fourth time, I began to think: "I have a Discord bot... I can do something about this!"

By the time Rui was first taking shape, LLMs had already become the default solution for almost every problem. It would have been natural to reach for one to detect scam messages. But I’m a bit old-fashioned and tend to trust ingenuity and algorithms more than statistical soup. The key observation is that these scam campaigns exhibit abnormal and targetable behavior: they post the same message in every channel, not just once. Their effectiveness comes from automation: they can blast the same message across an entire server in seconds without spending their own time doing it. Even if only one person out of hundreds falls for it, the campaign is still profitable.

Discord has rich text-based automatic moderation features. When you enable Discord Community on your server you gain access to a pretty sophisticated automod that can combat single channel spam, and flags messages that are common spam formats. It even lets you set up regex filters for anything else. Because of this scammers have moved from sending links and text messages to sending screenshots which cannot be targeted by automod.

Their strength is automation. Their weakness is repetition.

## The First Attempt: Simple Hash Matching

The first version of scam detection was straightforward: compute an MD5 hash of each message and attachment, and compare those hashes against recent messages from the same user. If the same hashes appeared in multiple channels within a short window, it was likely a scam campaign. MD5 was a natural starting point. It’s simple, widely available, and one of the first hashing algorithms most developers encounter. But MD5 was designed for cryptographic integrity, not performance. It is both slower and weaker than modern alternatives, while offering guarantees Rui didn’t need in the first place.

When cryptographic security isn’t required, non-cryptographic hash functions provide dramatically better performance while still producing reliable fingerprints. Switching to xxHash made an immediate difference. xxHash is designed for speed: it can hash a 50MB attachment in roughly 1.6 milliseconds, faster than the overhead of spawning a thread. This helped, but it didn’t solve the deeper problem. The bottleneck wasn’t just the hash function, it was the architecture. Detection was still running on the same event loop as everything else, which meant every message Rui analyzed delayed other tasks from executing.

## Converging on the Right Architecture

When scam detection was added, it ran in the same event loop as everything else. Rui was built using Python’s asynchronous programming model: where a single event loop coordinates all tasks. Coroutines cooperate by voluntarily yielding control, rather than running in parallel. I offloaded the hash computation itself to a worker thread, but the detection pipeline still lived within the event loop. Each message required scheduling work, awaiting results, and updating internal state. The event loop was still responsible for coordinating every step. Detection remained tightly coupled to Rui’s core execution flow.

This worked, but it introduced a new problem: Rui was no longer as responsive as it used to be, even while running in only a handful of servers: the ones I personally moderated and used for testing. Every message analyzed by the scam guard delayed other tasks from executing. Coming from a C++ background, where multithreading is the default tool for separating workloads, this felt like a natural limitation to work around rather than a design constraint to embrace. If Rui was going to scale beyond my personal use, this architecture wouldn’t hold. Scam detection needed to run without interfering with the responsiveness of user-facing commands.

### The Original OOP Event Override Model

### Moving to a Multithreaded Architecture

### Hitting Memory Limits and Profiling Failures

### Transitioning back to a Single-Threaded Async Architecture

### Redesigning Data Structures for Efficiency

### Making Rui Distributed

## Architecture: Sliding Windows and Guild-Scoped Memory

## When Attackers Adapted: Moving from Exact Matching to Similarity Detection

## Quarantine and Cleanup

## The Funding Problem

## The "Bites" System: Attempting Usage-Based Pricing

## Discovering I Was Wrong About Memory Usage

## Rethinking Premium: Funding Through Support, Not Usage

## Premium Feature: Deleted Message Forensics

## Lessons Learned

## What’s Next
