---
title: "How I Built jcall.engineer"
date: 2025-04-15
draft: true
author: "John Call"
tags: ["core values", "technical"]
---

# Core Values

I value **control**, **flexibility**, and **modularity**, all while keeping an eye on my budget. This website reflects my commitment to maintaining a high standard of ethics — **transparency**, **rigor**, and **clarity** — in both my work and the approach I take to building this online presence.

These values aren’t just abstract concepts; they guide every decision I make. When choosing the tools and services to build my site, I considered how well they would align with these principles. I wanted a tech stack that would be both **affordable** and **scalable**, allowing me to grow and adapt over time without compromising on security or performance. And I wanted my content to be transparent in its presentation with a clear version history so I can be held accountable for my writing.

Here’s how I selected my tech stack:

 - `Cloudflare` for security and domain management
 - `DigitalOcean` for flexible hosting
 - `Let's Encrypt` for secure SSL certificates which improve **trust**
 - `Nginx` for web server performance and **modularity**
 - `GitHub` for version control providing revision **transparency**
 - `Hugo` for static site generation using markdown syntax which is *git-friendly* and **flexible**

These services not only offer the performance I need today but also ensure I can grow and scale my project in the future, with a modular design that allows me to swap out individual components as needed — all while maintaining the values I hold dear.

---

# Choosing a Tech-stack

## Domain Registration

I knew I wanted the domain `jcall.engineer`. When I went to purchase it from Google Domains, I discovered that Google Domains had been bought by Squarespace (what Google **selling** not **buying**?).

After evaluating my options, I chose Cloudflare because:

 - They sell domains *at cost* with no renewal upcharges.
 - Their DNS management is top-notch.
 - They offer built-in DDoS protection via their free proxy service.
 - The *convenience* of managing both DDoS protection and domain registration on one platform was appealing.

This combination of **value**, **security**, and **simplicity** made Cloudflare the clear choice for me — particularly because their predictable pricing and robust security features offer **long-term stability** and **flexibility for potential growth**. I pay $27.18 annually which works out to about $2.27 per month.

## Hosting

I had previously hosted on Linode but I wanted to reevaluate modern options. While Linode is very affordable, and I have no complaint about the boxes they offered, I have been frustated in the past with their confiuration options "outside the box" (such as port forwarding). I vaguely remember being frustrated contacting their support because I couldn't figure out how to do basic things with thier web interface, but the details are long forggoten.

I arrived at DigitalOcean becuase

 - Most importantly: they are affordable. $6 per month affords me
   - 1 GB RAM
   - 25 GB of SSD storage
   - 1 TB of network tranfer per month
 - They have **flexibility** in infrasturcutre that would allow me to easily scale to a higher performance environment
   - You are given a virtual machine that can be swapped out to different hardware trivially
     - Need more RAM?
     - Need a better CPU?
     - Need more storage?
     - Need a second machine to balance the network load?
     - All of these can be upgraded **separately** *as needed*
   - You can ensure faster load times and lower latency for users, no matter where they are — DigitalOcean offers 9 global regions
     - New York
     - San Franscisco
     - Amsterdam
     - Singapore
     - London
     - Frankfurt
     - Toronto
     - Bangalore
     - Sydney

For now, one droplet in San Fransisco using the basic hardware is fine — but having the **option** to scale up is a big plus. I have a VM that I have **full control** over so I can *experiment* with new technologies that better fit my needs.

#### Out of pocket

| Service      | Annual | Monthly | Notes |
|--------------|--------|---------|-------|
| Cloudflare   | $27.18 | $2.27   | Domain Registration, DNS management, DDoS protection, caching optimizations, other security perks |
| DigitalOcean | $72    | $6      | Hosting with flexibility for upscaling deployment if needed |
| Total        | $99.18 | $8.27   | Montly cost is an estimate due to annual billing from cloudflare |

---

## Content

