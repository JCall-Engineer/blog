---
title: "How I Built jcall.engineer"
tags: []
draft: true
---
## Design Philosophy

This site isn’t powered by a marketing department chasing engagement metrics --- it’s something I built to reflect how I think and work. I don’t develop websites professionally, but I’ve done my best to follow best practices for performance, security, and accessibility. That honesty matters more to me than polish.

I like building things myself, even when it doesn’t make sense --- but I try to have the discipline to reign that in when a good existing solution exists. Whenever possible, I start with existing tools. When they don’t hold up, I allow myself to build something better to get the results I need. That process led to a modular, scalable system that I understand fully and can adapt over time.

The stack behind this site wasn’t chosen at random --- each tool was selected with intention --- not just to solve a problem, but to fit into a system I could reason about, scale, and trust. For a tool to be useful in my ecosystem, it needs to have a clear scope, handle edge cases rigorously, and avoid becoming a black box. I want to be able to reason about its behavior, swap it out cleanly, and integrate it without distortion.

I plan to write more about the black box problem in a future post, but the short version is this: if a tool’s internals are hidden and I’m expected to trust its outputs, then it absolutely must deliver. Too often I’m handed black boxes that don’t work as advertised --- and when they fail, I’m left with fewer options for solving the problem than if I had just built the damn thing myself.

That’s what makes the whole system robust and maintainable --- and why I can expect it to hold up over time.

Here’s what I ended up with:
 - **Cloudflare** --- handles DNS, DDoS protection, and domain registration, all at cost.
 - **DigitalOcean** --- provides a flexible VPS I can scale and customize.
 - **Let's Encrypt** --- issues free SSL certificates to enable secure HTTPS
 - **Nginx** --- serves content with performance and configuration flexibility
 - **Express.js** --- handles dynamic assembly of HTML content using a lightweight Node.js framework.
 - **markdown-it (Python)** --- powers the static site generator that converts Markdown to HTML with minimal tooling.
 - **Custom bash scripts** --- orchestrate my development flow, manage versioning, and allow me to roll back to previous builds when needed.

> Note: I also publish all markdown source files on GitHub, so the complete version history of each blog post is public and transparent. Scroll to the bottom of this blog post to see an example of how that will show up.

---

## Domain Registration
I knew I wanted the domain `jcall.engineer`. When I went to purchase it from Google Domains, I discovered they had been bought out by Squarespace --- which surprised me (Google **selling**, not **buying**?).

After evaluating my options, I chose Cloudflare because:
 - They sell domains *at cost* with no renewal upcharges.
 - Their DNS management is best-in-class.
 - Built-in DDoS protection comes free via their proxy layer.
 - Managing both DNS and DDoS from a single platform made things simpler.

That combination of **value**, **security**, and **convenience** made Cloudflare an easy choice --- especially with their predictable pricing and strong feature set. I pay $27.18 annually, which comes out to about $2.27/month.

## VM Hosting
I had previously hosted on Linode but I wanted to reevaluate modern options. While Linode is very affordable, and I have no complaints about the boxes they offer, I have been frustrated with how limited their interface was for things like port forwarding. I vaguely recall having to contact support because I couldn't figure out how to do basic things with their web interface, but the details are long forgotten.

I ended up choosing DigitalOcean because
 - Most importantly: they are affordable. $6 per month affords me
   - 1 GB RAM
   - 25 GB of SSD storage
   - 1 TB of network transfer per month
 - They have **flexibility** in infrastructure that would allow me to easily scale to a higher performance environment
   - You can upgrade individual components as needed
     - More RAM
     - A faster CPU
     - Extra storage
     - A second machine for load balancing
   - You can ensure faster load times and lower latency for users, no matter where they are --- DigitalOcean offers 9 global regions
     - New York
     - San Francisco
     - Amsterdam
     - Singapore
     - London
     - Frankfurt
     - Toronto
     - Bangalore
     - Sydney

For now, one droplet in San Francisco using the basic hardware is fine --- but having the **option** to scale up is a big plus. I have a VM that I have **full control** over so I can *experiment* freely and evolve the system as my needs change.

#### Out of pocket expenses
| Service      | Annual | Monthly | Notes |
|--------------|--------|---------|-------|
| Cloudflare   | $27.18 | $2.27   | Domain Registration, DNS management, DDoS protection, caching optimizations, other security perks |
| DigitalOcean | $72    | $6      | Hosting with flexibility for upscaling deployment if needed |
| Total        | $99.18 | $8.27   | Monthly cost is an estimate due to annual billing from Cloudflare |

Everything else you see on this website is made by myself or using free open source tools

---

## The Scripts Behind jcall.engineer

### Introduction: Why I Wrote My Own Dev Infrastructure
- Quick recap: this site reflects how I build things — transparent, modular, controlled
- Why off-the-shelf tools didn’t cut it (just a sentence or two)
- What this post will cover: the glue scripts and infrastructure that make this site work

### `utils.sh`: Small Script, Big Leverage
- Logging helpers: `log_info`, `log_success`, etc.
- Safe execution: dry run support with `--no-exec` via `run` wrapper
- YAML helpers for accessing and modifying `environment.yml`

### `domains.sh`: Fully Scripted Domain + SSL Management
- One script to register, renew, and remove domains
- Automatically disables and re-enables Cloudflare proxy for cert issuance
- Everything pulled from `environment.yml` — no need to touch `.bashrc`
- CLI-style breakdown: `register`, `renew`, `remove`, `proxy`, `list`

### `ship.sh`: The Brain of the Production Pipeline
- One CLI for build, link, deploy, and test across all projects
- Supports versioning (`--version`, `--draft`, `--publish`)
- Drafts and publishes to `/deploy/out/[project]/[version]/[publish|draft]`
- Symlink management for `/var/www` and `/etc/nginx`

### `markdown_translator.py`: Clean Markdown-to-HTML with LaTeX Support
- 240 lines of processing logic
- 220 lines of CLI and yq-based metadata handling
- Over 1200 lines of unit tests
- Supports math rendering, attribute parsing, and smart typographer output

### Template System: HTML Injection via Express
- `index.html`, `blog/index.html`, and `blog/post.html` define core layout
- Express dynamically assembles full pages by injecting HTML fragments into templates
- Static site-like performance with dynamic routing flexibility
- Build step controls what content appears — no SPA framework required

### System Services + Web Server Routing
- Ubuntu systemd unit keeps Express alive
- Nginx routes subdomains and forwards requests to Express
- Clean separation of static and dynamic content

### Looking Forward: Features, Projects, and Writing Plans

- I already have **rich LaTeX support**, and I’m excited to finally put it to use. Expect to see things like Sobel filter matrices, set theory notation, or even a well-placed differential equation — because why not show it off?
- I plan to host and document open-source projects here, starting with **Pixonomy**, a FOSS image tagging and management tool designed for flexibility and offline control.
- I also want to start writing more educational content, including:
  - How to use Git (beyond just memorizing commands)
  - Introductory Makefile usage
  - Foundational programming knowledge — tracing from transistors to C++
- You’ll likely see occasional **policy advocacy posts** — particularly around issues like medical patents or copyright — but I want those to be as **party-agnostic** as possible. I’m not interested in tribal debate. I’m interested in identifying practical reforms that people across the spectrum could agree on.
- And beyond that? Whatever holds my interest. I want this site to grow *with me* — evolving alongside the projects, tools, and ideas I care most about.

### Conclusion: Why I Went This Route
- What it’s like to work in this environment now
- Would I recommend it? Who should (and shouldn’t) do the same
- Link to GitHub if you’re open to sharing scripts
