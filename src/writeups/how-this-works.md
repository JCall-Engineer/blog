---
title: "How I Built a Blog That Versions My Thinking"
description: "How and why I built custom infrastructure for transparent revision - from version-controlled deployments to GitHub integration"
tags: [ projects, software ]
draft: true
---
## Building With Intent

This site reflects how I think and work, not what a marketing department would build. I financed college with web development internships, but it was, decidedly, never my passion. In particular, I get quite a headache having to worry about:

- Security: handling diverse threat models
- Design: CSS and visual aesthetics  
- Responsive layouts: ensuring readability across screen sizes
- Browser compatibility: supporting legacy or niche browsers
- Accessibility: following WCAG guidelines

I've done my best to follow best practices in these areas, but I'm honest about the gaps. I test in modern Firefox and Chrome - that's the bandwidth I have. And mobile support is admittedly something of an afterthought for me, but I try to consider it.

One of the curses of my psychology when combined with my technical ability is that I struggle to accept third-party tools. It takes rigor, polish, transparency, and well-defined scope to impress me. A tool needs to handle edge cases well, stay within its boundaries, and not become a black box I can't reason about. Anything less, and I get the urge to build my own --- even if it costs time and delays a release. If I think I can do it better in weeks instead of years, I usually try. That said, here are tools I *didn't* try and reinvent:

- Git
- Nginx
- Node.js
- Express
- MongoDB
- yq (Go version by mikefarah)
- markdown-it (Python)
- Katex
- jQuery

This approach --- building custom where it matters, trusting proven tools elsewhere --- is what makes the system maintainable. I understand every piece well enough to fix, replace, or extend it.

## The Problem: Infrastructure for Intellectual Honesty

If you've read [my intro post](https://jcall.engineer) you'll understand that the motivation for this site was driven by a need to revise fearlessly and honestly. I wanted edit history to be transparent and accessible --- like Facebook's edit tracking, but public and permanent.

Git was the natural choice for version control. And since Git diffs work best with plain text, markdown became the obvious format for content. HTML diffs are incomprehensible at a glance, but markdown is readable by anyone --- even in raw form. It's widespread enough that people use it daily without realizing: Discord messages, Reddit comments, AI chat interfaces. For public version history (and simplicity of writing), it's perfect.

Making that history accessible is simple: push the blog repo to GitHub and link to it. You can see this at the top of every post --- a direct link to the commit history for that specific file. As for turning markdown into HTML, a quick search suggests Hugo as the "natural" choice for static site generation.
