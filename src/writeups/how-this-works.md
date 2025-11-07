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
- Responsive layouts: ensuring usability across screen sizes
- Browser compatibility: supporting legacy or niche browsers
- Accessibility: following WCAG guidelines

I've done my best to follow best practices in these areas, but I'm honest about the gaps. I test in modern Firefox and Chrome - that's the bandwidth I have. And mobile support is admittedly something of an afterthought for me, but I try to consider it.

One of the curses of my psychology when combined with my technical ability is that I struggle to accept third-party tools. It takes rigor, polish, transparency, and well-defined scope to impress me. A tool needs to handle edge cases well, stay within its boundaries, and not become a black box I can't reason about. Anything less, and I get the urge to build my own --- even if it costs time and delays a release. If I think I can do it better in weeks instead of years, I usually try. That said, here are tools I *didn't* try and reinvent:

- Git
- Nginx
- Node.js
- Express
- yq (Go version by mikefarah)
- markdown-it (Python)
- Katex
- jQuery

This approach --- building custom where it matters, trusting proven tools elsewhere --- is what makes the system maintainable. I understand every piece well enough to fix, replace, or extend it.

## The Problem: Infrastructure for Intellectual Honesty

If you've read [my intro post](https://jcall.engineer) you'll understand that the motivation for this site was driven by a need to revise fearlessly and honestly. I wanted edit history to be transparent and accessible --- like Facebook's edit tracking, but public and permanent.

Git was the natural choice for version control. And since Git diffs work best with plain text, markdown became the obvious format for content. HTML diffs are incomprehensible at a glance, but markdown is readable by anyone --- even in raw form. It's widespread enough that people use it daily without realizing: Discord messages, Reddit comments, AI chat interfaces. For public version history (and simplicity of writing), it's perfect.

Making that history accessible is simple: push the blog repo to GitHub and link to it. You can see this at the top of every post --- a direct link to the commit history for that specific file. As for turning markdown into HTML, a quick search suggests Hugo as the "natural" choice for static site generation.

## Why Not Hugo or WordPress?

I started with Hugo. It's fast, well-documented, and widely recommended for markdown-based sites. But it didn't take long to hit friction.

Hugo is opinionated about structure --- where files go, how templates work, how HTML gets generated. That's great when you want what Hugo wants to give you. But I didn't. The online examples all pointed toward "let Hugo do the work," and the result was bloated HTML that gave me the ick. Nested `<div>` wrappers, theme-specific classes everywhere, markup I didn't ask for and couldn't easily strip out.

While web development isn't my passion, I'm strongly opinionated about what makes a good website. Clean, semantic HTML matters. Here's what a blog post looks like on my site:

```html
<body>
	<header class="noselect">
		<nav>
			<ul>
				<li><a href="https://jcall.engineer">Home</a></li>
				<li><a href="https://blog.jcall.engineer">Blog</a></li>
				...
			</ul>
		</nav>
	</header>
	<main>
		<aside id="blog-nav">
			<nav aria-label="Blog Navigation">
				<section id="blog-nav-tag">
					<h2>Browse by Tag</h2>
					<ul>
						...
					</ul>
				</section>
				<section id="blog-nav-recent">
					<h2>Recent Posts</h2>
					<ul>
						...
					</ul>
				</section>
				<a id="blog-nav-all" href="/sitemap">Browse All</a>
			</nav>
		</aside>
		<section id="blog-content">
			<article id="{{slug}}">
				<header>
					<h1>{{blog_title}}</h1>
					<dl class="byline" aria-label="Post metadata">
						...
						<dt>Version History</dt><dd><a href="https://github.com/JCall-Engineer/blog/commits/main/src/..." target="_blank" rel="noopener noreferrer">https://github.com/JCall-Engineer/blog/commits/main/src/...</a></dd>
					</dl>
				</header>
				<section>
					The article
				</section>
			</article>
		</section>
	</main>
	<footer class="noselect">
		...
	</footer>
</body>
```

No unnecessary wrappers. No framework cruft. Just the structure the content actually needs. The `{{variables}}` you see are injection points for my templating system --- more on that later.

Beyond the HTML bloat, I also didn't see a clear way to automate the GitHub commit history links I wanted on every post. I'm sure it's *possible* to extend Hugo to do that --- but at that point, you're fighting the framework instead of using it. The third-party themes added dependencies I didn't want. The templating system felt rigid. And I kept hitting walls where I'd think "I could just write this myself in 50 lines and have exactly what I want."

WordPress was a non-starter for different reasons. It stores content in a database, which makes public version history a significant undertaking. The whole point was Git-based transparency --- WordPress works against that from the ground up.

So I built my own pipeline instead.

## Version-Controlled Deployments: Fearless Experimentation

The core of my deployment system is `ship.sh` --- a bash script that automates fetching, building, and deploying website updates. It's quite lengthy, so I'll show the main structure rather than all ~600 lines.

```bash
# ────── Main ──────
ARGS=()
parse_common_flags ARGS "$@"
set -- "${ARGS[@]}"
get_flags ARGS "$@"
set -- "${ARGS[@]}"

# Handle help early
if [[ "$HELP" == true || "$1" == "help" ]]; then
	print_help
	exit 0
fi

PROJECTS=("${ARGS[@]}")
if [[ ${#PROJECTS[@]} -eq 0 || "${PROJECTS[0]}" == "all" ]]; then
	PROJECTS=( $(yaml_get '.versions | keys | .[]') )
fi

# Track if any action was taken
ACTION_TAKEN=false

if [[ "$SET_PRESET" == true ]]; then
	project_iterator set_preset "$VERSION" "${MODE:-publish}"
	ACTION_TAKEN=true
fi

if [[ "$DO_FETCH" == true ]]; then
	project_iterator git_checkout "$VERSION" "both"
	ACTION_TAKEN=true
fi

if [[ "$DO_BUILD" == true ]]; then
	project_iterator project_builder "$VERSION" "${MODE:-both}"
	ACTION_TAKEN=true
fi

if [[ "$DO_LINK" == true ]]; then
	project_iterator project_linker "$VERSION" "$MODE"
	ACTION_TAKEN=true
fi

# If no action was specified, show help and exit
if [[ "$ACTION_TAKEN" == false ]]; then
	log_error "No action specified"
	print_help
	exit 1
fi
```

`ship.sh` has 3 basic tasks:

- **Fetch**: Pull a specific version (tag or branch) of a project from my Git repository
- **Build**: Copy files to a version-specific directory (*e.g.*: `/deploy/out/blog/v1.2/publish/`) and run any required processing (*e.g.*: markdown <span aria-label="to">→</span> HTML)
- **Link**: Update a symlink (*e.g.*: `/var/www/blog`) to point at the new build

My typical workflow: `./ship.sh blog --full`, which runs all three steps interactively. The `project_iterator` helper applies the requested operation to each project --- or every project in `environment.yml` if I specify `all`. I use this script for all projects on my website, of which blog is just one.

```bash
jcall@jcall-engineer:/jcall.engineer/deploy$ ./ship.sh blog --full
=== ship.sh run started at Thu Oct 30 16:34:16 UTC 2025 ===
Do a git checkout? (y/n): y
Fetching updates from origin in blog
From gitlab.com:jcall.engineer/public-domains/blog
   8b5be4a..c23aeb8  main       -> origin/main
Fetched successfully
Checking out branch main and pulling latest changes
Already on 'main'
Your branch is behind 'origin/main' by 4 commits, and can be fast-forwarded.
  (use "git pull" to update your local branch)
Updating 8b5be4a..c23aeb8
Fast-forward
 src/writeups/how-this-works.md | 232 ++++++++++++++++++-----------------------
 1 file changed, 101 insertions(+), 131 deletions(-)
Branch main up to date
Rebuild the project? (y/n): y
[INFO] The following project will be built: blog version main[publish]
Press any key to continue...
Sourcing markdown files from blog
Successfully sourced blog
Translating markdown files
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/index.md -> /jcall.engineer/deploy/out/blog/main/publish/html/index.html
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/tags/engineering.md -> /jcall.engineer/deploy/out/blog/main/publish/html/tags/engineering.html
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/tags/education.md -> /jcall.engineer/deploy/out/blog/main/publish/html/tags/education.html
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/tags/projects.md -> /jcall.engineer/deploy/out/blog/main/publish/html/tags/projects.html
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/tags/software.md -> /jcall.engineer/deploy/out/blog/main/publish/html/tags/software.html
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/tags/civics.md -> /jcall.engineer/deploy/out/blog/main/publish/html/tags/civics.html
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/research/why-eliquis-costs-so-much.md -> /jcall.engineer/deploy/out/blog/main/publish/html/research/why-eliquis-costs-so-much.html
Processed: /jcall.engineer/deploy/out/blog/main/publish/src/letters/mike-lee-healthcare.md -> /jcall.engineer/deploy/out/blog/main/publish/html/letters/mike-lee-healthcare.html
Wrote map: /jcall.engineer/deploy/out/blog/main/publish/html/sitemap.json
Metadata and Html translation successful
[✓] Build complete for blog (version: main[publish]).
[INFO] The following project will be built: blog version main[draft]
Press any key to continue...
Sourcing markdown files from blog
Successfully sourced blog
Translating markdown files
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/index.md -> /jcall.engineer/deploy/out/blog/main/draft/html/index.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/essays/rethinking-ip.md -> /jcall.engineer/deploy/out/blog/main/draft/html/essays/rethinking-ip.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/essays/teaching-software-right.md -> /jcall.engineer/deploy/out/blog/main/draft/html/essays/teaching-software-right.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/essays/patents-vs-patients.md -> /jcall.engineer/deploy/out/blog/main/draft/html/essays/patents-vs-patients.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/tags/engineering.md -> /jcall.engineer/deploy/out/blog/main/draft/html/tags/engineering.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/tags/education.md -> /jcall.engineer/deploy/out/blog/main/draft/html/tags/education.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/tags/projects.md -> /jcall.engineer/deploy/out/blog/main/draft/html/tags/projects.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/tags/software.md -> /jcall.engineer/deploy/out/blog/main/draft/html/tags/software.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/tags/civics.md -> /jcall.engineer/deploy/out/blog/main/draft/html/tags/civics.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/writeups/proving-bad-luck.md -> /jcall.engineer/deploy/out/blog/main/draft/html/writeups/proving-bad-luck.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/writeups/the-blackbox-problem.md -> /jcall.engineer/deploy/out/blog/main/draft/html/writeups/the-blackbox-problem.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/writeups/how-this-works.md -> /jcall.engineer/deploy/out/blog/main/draft/html/writeups/how-this-works.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/writeups/discord-safety.md -> /jcall.engineer/deploy/out/blog/main/draft/html/writeups/discord-safety.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/ideas/secure-feedback.md -> /jcall.engineer/deploy/out/blog/main/draft/html/ideas/secure-feedback.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/research/why-eliquis-costs-so-much.md -> /jcall.engineer/deploy/out/blog/main/draft/html/research/why-eliquis-costs-so-much.html
Processed: /jcall.engineer/deploy/out/blog/main/draft/src/letters/mike-lee-healthcare.md -> /jcall.engineer/deploy/out/blog/main/draft/html/letters/mike-lee-healthcare.html
Wrote map: /jcall.engineer/deploy/out/blog/main/draft/html/sitemap.json
Metadata and Html translation successful
[✓] Build complete for blog (version: main[draft]).
Update symlinks? (y/n): n
```

This setup enables fearless experimentation:

- **Instant rollback**: If an update breaks something, I can revert by re-linking to the previous build (*e.g.*: `./ship.sh blog --link --version v1.1`)
- **Independent deployments**: I can update the blog without touching other systems, or test experimental templates on a dev subdomain
- **Fast iteration**: The build process is scoped to what changed --- updating the blog doesn't require rebuilding templates, restarting Express, or touching nginx config

## The Markdown Pipeline: Processing Your Way

I started working on this website back in April of 2025. My first approach was to try Hugo: a static site generator that translates markdown content into a complete static site. As of writing this it is now more than half a year later, so there may be some inaccuracy in my recollection of how Hugo worked. But the problems I remember having were:

- I didn't agree with how Hugo composed HTML documents (`<div>` wrappers everywhere)
- Theming and layout adjustments were rigid
- No clear way to inject the GitHub commit history links I wanted

So I wrote a small Python script (~200 lines of logic + 300 lines of supporting structure, 1200 lines of unit tests) that was structure-agnostic, tightly scoped, and did its job well. Here's what it does:

```bash
usage: markdown_translator.py [-h] [-d INPUT_DIR] [-f INPUT_FILE]
       [-@ RELATIVE_ROOT] [-m MAP] [-o OUTPUT] [--draft] [--unit-tests]
       [--dry-run] [--flatten-output] [--force]

Markdown to HTML partial converter

options:
  -d, --input-dir       Path to directory of markdown files (recursive)
  -f, --input-file      Individual markdown file to process
  -@, --relative-root   Base path to compute relative output paths from
  -m, --map             Output a json map/summary of generated files
  -o, --output          Output directory (filenames from slug or source name)
  --draft               Include files with "draft: true" in frontmatter
  --dry-run             Print output paths without writing files
  --flatten-output      Ignore input tree structure
  --force               Allow overwriting existing files
```

The tool is intentionally "dumb" --- it doesn't care about your site structure, it just mirrors your input tree and translates what you tell it to where you tell it to. In practice, `ship.sh` calls it like this:

```bash
python markdown_translator.py \
	-d "$dir/src" \
	-o "$dir/html" \
	-m "$dir/html/sitemap.json" \
	--force
```

When `-d` is the only input source, the script automatically treats it as the relative root --- preserving the directory structure from that point. So `src/writeups/how-this-works.md` becomes `html/writeups/how-this-works.html`. The `--draft` flag gets added conditionally when building the draft version of the site (you saw this in the `ship.sh` output earlier --- it builds both `main[publish]` and `main[draft]`).

The script handles the hard parts --- markdown <span aria-label="to">→</span> HTML via `markdown-it`, LaTeX rendering via `KaTeX`, and frontmatter extraction via `yq` --- and leaves the easy parts (site composition, navigation, templating) to other processes. For each markdown file, it outputs:

- `.html` - The translated content (just the article body, no site chrome)
- `.json` - The extracted metadata (title, tags, dates, slug)
- `sitemap.json` - A hierarchical manifest of all processed files

That sitemap becomes critical in the next step: template composition. The markdown translator doesn't know or care about your site's header, footer, or navigation --- it just gives you clean HTML partials and the metadata needed to build those things elsewhere.

This separation of concerns is what makes the system maintainable. The markdown translator has one job, does it well, and never needs to change unless markdown itself changes. Everything else --- how pages get assembled, styled, or served --- lives in different, independently versioned components.

## Template Composability: Three Layers Deep

After the markdown pipeline produces clean HTML partials and metadata, the next challenge is composing complete pages. I needed something flexible enough to handle shared layouts, reusable components, and page-specific content --- without the rigidity of Hugo's templating or the database dependencies of WordPress.

So I built a custom template system that treats pages as composable layers. The core insight: a blog post isn't one monolithic HTML file --- it's three distinct layers that get merged:

1. **Site layer** (`index.html`) - The outer shell: `<html>`, `<head>`, navigation, footer
2. **Domain layer** (`blog/index.html`) - Blog-specific structure: sidebar navigation, content wrapper
3. **Page layer** (`blog/post.html`) - The actual content: article header, body, metadata

Each layer is just an HTML template with injection points marked by `{{variable}}` syntax. When a request comes in, Express composes these layers on-demand, injecting the markdown-generated HTML and computed metadata where needed.

Here's what that looks like in practice. The site layer defines the overall page structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
	<title>{{title}}</title>
	...
{{head_elems?, 1}}
</head>
<body>
	<header class="noselect">
		<nav>...</nav>
	</header>
	<main>
{{domain, 2}}
	</main>
	<footer>...</footer>
</body>
</html>
```

The `{{domain, 2}}` tells the system: "inject the domain layer here, indented 2 tabs." The domain layer (for the blog) defines the two-column layout:

```html
<aside id="blog-nav">
	<nav>
		<section id="blog-nav-tag">
			<h2>Browse by Tag</h2>
			<ul>
{{nav_tags, 4}}
			</ul>
		</section>
		...
	</nav>
</aside>
<section id="blog-content">
{{page, 1}}
</section>
```

And the page layer is where the actual post content lives:

```html
<article id="{{slug}}">
	<header>
		<h1>{{blog_title}}</h1>
		<dl class="byline">
{{blog_metadata, 3}}
		</dl>
	</header>
	<section>
{{blog_post, 2}}
	</section>
</article>
```

### Why Not Just Use EJS or Handlebars?

Existing templating engines solve similar problems, but they come with baggage I didn't want:

- **Feature creep**: Built-in loops, conditionals, partials, helpers --- most of which I don't need
- **Syntax overhead**: Learning another DSL when basic string injection does the job
- **Indirect dependencies**: Another package to maintain, update, and potentially break

My system is ~200 lines of code in `template.js`. It handles exactly what I need:

**Indentation preservation**: The number after a variable (`{{domain, 2}}`) specifies tab depth, keeping the composed HTML readable instead of collapsing everything to the left margin.

**Dependency resolution**: When the domain layer references `{{page}}`, the system detects it's an alias to another layer, resolves dependencies in the right order, and detects cycles. This keeps templates decoupled --- the site layer doesn't need to know what's inside `{{domain}}`, it just injects whatever that resolves to.

**Optional and conditional injection**: Three patterns handle different cases:

- `{{variable?}}` - If undefined, omit it silently (used for page-specific CSS)
- `{{variable ? value}}` - If truthy, inject `value` (used for checkbox `checked` attributes)
- `{{variable(compare) ? value}}` - If variable equals `compare`, inject `value`

The comparison pattern is what makes the web dashboard for my Discord bot work, letting me write declarative form controls:

```html
<select name="queue_duration">
	<option value="15" {{queue_duration(15) ? selected}}>15 seconds</option>
	<option value="30" {{queue_duration(30) ? selected}}>30 seconds</option>
	<option value="60" {{queue_duration(60) ? selected}}>1 minute</option>
</select>
```

The template system compares `queue_duration` against each value and injects `selected` only where they match. For complex logic that doesn't fit these patterns, that happens in JavaScript before passing data to the template.

When something goes wrong, error messages track the full dependency chain (e.g., `site → domain → page`), making debugging straightforward.

The simplicity pays off. I can read the entire implementation in one sitting. There's no documentation to search through, no edge cases to memorize, no updates to track. It's frozen code that will work unchanged for years.

### Runtime Composition vs. Static Generation

The blog *could* be fully static --- the markdown is pre-built, templates rarely change, and there's no user-specific content. But runtime composition gives me version testing (test new templates on my dev subdomain without rebuilding) and shared infrastructure (one system for both the blog and the Discord dashboard that *does* need user-specific data).

Could I optimize the blog to be fully static? Sure. But the added complexity of maintaining two separate rendering paths didn't seem worth it when runtime composition works fine and costs almost nothing.

## The Details Matter: CSS, Themes, and Sharing

## Closing the Loop: GitHub Integration

## What It Costs (Money and Time)

## Should You Do This?
