---
title: "Rui: The Ongoing Battle Against Scam Campaigns on Discord"
tags: [ projects, software ]
draft: true
---

## Rui's Origin: A personal tool for tracking data

I didn’t set out to build a Discord bot. With all the health challenges I’ve faced, one pattern became clear: improvement was always preceded by data collection. I discovered I had Circadian Rhythm Disorder *because* I started tracking my sleep. Instead of trying to explain my experience to a doctor, I could show them a graph:

![Graph of my sleep prior to CRD diagnosis](@assets/sleep.jpg)

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

Discord has rich text-based automatic moderation features. When you enable Discord Community on your server, you gain access to a sophisticated automod system that can combat single channel spam and flag common spam formats. It even lets you set up regex filters for anything else. Because of this, scammers have moved from sending links and text messages to sending screenshots, which cannot be targeted by automod.

Their strength is automation. Their weakness is repetition.

## The First Attempt: Simple Hash Matching

The first version of scam detection was straightforward: compute an MD5 hash of each message and attachment, and compare those hashes against recent messages from the same user. If the same hashes appeared in multiple channels within a short window, it was likely a scam campaign. MD5 was a natural starting point. It’s simple, widely available, and one of the first hashing algorithms most developers encounter. But MD5 was designed for cryptographic integrity, not performance. It is both slower and weaker than modern alternatives, while offering guarantees Rui didn’t need in the first place.

When cryptographic security isn’t required, non-cryptographic hash functions provide dramatically better performance while still producing reliable fingerprints. Switching to xxHash made an immediate difference. xxHash is designed for speed: it can hash a 50MB attachment in roughly 1.6 milliseconds, faster than the overhead of spawning a thread. This helped, but it didn’t solve the deeper problem. The bottleneck wasn’t just the hash function, it was the architecture. Detection was still running on the same event loop as everything else, which meant every message Rui analyzed delayed other tasks from executing.

## Converging on the Right Architecture

When scam detection was added, it ran in the same event loop as everything else. Rui was built using Python’s asynchronous programming model, where a single event loop coordinates all tasks. Coroutines cooperate by voluntarily yielding control, rather than running in parallel. I offloaded the hash computation itself to a worker thread, but the detection pipeline still lived within the event loop. Each message required scheduling work, awaiting results, and updating internal state. The event loop was still responsible for coordinating every step, and detection remained tightly coupled to Rui’s core execution flow.

This worked, but it introduced a new problem: Rui was no longer as responsive as it used to be, even while running in only a handful of servers---the ones I personally moderated and used for testing. Every message analyzed by the scam guard delayed other tasks from executing. Coming from a C++ background, where multithreading is the default tool for separating workloads, this felt like a natural limitation to work around rather than a design constraint to embrace. If Rui was going to scale beyond my personal use, this architecture wouldn’t hold. Scam detection needed to run without interfering with the responsiveness of user-facing commands.

### The Original OOP Event Override Model

My original implementation was built on top of discord.py, which exposes Discord events like `on_ready` and `on_message` as coroutine hooks. Rather than binding all logic directly to those hooks, Rui treated the Discord client as shared context and passed it into a set of modules. Each module could override event handlers, and Rui would forward incoming events to every registered module. This effectively created a second event dispatch layer on top of discord.py’s native event system. The implementation looked something like this:

```python
class RuiContext:
	def __init__(self, client: discord.Client, slash_tree: discord.app_commands.CommandTree):
		self.client = client
		self.slash_tree = slash_tree

class RuiModule:
	def __init__(self, context: RuiContext, **kwargs):
		self.context = context
		self.initialize(**kwargs)

	# Hook for modules to bind commands or set-up external API state before discord.client is initialized
	def initialize(self, **kwargs):
		pass

	# Hook for modules to bind to discord.client.event.on_ready
	async def on_ready(self):
		pass

	# Hook for modules to bind to discord.client.event.on_message
	async def on_message(self, message : discord.Message):
		pass

class SpamGuard(RuiModule):
	def initialize(self, **kwargs):
		self.buffer = deque()

	async def on_message(self, message : discord.Message):
		message_hash = await SpamGuard.hash_message(message.content)
		# etc

modules: List[RuiModule] = [
	RootTasks(context),
	SpamGuard(context)
]

# Event Bindings
@context.client.event
async def on_ready():
	await context.slash_tree.sync(guild=None)

	for module in modules:
		await module.on_ready()

	print(f"Rui is online as {context.client.user}")

@context.client.event
async def on_message(message : discord.Message):
	# Ignore messages Rui sent
	if message.author == context.client.user:
		return

	for module in modules:
		await module.on_message(message)
```

This design made Rui modular. Features could be added as independent components without modifying the core event loop. The spam guard, my personal logging commands, and other functionality all lived side by side as peers in the same system. It also made Rui easy to reason about: each module owned its own state and reacted to events independently, without needing to know about other modules.

However, this architecture came with tradeoffs. Features were tightly coupled to the `RuiModule` contract, and extending Rui sometimes meant modifying the base class itself. As the system grew, the abstraction began to leak, and changes intended for one module could unintentionally affect others. What had started as a clean separation of concerns became a shared execution surface.

Another key challenge was event fan-out. Every incoming message was forwarded to every module, and all of that work was coordinated by the same event loop. Even if individual operations were offloaded to worker threads, the event loop still had to schedule and await each step. As Rui grew, the amount of work performed for each message increased. The spam guard was no longer just observing events --- it was adding measurable overhead to Rui’s core execution flow.

### Moving to a Multithreaded Architecture

As I mentioned earlier, I am more of a C++ developer and think in terms of threads. The single event loop model made all workloads compete for the same execution time, even when their priorities were very different. Slash commands needed to be responsive, statistics needed to be recorded at precise intervals, while spam detection and configuration updates could tolerate some delay. These workloads had fundamentally different latency requirements, but the single event loop forced them to compete equally.

My first architectural shift was to isolate major subsystems into separate threads, each running its own asynchronous event loop. Message processing, slash commands, error reporting, configuration updates, and statistics aggregation were all separated. This ensured that delays in one subsystem would not interfere with others.

The entry point for this architecture looked like this:

#### main.py

```python
import asyncio
import threading
from collections.abc import Callable, Awaitable
from procs.commands import main as commands
from procs.config import main as config
from procs.errors import main as errors
from procs.messages import main as messages
from procs.stats import main as stats

def target_thread(ready: threading.Event, main: Callable[[threading.Event], Awaitable[None]]):
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	loop.run_until_complete(main(ready))

async def run(main: Callable[[threading.Event], Awaitable[None]], timeout: float, label: str):
	ready = threading.Event()
	thread = threading.Thread(
		target=target_thread,
		args=(ready, main),
		daemon=True
	)
	thread.start()
	if not ready.wait(timeout=timeout):
		raise RuntimeError(f"{label} failed to initialize in a timely manner")
	print(f"{label} Initialized")

async def main():
	await run(errors, 5, "Error Handler")
	await run(config, 5, "Mongo Listener")
	await run(stats, 5, "Statistics Aggregator")
	await run(commands, 5, "Commands Module")
	await run(messages, 5, "Messages Module")

if __name__ == '__main__':
	asyncio.run(main())
```

Each subsystem ran independently in its own thread and event loop. The commands thread handled slash commands and user interaction. The messages thread handled spam detection and message analysis. The errors thread monitored exceptions and reported them to me via direct message. The stats thread aggregated metrics on a fixed schedule, and the config thread ran an HTTP server that allowed Rui’s web dashboard to notify the bot when server settings changed.

As an example, the messages subsystem looked like this:

#### procs/messages.py

```python
import discord
import threading
from dataclasses import dataclass
from collections.abc import Callable, Awaitable
from procs.errors import report_exception

@dataclass
class RuiMessageHandler:
	name:       str
	on_ready:   Callable[[],                Awaitable[None]]
	on_message: Callable[[discord.Message], Awaitable[None]]

async def main(ready: threading.Event):
	INTENTS = discord.Intents.default()
	INTENTS.message_content = True
	INTENTS.guilds = True
	INTENTS.messages = True
	INTENTS.members = True

	discord_client = discord.Client(intents=INTENTS)

	from modules.scam_guard import register as scam_guard
	handlers: list[RuiMessageHandler] = [
		scam_guard(discord_client)
	]

	@discord_client.event
	async def on_ready():
		for handler in handlers:
			await handler.on_ready()
		ready.set()

	@discord_client.event
	async def on_message(message: discord.Message):
		# Ignore messages Rui sent
		if message.author == discord_client.user: return
		for handler in handlers:
			try:
				await handler.on_message(message)
			except Exception as err:
				report_exception(f"{handler.name}.on_message", err)

	from utils.discord import start
	await start(discord_client)
```

Each subsystem followed the same pattern: its own Discord client, its own event loop, and its own isolated execution environment.

This separation solved the original problem. Spam detection could run as slowly as needed without affecting command responsiveness. Background tasks like statistics collection and configuration updates no longer competed with user-facing functionality. However, this architecture introduced new problems.

Each thread required its own Discord client, which meant maintaining multiple independent gateway connections to Discord. This increased memory usage and consumed additional gateway and API capacity. I began encountering rate limits more frequently, not because Rui was doing more work, but because that work was now spread across multiple clients.

Sharing state between subsystems also had to follow careful, deliberate contracts. Each thread had its own event loop and its own Discord client, which meant asynchronous state could not be shared directly. Communication between subsystems, such as error reporting, required crossing event loop boundaries. This added coordination overhead and reduced architectural flexibility.

This architecture solved the responsiveness problem, but it did so by fragmenting Rui into loosely connected subsystems. It increased operational complexity, introduced synchronization challenges, and consumed more system and API resources than anticipated. It was clear that I had traded one set of constraints for another, and that threading alone was not the right long-term architecture for Rui.

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
