---
title: "How an Engineer Budgets"
description: ""
tags: [ projects, software ]
date_effective: 2026-07-04
draft: true
---

## Motivation

I have been writing software since I was 13 years old. I am now almost 34. When I was an adolescent, my mother worked for a company that taught people how to invest in stocks, options, futures, forex, and more. Financial education was part of my childhood --- and naturally, one of my first software projects was budgeting software.

Around that time, budgeting software was transitioning from desktop applications to cloud-based services, and companies were clamoring to copy Mint's success. So many budgeting websites emerged, and I watched most of them wither into obscurity. Nearly all of them were very simple, and their idea of a budget was limited to categorizing expenses and reserving a fixed dollar amount each month for each category. None of them were a tool for planning for the future the way I had been taught.

I first wrote my budgeting software as a Windows Forms application in Visual Basic .NET and --- if I recall correctly --- stored the data as an XML file. A few years later, I rewrote it in C#, still using Windows Forms, but switched from XML to binary serialization for the save data. A few years after that, I lost the source code but continued using the software myself.

Then a friend asked me to help him build a budget in Excel. That led me to recreate my system in VBA. I used that version for a while, but Excel felt too rigid for what I needed. Around this time, I was also managing my own income as a college student, paid by the hour while juggling school and inconsistent paychecks. Managing an hourly income while balancing school made it increasingly apparent that a budgeting system built around predictable monthly income was a poor fit for many people. Life is messy, and I wanted budgeting software that reflected that reality rather than an idealized one.

Fast forward to 2019 --- I had a stroke during my final semester of college. I limped toward the finish line and graduated, but the next several years were dedicated to recovery and discovering new diagnoses that affected my life. I never established a stable career in that time. By 2025, when my situation started to improve dramatically, I began thinking about how to reintegrate myself into the job market.

I built this website and created a Discord bot, but --- as an electrical engineer --- JavaScript, CSS, and Python are not exactly the skills I want to showcase to a potential employer. They are skills I picked up while putting myself through college doing web development, but web development was never my passion. Those projects were valuable for shaking the dust off my programming skills, but I wanted something closer to home: a project at a slightly lower level using C++, a language I considered one of my strongest skills but hadn't touched in seven years. When I last used it, C++14 was the current standard.

FundOS was born at the intersection of those interests: a budgeting system I had rewritten a dozen times throughout my life that didn't exist anywhere else; a desire for local control over my data instead of relying on a cloud service or subscription; a way to ease myself back into the skill set I had let grow rusty over the years; a portfolio project to demonstrate my abilities; and an opportunity to reconnect with how C++ had evolved since I last worked with it.

## What Makes FundOS Different

Most budgeting tools force a choice: track spending after the fact, or commit to fixed monthly allocations decided in advance. YNAB popularized "give every dollar a job," which is the right idea, but the execution assumes a fairly predictable paycheck arriving on a fairly predictable schedule. Mint and its many imitators went the other direction entirely --- they categorized spending after it happened, which is useful for seeing where money went but does nothing to help you decide where it should go next.

Neither model handles irregular income gracefully. Consider someone paid biweekly who also picks up occasional freelance or substitute work. In YNAB, every dollar gets assigned to a category the moment it arrives, and each category is filled to a target amount. That works fine for a regular paycheck. But a freelance payment isn't a regular paycheck --- it's a windfall of unpredictable size arriving at an unpredictable time. Forcing it through the same fixed-category assignment either means manually recalculating what portion should go where every single time, or dumping it all into one catch-all category and losing the intentionality budgeting is supposed to provide in the first place.

FundOS handles this by making a budget a collection of rules rather than a fixed schedule. A budget is built from phases --- fixed or percentage --- executed in order, and nothing about applying that rule cares how much the transaction is or when it arrived. Ten percent to savings is ten percent whether the paycheck is $2,400.00 or $270.00. A fixed phase claiming $800.00 for rent claims exactly $800.00 whether that's easy or a stretch, and overdraw settings decide what happens when it's a stretch.

![The Paycheck budget editor showing three phases: a percentage phase with Emergency Savings capped at $10,000 and uncapped Financial Freedom, a fixed phase with eight expense targets including Rent, Food, and Utilities, and a percentage phase splitting the remainder across Long Term Spending, Flex, Dating, and Charity](@assets/demo_budget.png)

More importantly, you are not locked into one budget. FundOS lets you build multiple budgets --- multiple strategies for giving your money its job --- and apply whichever one fits the income that just arrived. A regular paycheck might run through a budget with a full savings and fixed-expense structure; a freelance payment or a substitute teaching shift, with no fixed bills riding on it, can run through a leaner budget that's pure savings and goals. The dollars aren't forced through the same rule just because they landed in the same account.

This is the difference between a budgeting tool that assumes a predictable financial life and one that assumes a real one. Most people's income isn't perfectly regular. FundOS doesn't ask you to pretend it is.

## Designing an Architecture

Before writing any budgeting logic, I had to decide how the pieces would fit together, and that decision mattered more this time than it had in any earlier rewrite. Earlier versions were tied to one platform because the language chose it for me: Windows Forms meant Windows, VBA meant Excel, and the data format followed whatever that platform made easiest, an XML file, then binary serialization. This time I wanted the actual logic, the phases, the targets, the allocation math, to exist once and be usable no matter the platform, and I wanted the data format to outlive any particular version of the app.

That pushed me toward a core library written in platform-agnostic C++20, with thin clients on top of it. Qt handles the desktop GUI today, and Kotlin over JNI is the plan for Android. The core library knows nothing about Qt or the JVM. It takes input, talks to SQLite, and returns results. The GUI layer's job is to display those results and call back into core when the user does something, nothing more. Business logic that ends up in a GUI file is a bug, not a style choice.

SQLite was an easy call once I framed the problem that way. I'd stored budgets as XML and later as binary serialization, and both approaches meant the format was only ever as good as the code I'd written to read and write it. SQLite is a single file, requires no server, and enforces its own consistency, foreign keys, constraints, transactions, without me hand rolling any of it. It has also been in continuous, heavily tested production use for over two decades, which is not a track record I could match by writing my own format no matter how carefully. Betting my users' financial data on a format I invented myself felt like exactly the wrong kind of confidence.

CMake was the natural build system for something targeting multiple platforms, but coming from a Makefile background, it fought me for longer than I expected. Make has one mental model: name a target, and it figures out what needs rebuilding to produce it. CMake splits that into two distinct steps, configure and build. I kept expecting `--build` to work like `make`, where naming a target builds just that target and its dependencies. Instead, a preset's build step builds everything the preset configures. I didn't yet know `--target` existed to narrow that down, so for a while I was treating presets themselves as if they were targets, when they were really just bundled configure and build commands.

## Relearning C++

The last time I'd written serious C++, C++11 was what I was actually using day to day, even though C++14 had already landed. Coming back to it years later meant catching up on more than one generation of changes at once, and a few features stood out enough that I wanted to write about them.

The one that impressed me most was `std::variant` paired with `std::visit`. A variant is a type-safe union: it holds exactly one of a fixed set of alternative types at a time, and you can only get at the value by proving you handled every alternative. `std::visit` is how you prove it --- it forces you to supply a callable that covers every type the variant could hold. Miss one and the code doesn't compile. I do not envy the people who write the compiler for this.

I used this pattern directly for locale handling. Both currency and percentage formatting need to support a fixed set of known locales as well as a one-off custom configuration, and a variant expresses that distinction cleanly:

```cpp
struct selection {
	std::variant<const currency_locale_entry*, spec> raw;

	const char* identifier() const {
		if (std::holds_alternative<const currency_locale_entry*>(raw)) {
			return std::get<const currency_locale_entry*>(raw)->identifier;
		}
		return custom_id;
	}
};
```

The other surprise was smaller but just as satisfying: generic lambdas. Writing auto as a parameter type turns the lambda into a templated callable, which I hadn't internalized was possible until I was already using it. This turned out to be exactly the tool I needed for functions that operate identically on fixed-phase and percentage-phase targets despite the two being distinct types:

```cpp
static auto collect_target_ids = [](auto* phase) -> std::vector<int64_t> {
	std::vector<int64_t> preserve_ids;
	phase->each_target([&preserve_ids](int, auto* target) {
		if (target->is_persisted()) {
			preserve_ids.push_back(target->id_);
		}
	});
	return preserve_ids;
};
```

One function, both phase types, no manual overloading. It was only after I'd already leaned on this pattern a few times that I learned generic lambdas actually shipped in C++14, not C++17 or C++20 as I'd assumed while catching up. A small reminder that "new to me" and "new" are not the same thing.

Another "new to me" feature that I leaned on constantly, to the point of it becoming a reflex, was `std::optional`. Anywhere a value might legitimately be absent --- a manually created transaction won't have a bank-assigned identifier, a target with no ceiling on how much it can accumulate --- optional says so directly in the type. No sentinel values, no separate boolean flag tracked alongside a value that might be garbage. Once I started using it I had a hard time going back to anything else for representing "this might not exist":

```cpp
struct fixed_target : db_managed {
	int64_t fund_id = 0;
	currency amount;
	std::optional<currency> cap;
	bool allow_overdraw = false;
};
```

If cap holds a value, the target stops claiming money once its fund reaches that balance. If it doesn't, there is no ceiling at all, the target claims its full amount every time regardless of the fund's balance. The absence of a cap is not a special number or a flag that has to agree with some other field, it's simply nothing there.

## My Primitive Types

Maybe this speaks to my inclination for systems programming, but I don't trust floating point math. In FundOS I express every number as an integer type. This makes a lot of sense for currency types where money is on the line and precision matters, but I think my obsession is a little more...pathological.

If FundOS ever needed to handle very large or very small scientific values, I'd reach for the same instinct: write a type that tracks significant digits and stores an integer underneath, rather than reach for a float and hope the rounding stays out of the way. Integers are predictable. Floating point is a negotiation.

Integers also make for interesting design decisions: what level of precision is important to you is reflected in how you design your numbers. Currency in FundOS is stored as `minor_units`, an `int64_t` counting the smallest denomination a locale uses, cents for USD, yen for JPY. The scale a locale defines, 100, 1, or any power of ten, decides how many of those minor units make up one major unit. Percentage works the same way but at a different resolution: a percentage is stored as `basis_points`, an `int32_t` where 10000 means 100%.

### Currency Type

```cpp
struct currency {
	/// The amount in minor units: cents, pence, yen, etc.
	int64_t minor_units = 0;

	static std::optional<currency> from_string(const std::string& text, const currency_locale::spec& locale) {
		auto parsed = parse_currency(text, locale);
		if (!parsed) { return std::nullopt; }
		return currency{*parsed};
	}
	std::string to_string(const currency_locale::spec& locale) const { return format_currency(minor_units, locale); }

	// Convert operator allows this to be used like a true primitive type
	constexpr explicit operator int64_t() const { return minor_units; }
	constexpr currency operator-() const { return { -minor_units }; }

	constexpr currency operator+(const currency& rhs) const { return { minor_units + rhs.minor_units }; }
	constexpr currency operator-(const currency& rhs) const { return { minor_units - rhs.minor_units }; }
	constexpr currency operator*(int64_t rhs) const {                                                                             return { minor_units * rhs }; }
	constexpr currency operator%(int64_t rhs) const { FUNDOS_REQUIRE_OR_FALLBACK(rhs != 0, "cannot modulo by zero", currency{0}); return { minor_units % rhs }; }
	constexpr currency operator/(int64_t rhs) const { FUNDOS_REQUIRE_OR_FALLBACK(rhs != 0, "cannot divide by zero", currency{0}); return { minor_units / rhs }; }

	constexpr currency& operator+=(const currency& rhs) { minor_units += rhs.minor_units; return *this; }
	constexpr currency& operator-=(const currency& rhs) { minor_units -= rhs.minor_units; return *this; }
	constexpr currency& operator*=(int64_t rhs) {                                                                       minor_units *= rhs; return *this; }
	constexpr currency& operator%=(int64_t rhs) { FUNDOS_REQUIRE_OR_FALLBACK(rhs != 0, "cannot modulo by zero", *this); minor_units %= rhs; return *this; }
	constexpr currency& operator/=(int64_t rhs) { FUNDOS_REQUIRE_OR_FALLBACK(rhs != 0, "cannot divide by zero", *this); minor_units /= rhs; return *this; }

	constexpr bool operator==(const currency& rhs) const { return minor_units == rhs.minor_units; }
	constexpr bool operator!=(const currency& rhs) const { return minor_units != rhs.minor_units; }
	constexpr bool operator< (const currency& rhs) const { return minor_units <  rhs.minor_units; }
	constexpr bool operator> (const currency& rhs) const { return minor_units >  rhs.minor_units; }
	constexpr bool operator<=(const currency& rhs) const { return minor_units <= rhs.minor_units; }
	constexpr bool operator>=(const currency& rhs) const { return minor_units >= rhs.minor_units; }
};
```

One more thing worth mentioning: I know about the spaceship operator, operator<=>, which can generate all six comparison operators from a single definition. I don't use it. Writing each comparison out explicitly costs a few extra lines, but it means every operator is something I chose deliberately rather than something the compiler derived on my behalf. For a type as small as currency, spelling it out isn't a burden, it's just clarity.

It's good to think about what numbers can be represented when you're defining types. On the extreme end, the largest dollar amount FundOS can express with this setup for USD, which has scale = 100, is $\frac{2^{63}}{100}$ since `int64_t` is a signed integer. This comes out to roughly $92,233 trillion, and Elon Musk has a net worth barely over a trillion dollars. If he converted his wealth into Iraqi dinar, one of the few currencies using a scale of 1000, at roughly 1,300 dinars to the dollar, his fortune would come out to about $1.43 \times 10^{18}$ minor units, still only around 15% of what a single int64_t balance can hold. Even the richest person alive would need to be roughly six and a half times wealthier before a FundOS account, in the weakest common currency it supports, would overflow.

### Currency Locales

The design of `currency` actually went through a bigger shift than the final version suggests. My first pass made currency a template class, parameterized on a locale type constrained by a concept. That worked until I wanted to let users define a custom locale at runtime, a locale a user builds by picking their own separators and symbol placement rather than selecting a preset. A template parameter is fixed at compile time, so it can't hold something a user configures while the program is running.

The fix was realizing `currency` didn't need the locale at all for anything except turning itself into a string. Addition, subtraction, comparison, none of that cares what symbol you use or even where the decimal separator goes. Even the scale property of the locale --- how many minor units make up a major unit --- doesn't affect mathematical operations when you work in minor units. The math is *identical* no matter the locale. So instead of templating the whole type on a locale, currency stays a plain struct wrapping an `int64_t`, and only `to_string` and `from_string` accept a `currency_locale::spec` as a parameter. The locale becomes something currency briefly borrows to format itself, not something baked into its identity. And, naturally, having preset locale options *or* a custom locale makes a great use case for my new favorite C++ feature: `std::variant`.

```cpp
namespace fundos::currency_locale {

struct spec {
	enum class symbol_placement : uint8_t {
		before,
		after
	};
	enum class negative_notation : uint8_t {
		leading_minus,
		trailing_minus,
		parentheses,
		angle_brackets,
	};

	/// Number of minor units per major unit (e.g. 100 for cents, 1 for currencies with no minor unit).
	/// Invariant: must be a power of 10 (1, 10, 100, 1000, ...). Enforced by the database and GUI layers;
	/// parse_currency/format_currency assume it and will misbehave silently if violated.
	int16_t scale;
	std::string symbol;
	char thousands_separator;
	char decimal_separator;
	symbol_placement symbol_position;
	negative_notation negative_format;
};

struct currency_locale_entry {
	const char* identifier;
	spec info;
};

/// The unit of exchange between the database and GUI for locale data:
/// the DB persists a selection and returns a selection, which the GUI consumes directly without needing to know which case it's in.
/// Wraps either a pointer to a known locale (so registry updates are reflected automatically) or a one-off custom spec.
/// identifier() doubles as the DB's serialization discriminator: a preset's identifier (e.g. "USD") means look it up by name;
/// "Custom" means reconstruct the spec field-by-field from individual meta settings instead.
struct selection {
	std::variant<const currency_locale_entry*, spec> raw;

	selection(const currency_locale_entry* entry) : raw(entry)  {}
	selection(const spec& custom)                 : raw(custom) {}

	static constexpr const char* custom_id = "Custom";
	const char* identifier() const {
		if (std::holds_alternative<const currency_locale_entry*>(raw)) {
			return std::get<const currency_locale_entry*>(raw)->identifier;
		}
		return custom_id;
	 }
	 const spec& info() const {
		if (std::holds_alternative<const currency_locale_entry*>(raw)) {
			return std::get<const currency_locale_entry*>(raw)->info;
		}
		return std::get<spec>(raw);
	 }

	 bool is_preset() const { return std::holds_alternative<const currency_locale_entry*>(raw); }
	 bool is_custom() const { return !is_preset(); }
};

struct data {
	currency_locale_entry USD {
		"USD", {
			.scale                = 100,
			.symbol               = "$",
			.thousands_separator  = ',',
			.decimal_separator    = '.',
			.symbol_position      = spec::symbol_placement::before,
			.negative_format      = spec::negative_notation::parentheses,
		}
	};
	// ...
};
static constexpr std::size_t num_locales = sizeof(data) / sizeof(currency_locale_entry);
static_assert(sizeof(data) % sizeof(currency_locale_entry) == 0, "currency locale data must contain only entry members with no padding");

/// Allows both named (locales.USD) and indexed (locales.entries[i]) access to the same locale data without duplicating it.
/// Relies on `data`'s members being identically-typed entries with no padding between them (enforced by the static_assert above)
/// if that assumption breaks, entries[] silently reads garbage or misaligned data instead of failing to compile.
/// Explicit destructor required because info contains a non-trivial member (symbol), so the active union member must be explicitly destroyed.
union registry {
	currency_locale_entry entries[num_locales];
	data named;
	~registry() { named.~data(); } // non-trivial because spec::symbol is a std::string; see locale.cpp for the full tradeoff this forces
};
extern const registry locales;

static inline const currency_locale_entry* get_locale(std::string_view identifier) {
	for (std::size_t i = 0; i < num_locales; i++) {
		if (locales.entries[i].identifier == identifier)
			return &locales.entries[i];
	}
	return nullptr;
}

} // namespace fundos::currency_locale
```

The `selection` type is where the variant actually earns its keep. A locale is either one of the known presets, USD, EUR, and so on, or a custom spec the user built by hand. Those are fundamentally different cases: a preset is a pointer into a table that the app owns and can update later, while a custom spec is a value the user configured and nobody else is tracking. `std::variant` expresses that distinction directly in the type instead of through a boolean flag and a pair of fields where only one is ever meaningful. `is_preset()` and `is_custom()` are just `std::holds_alternative` underneath, and `identifier()` doubles as the serialization discriminator: a preset's name, or the literal string "Custom" when there's no preset to name.

The registry itself uses a different trick for a different problem. Every preset locale needs two ways to reach it: by name, so code can write `locales.named.USD` and read like plain English, and by index, so code can loop over "every known locale" without hardcoding each one. A `union` lets both views share the same memory: `data` gives named field access, and `entries[]` reinterprets that same block as an array. The `static_assert` next to it exists because this only works if every member of `data` is identical in layout with no padding between them; if that assumption ever broke, `entries[]` would start reading garbage instead of failing to compile.

### Percentage Type

Percentage's basis points give four digits of precision after the decimal point, enough to express something like 33.33% precisely, without ever touching a float. But percentage exists for more than storage. FundOS needs to actually multiply a percentage against a currency, take 10% of a paycheck, split a remainder three ways, and that operation is where integer math gets dangerous if you're not careful.

```cpp

/// Minimal contract for any type that scale() can operate on: must support negation, conversion to/from a raw 64-bit integer, and construction from one.
/// Currently only fundos::currency conforms to this; the concept exists so scale() isn't hardcoded to one type,
/// in case a future scalar (quantity, weight, etc.) needs the same percentage-scaling logic.
template<typename Scalar>
concept SignedScalar = requires(Scalar value, int64_t factor) {
	{ value * -1 } -> std::convertible_to<Scalar>; // no mixing signed and unsigned math
	{ static_cast<int64_t>(value) };               // allow working with a raw int type
	{ Scalar{factor} };                            // allow initialization from a raw int type
};

/// Represents a percentage as basis points (1 basis point = 0.01%).
/// Values are not inherently limited to [0, 1]; string conversions and scale() assume that range.
struct percentage {
	/// 1 basis point = 0.01%
	int32_t basis_points = 0;

	/// Returns 100% as a percentage (10000 basis points).
	static constexpr percentage whole() { return percentage{10000}; }

	static std::optional<percentage> from_string(const std::string& text) {
		auto parsed = parse_percentage(text);
		if (!parsed) { return std::nullopt; }
		return percentage{*parsed};
	}
	std::string to_string(const percentage_locale::spec& locale) const { return format_percentage(basis_points, locale); }

	// Convert operator allows this to be used like a true primitive type
	constexpr explicit operator int64_t() const { return basis_points; }
	constexpr percentage operator-() const { return { -basis_points }; }

	constexpr percentage operator+(const percentage& rhs) const { return { basis_points + rhs.basis_points }; }
	constexpr percentage operator-(const percentage& rhs) const { return { basis_points - rhs.basis_points }; }
	// Operator * % and / are ambiguous for percentage: should they return the rhs type or a percentage?

	constexpr percentage& operator+=(const percentage& rhs) { basis_points += rhs.basis_points; return *this; }
	constexpr percentage& operator-=(const percentage& rhs) { basis_points -= rhs.basis_points; return *this; }
	// Therefore consumers must rely on the more explicit scale() or extracting/mutating basis_points directly

	/// Multiplies value by this percentage without overflow, provided the percentage is within [0, 1].
	/// @note Accepts by const& as S is not guaranteed by SignedScalar to be trivially copyable.
	/// @param value The value to scale; asserts this percentage is within [0, 1] to prevent integer overflow.
	/// @return value scaled by this percentage.
	template<SignedScalar S>
	constexpr S scale(const S& value) const {
		// Split on percentage::whole() keeps high * basis_points within int64_t for ratio <= percentage::whole()
		constexpr int64_t split = whole().basis_points;

		// Intentionally left as a bare assert rather than FUNDOS_REQUIRE_OR_FALLBACK: there is no value of S
		// that would represent "this computation didn't happen" without being arbitrary and potentially as
		// misleading as the overflow it would be guarding against. A violated precondition here is a logic
		// error in the caller, not a recoverable runtime condition.
		//
		// In practice this assert never fires because parse_percentage's best-effort design treats '-' as
		// an unrecognized character and strips it (see its comment) — negative percentages are unrepresentable via
		// the parser, by design, not merely incidentally absent. Still, this is a property of one call site, not
		// something scale() itself enforces: a percentage built another way (basis_points is a public field) or
		// a future parser change could violate this silently in release builds.
		FUNDOS_ASSERT(basis_points >= 0 && basis_points <= split, "scale() risks overflow for percentages outside [0, 1]");

		const int64_t raw = static_cast<int64_t>(value);
		int64_t high = raw / split;
		int64_t low  = raw % split;
		return { high * basis_points + (low * basis_points / split) };
	}

	constexpr bool operator==(const percentage& rhs) const { return basis_points == rhs.basis_points; }
	constexpr bool operator!=(const percentage& rhs) const { return basis_points != rhs.basis_points; }
	constexpr bool operator< (const percentage& rhs) const { return basis_points <  rhs.basis_points; }
	constexpr bool operator> (const percentage& rhs) const { return basis_points >  rhs.basis_points; }
	constexpr bool operator<=(const percentage& rhs) const { return basis_points <= rhs.basis_points; }
	constexpr bool operator>=(const percentage& rhs) const { return basis_points >= rhs.basis_points; }
};
```

The naive way to scale a currency by a percentage is `value * basis_points / 10000`. That's correct, but it multiplies first, and a large enough currency value times a large enough basis_points can overflow `int64_t` before the division ever gets a chance to bring the result back down to a sane size. `scale()` avoids this by splitting the value into a whole part and a remainder before multiplying anything. It divides value by 10000 to get `high`, the part of the value large enough to represent a whole percent, and takes the remainder as `low`. `high` gets multiplied by `basis_points` directly, since a value already divided down is far less likely to overflow. `low`, being strictly smaller than 10000, is multiplied by `basis_points` and divided by 10000 in the same expression, keeping the intermediate product small. Add the two back together and you get the same answer the naive version would give, just without ever letting the multiplication get large enough to overflow along the way.

This is also why `scale()` is a template constrained by `SignedScalar` rather than a function hardcoded to `currency`. Right now `currency` is the only type that needs percentage scaling, but the overflow-avoidance trick has nothing to do with `currency` specifically, it's a property of the math, not the type. Constraining it with a concept means any future scalar type that fits the same shape --- negatable, convertible to and from a raw integer --- gets the same safe scaling for free.

### Datetime and Timedelta

`datetime` and `timedelta` are the two simplest types in the core library, deliberately so. Both wrap a single int64_t of milliseconds, one counting milliseconds since the epoch, the other counting a duration, and neither one knows what a calendar is.

```cpp
struct timedelta {
	int64_t milliseconds = 0;

	static constexpr timedelta seconds(int64_t seconds) { return { seconds * 1000 }; }
	static constexpr timedelta minutes(int64_t minutes) { return seconds(minutes * 60); }
	static constexpr timedelta hours  (int64_t hours)   { return minutes(hours * 60); }
	static constexpr timedelta days   (int64_t days)    { return hours(days * 24); }

	// Convert operator allows this to be used like a true primitive type
	constexpr explicit operator int64_t() const { return milliseconds; }

	/// Returns the absolute value as a timedelta
	/// (kept as timedelta rather than a raw int64_t so it can be compared directly against things like timedelta::days(7)).
	/// Negating INT64_MIN is signed overflow (UB) same as currency.cpp's std::abs(INT64_MIN) case;
	/// accepted here for the same reason: a timedelta that extreme is not a value this library expects to see.
	constexpr timedelta magnitude() const {
		if (milliseconds < 0) {
			return { -milliseconds };
		}
		return { milliseconds };
	}

	constexpr bool operator==(const timedelta& rhs) const { return milliseconds == rhs.milliseconds; }
	constexpr bool operator!=(const timedelta& rhs) const { return milliseconds != rhs.milliseconds; }
	constexpr bool operator< (const timedelta& rhs) const { return milliseconds <  rhs.milliseconds; }
	constexpr bool operator> (const timedelta& rhs) const { return milliseconds >  rhs.milliseconds; }
	constexpr bool operator<=(const timedelta& rhs) const { return milliseconds <= rhs.milliseconds; }
	constexpr bool operator>=(const timedelta& rhs) const { return milliseconds >= rhs.milliseconds; }
};

struct datetime {
	int64_t milliseconds_since_epoch = 0;

	// Convert operator allows this to be used like a true primitive type
	constexpr explicit operator int64_t() const { return milliseconds_since_epoch; }

	constexpr datetime operator+(const timedelta& delta) const { return { milliseconds_since_epoch + delta.milliseconds }; }
	constexpr datetime operator-(const timedelta& delta) const { return { milliseconds_since_epoch - delta.milliseconds }; }
	constexpr timedelta operator-(const datetime& rhs) const { return { milliseconds_since_epoch - rhs.milliseconds_since_epoch }; }

	constexpr bool operator==(const datetime& rhs) const { return milliseconds_since_epoch == rhs.milliseconds_since_epoch; }
	constexpr bool operator!=(const datetime& rhs) const { return milliseconds_since_epoch != rhs.milliseconds_since_epoch; }
	constexpr bool operator< (const datetime& rhs) const { return milliseconds_since_epoch <  rhs.milliseconds_since_epoch; }
	constexpr bool operator> (const datetime& rhs) const { return milliseconds_since_epoch >  rhs.milliseconds_since_epoch; }
	constexpr bool operator<=(const datetime& rhs) const { return milliseconds_since_epoch <= rhs.milliseconds_since_epoch; }
	constexpr bool operator>=(const datetime& rhs) const { return milliseconds_since_epoch >= rhs.milliseconds_since_epoch; }
};
```

This is a deliberate gap, not an oversight. Every GUI framework FundOS targets already has a mature datetime type of its own, Qt's `QDateTime` being the immediate example, and that type already knows about leap years, calendar formatting, and localized date display. Reimplementing any of that in the core library would mean building a calendar from scratch just to throw it away the moment the value crosses into the GUI layer. So `datetime` and `timedelta` stay interchange types: the core library converts a `QDateTime` to milliseconds on the way in, does its arithmetic and comparisons on plain integers, and converts back on the way out. Neither type even gets a `to_string()` or `from_string()`, unlike `currency` and `percentage`. Those two have no external library to defer to for formatting, so they own it themselves. `datetime` doesn't need to, because something downstream always already does.

The reason `timedelta` exists at all, rather than just subtracting two `datetime` values and working with a raw `int64_t`, came out of OFX import. Banks don't always report a transaction on the same date FundOS's own records show, a purchase might clear a day or two later than it was recorded, so matching an imported transaction against an existing one can't rely on an exact date match. It needs a fuzzy one.

```cpp
auto view = account.valid_candidates_for(txn);
for (auto* candidate : view) {
	if ((candidate->date_recorded - *txn.record.date_cleared).magnitude() < timedelta::days(7)) {
		match(candidate);
		break;
	}
}
```

`valid_candidates_for` has already narrowed the list down to candidates with a matching amount. Subtracting two `datetime` values returns a `timedelta`, and `magnitude()` gives the absolute distance between them regardless of which one came first. Comparing that against `timedelta::days(7)` reads exactly like the rule it's expressing: close enough in time to plausibly be the same transaction. Without `timedelta`, this would have been raw millisecond arithmetic with a magic number buried in it, correct, but not something you could read back a week later and immediately trust.

This isn't a considered matching heuristic so much as a "good enough." Amount and a loose date window can still tie: a week of identical charges at the same coffee shop, some imported, some not yet cleared, has no real signal to distinguish which import corresponds to which record. That ambiguity is exactly why matches surfaced this way still go through user confirmation downstream rather than being applied silently.

## Handling Errors in my Core Library

In FundOS, I store data in an SQLite database. The bulk of the code in my core library is interfacing with SQLite and preempting or handling errors that it could produce. To implement this I started with a simple enum return type

```cpp
enum class error : uint8_t {
	none,            // happy path
	not_ready,       // called a query function before migration or after a closed connection
	corrupted,       // unrecoverable fs error, connection closed
	unavailable,     // busy or locked, try again
	inaccessible,    // can't open the database
	readonly,        // filesystem permission check necessary
	out_of_memory,   // potentially transient error
	disk_full,       // potentially transient error
	constraint,      // either a FOREIGN KEY or UNIQUE violation
	not_found,       // query did not yield resulting data
	bad_request,     // incorrect API usage
	rejected,        // data does not satisfy preconditions
	interrupted,     // sqlite3_interrupt was called, most likely from fundos::db::interrupt
	internal,        // an unexpected situation that would abort a debug build
};
```

As the code developed, however, I found myself wanting to attach human readable messages to errors so I wouldn't have to break out a debugger to figure out what caused the error. SQLite already provides a message in most failure cases, and I wanted the ability to add my own for the rest, without paying for a heap allocation every time an error carries nothing more than a static description. This led me to `message`, a `std::variant<const char*, std::shared_ptr<std::string>>` where a string literal costs nothing to store and a `shared_ptr` to an existing SQLite error string costs exactly one allocation, shared rather than copied. Wrapping that in `outcome` gave me a type that carried both an error code and a message together:

```cpp
/// Carries a human-readable error description without unnecessary allocation.
/// Constructed from either:
/// - a string literal (no heap allocation)
/// - a shared sqlite3 error string (shared ownership, one allocation)
/// @warning do not construct from a raw pointer unless it is a string literal
struct message {
	std::variant<const char*, std::shared_ptr<std::string>> content;

	/// implicit construction from string literals — no allocation
	message(const char* literal) : content(literal) {}

	/// from existing shared sqlite error string
	message(std::shared_ptr<std::string> dynamic) : content(std::move(dynamic)) {}

	std::string_view view() const {
		return std::visit([](auto& value) -> std::string_view {
			using ValueType = std::decay_t<decltype(value)>;
			if constexpr (std::is_same_v<ValueType, const char*>) {
				return value ? std::string_view(value) : std::string_view{};
			} else {
				return value ? std::string_view(*value) : std::string_view{};
			}
		}, content);
	}
};

/// Carries an error code and an optional human-readable description.
/// Used as the error carrier in result<T> and the return type for operations that produce no value on success.
/// Defaults to error::none; boolean conversion returns true on success.
struct outcome {
	error code = error::none;
	std::optional<message> msg;
	operator bool() const { return code == error::none; }

	outcome() = default;
	outcome(error e, std::optional<message> m = std::nullopt) : code{e}, msg(std::move(m)) {}
};
```

Some of my database functions return either a value or a potential error. C++23 introduced `std::expected` to handle exactly this, and I think it's a great addition to the language, but I wasn't confident it would be safe to rely on by the time I got to packaging an Android build. NDK toolchains lag behind desktop compilers, and I didn't want the core library's language standard to become the reason Android support stalled out. So I wrote a C++20 equivalent instead.

```cpp
/// @brief Lightweight C++20 alternative to std::expected for database operations.
///
/// Represents either a successful result containing a value of type T, or a failure represented by db::outcome.
///
/// This is a strict discriminated union where exactly one of (T, outcome) is active at any time.
/// No empty or double-value states exist.
///
/// @warning This type does not enforce safe access at runtime.
/// Calling value() or status() in the wrong state triggers std::bad_variant_access.
/// The caller is responsible for checking the state before access.
///
/// Success is determined via the boolean conversion operator; if(result) indicates that a value is present.
template<typename T>
struct result {
	std::variant<T, outcome> data;

	result() = delete;
	result(T v) : data(std::move(v)) {}
	result(outcome o) : data(std::move(o)) {}

	explicit operator bool() const { return std::holds_alternative<T>(data); }

	// All accessors are lvalue-qualified; none make sense on a temporary result.
	// Callers must verify via operator bool before calling value() or status().

	      T&       value()        & { return std::get<T>(data); }
	const T&       value()  const & { return std::get<T>(data); }
	const outcome& status() const & { return std::get<outcome>(data); }
};
```

`result<T>` is the actual `std::expected` equivalent, and it follows the same idea: a `std::variant<T, outcome>` where exactly one side is active at a time. There's no empty state and no way to hold both a value and an error simultaneously, the type itself rules that out. The boolean conversion operator is the only thing a caller needs to check, `if (result)` means a value is present, and calling `value()` without checking first throws `std::bad_variant_access` rather than silently returning garbage.

Most call sites never touch `outcome` directly. Parsing an error into something a user can read is `MainWindow`'s job; everywhere else just checks for a value and moves on:

```cpp
void AccountPage::on_history(fundos::db::result<fundos::db::transaction_history> received) {
	clear_history();
	if (!received) {
		auto* info_row = new QWidget(history_panel);
		history_layout->addWidget(info_row);

		auto* info_layout = new QHBoxLayout(info_row);
		info_layout->addWidget(theme::header_label(tr("Error getting transaction history."), info_row), 0, Qt::AlignCenter);
		return;
	}
	auto& history = received.value();
```

The check-then-use shape repeats everywhere `result<T>` shows up, and that repetition is the point, but I won't overstate what it buys me. Skipping the check and calling `value()` on an error state throws `std::bad_variant_access`, so if that code path executes, the mistake surfaces. But if it doesn't execute, nothing catches it. A caller who never exercises the failure case in testing can ship code that calls `value()` unchecked and never know it's wrong until it actually fails in production. The type doesn't guarantee correctness, it just makes the correct pattern easy to repeat and the incorrect one occasionally loud.

### My Result Type Made my Code Better

There's a smaller nicety here too: a function returning `result<T>` can return either a `T` or an `outcome` directly, and the constructor overloads pick the right variant alternative automatically. No wrapping, no explicit tagging, just return whichever one you have:

```cpp
db::result<currency_locale::selection> db::get_currency_locale() {
	static const std::unordered_map<std::string, int16_t> valid_scales = {
		{ "1", int16_t{1} }, { "10", int16_t{10} }, { "100", int16_t{100} }, { "1000", int16_t{1000} },
	};

	auto preset_result = get_meta(locale_meta.currency_locale_key);
	if (!preset_result) {
		auto& status = preset_result.status();
		if (status.code == error::not_found) {
			return outcome(error::not_found, "Currency Locale is not yet set");
		}
		return status;
	}

	std::string &preset_value = preset_result.value();
	if (preset_value == currency_locale::selection::custom_id) {
		// Extract scale from meta
		auto scale_result = get_meta(locale_meta.currency_locale_scale_key);
		if (!scale_result) { return scale_result.status(); }
		auto scale_it = valid_scales.find(scale_result.value());
		if (scale_it == valid_scales.end()) { return outcome(error::rejected, "Recorded currency scale is invalid"); }
		int16_t scale = scale_it->second;

		// Extract symbol from meta
		auto symbol_result = get_meta(locale_meta.currency_locale_symbol_key);
		if (!symbol_result) { return symbol_result.status(); }
		std::string& symbol = symbol_result.value();
		if (symbol.length() > 4) { return outcome(error::rejected, "Recorded currency symbol is too long"); } // This is an implicit assumption for currency::to_string

		// Extract thousands_separator from meta
		auto thousands_result = get_meta(locale_meta.currency_locale_thousands_separator_key);
		if (!thousands_result) { return thousands_result.status(); }
		if (thousands_result.value().empty()) { return outcome(error::rejected, "Recorded thousands separator is empty"); }
		char thousands_separator = thousands_result.value()[0];

		// Extract decimal_separator from meta
		auto decimal_result = get_meta(locale_meta.currency_locale_decimal_separator_key);
		if (!decimal_result) { return decimal_result.status(); }
		if (decimal_result.value().empty()) { return outcome(error::rejected, "Recorded decimal separator is empty"); }
		char decimal_separator = decimal_result.value()[0];

		// Extract symbol position from meta
		auto position_result = get_meta(locale_meta.currency_locale_symbol_position_key);
		if (!position_result) { return position_result.status(); }
		auto position_enum = string_to_enum(currency_symbol_placement_map, position_result.value());
		if (!position_enum.has_value()) { return outcome(error::rejected, "Recorded symbol position is not a recognized string"); }
		currency_locale::spec::symbol_placement symbol_position = position_enum.value();

		// Extract negative format from meta
		auto negative_result = get_meta(locale_meta.currency_locale_negative_format_key);
		if (!negative_result) { return negative_result.status(); }
		auto negative_enum = string_to_enum(currency_negative_notation_map, negative_result.value());
		if (!negative_enum.has_value()) { return outcome(error::rejected, "Recorded negative format is not a recognized string"); }
		currency_locale::spec::negative_notation negative_format = negative_enum.value();

		return currency_locale::selection(currency_locale::spec{
			.scale = scale,
			.symbol = symbol,
			.thousands_separator = thousands_separator,
			.decimal_separator = decimal_separator,
			.symbol_position = symbol_position,
			.negative_format = negative_format,
		});
	}
	auto locale = currency_locale::get_locale(preset_value);
	if (locale) {
		return currency_locale::selection(locale);
	}
	return outcome(error::rejected, "Recorded currency locale preset is not a recognized string");
}
```

This function used to be the ugliest one in the codebase before I moved it over to `result<T>`, chaining through six or seven meta lookups, any one of which could fail for a different reason. Every early exit is just whichever thing was on hand at the time, a status passed straight through, a fresh outcome built with a specific message, or the actual value once every lookup succeeds, and the return type sorts out which is which without needing to construct an aggregate initializer.

## Moving to the GUI

Once the core library was in good enough shape to build against, I started on the Qt desktop client, and the first thing I reached for, instinctively, was a loading spinner. Something in a database-backed app just feels incomplete without one. But building it made me realize the spinner was pointless as things stood: every database call was running directly on the main thread, which also drives Qt's event loop. A spinner needs that event loop to keep running so it can actually paint, and a synchronous query blocks the loop entirely. The spinner would never render, it would just get torn down the instant the query returned.

That pushed the database onto its own thread, communicating with the GUI purely through signals and slots. Once the database was async, it made sense to give the user a way to cancel a request that was taking too long, and SQLite already has an interrupt mechanism built in for exactly that. So I wired one up.

Then I ran into a problem I hadn't expected: I could barely trigger it. My queries were finishing before I could click cancel. The core library was fast enough that the interrupt I'd built almost never had anything to interrupt.

The actual slow part turned out to be the GUI itself, specifically widget construction. Loading a long transaction history means creating a widget per row, and that has to happen on the main thread no matter how fast the database behind it is. The real fix would be virtualizing the list, only constructing widgets for rows currently visible, but that's a significant rework of how the transaction views are built. I decided it wasn't worth it yet. The delay only shows up if someone selects a long timescale or imports a large OFX file, and even then it's a few seconds, not a freeze.

I say all this a little sheepishly, because I don't particularly enjoy GUI work. The core library took about two months to build and the GUI took about one, so I wouldn't call it the smaller half of the project by any stretch. It was, however, the more enjoyable half by a wide margin.

## A Note on Translations

Locale awareness runs deeper in FundOS than just currency and percentage formatting. Every user-facing string in the Qt client goes through `tr()`, and I used numbered arguments, `%1`, `%2`, rather than string concatenation, specifically so a translator can reorder them to fit their language's grammar rather than being stuck with English word order. I don't want FundOS's usability to stop at whichever country I happen to live in.

I'm not a translator myself, so the strings exist in English and nowhere else yet. But the scaffolding is there for anyone who wants to contribute a translation, and it was worth doing from the start rather than retrofitting later. Retrofitting translator-friendly strings into a codebase that wasn't built for them usually means finding every place someone concatenated a sentence out of fragments and untangling it, which is a much worse task than just doing it right the first time.

## What's Next

Android is still on the roadmap, Kotlin over JNI calling into the same core library that already runs the desktop client. But I'm taking a break from FundOS before I get there.

The portfolio piece is done. The C++ core library is what I set out to build, and it's what I want in front of a hiring manager, not another JavaScript project, not a mobile app built through a JNI bridge. Job hunting is the more serious task in front of me now, and an Android client, however useful, doesn't add much to that particular case. It would say something else instead: that I can pick up a new language and a new platform and still ship something coherent. That's a real skill, and there's a version of this where I make that argument. But it's not the argument I need to make right now.

FundOS will still be here when I'm ready to come back to it. For now, it's done the job I built it to do.
