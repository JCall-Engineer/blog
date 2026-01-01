const FactorialCache = (() => {
	const cache = [1n];
	return {
		get: (n) => {
			while (cache.length <= n) {
				cache.push(cache[cache.length - 1] * BigInt(cache.length));
			}
			return cache[n];
		}
	};
})();

function bigIntToFloat(x) {
	if (x === 0n) return 0;

	let neg = x < 0n;
	if (neg) x = -x;

	// bit length
	let bits = x.toString(2).length;

	// extract the top 53 bits (mantissa width)
	let shift = bits - 53;
	let mantissa = Number(x >> BigInt(shift));

	// assemble float
	let result = mantissa * 2 ** shift;

	return neg ? -result : result;
}

function gcf(a, b) {
	a = a < 0n ? -a : a;
	b = b < 0n ? -b : b;
	while (b !== 0n) {
		[a, b] = [b, a % b];
	}
	return a;
}

class Fraction {
	constructor(numerator, denominator = 1n) {
		this.numerator = BigInt(numerator);
		this.denominator = BigInt(denominator);
		this._reduce();
		this.float = bigIntToFloat(this.numerator) / bigIntToFloat(this.denominator);
		this.string = this.denominator === 1n ? `${this.numerator}` : `${this.numerator}/${this.denominator}`;
	}

	_reduce() {
		if (this.denominator === 0n) throw new Error("Division by zero");
		if (this.denominator < 0n) {
			this.numerator = -this.numerator;
			this.denominator = -this.denominator;
		}
		const gcd = gcf(this.numerator, this.denominator);
		this.numerator /= gcd;
		this.denominator /= gcd;
	}

	add(other) {
		return new Fraction(
			this.numerator * other.denominator + other.numerator * this.denominator,
			this.denominator * other.denominator
		);
	}

	mul(other) {
		return new Fraction(this.numerator * other.numerator, this.denominator * other.denominator);
	}

	pow(exp) {
		if (exp === 0) return new Fraction(1n);
		let result = new Fraction(this.numerator, this.denominator);
		for (let i = 1; i < exp; i++) {
			result = result.mul(this);
		}
		return result;
	}
}

class Node {
	constructor(attackers, defenders) {
		this.attackers = attackers;
		this.defenders = defenders;
	}

	key() {
		return `${this.attackers},${this.defenders}`;
	}

	isValid() {
		return this.attackers >= 0 && this.defenders >= 0 &&
			(this.attackers + this.defenders) > 0;
	}

	hasEdges() {
		return this.attackers > 0 && this.defenders > 0;
	}

	subtract(other) {
		return new Node(
			this.attackers - other.attackers,
			this.defenders - other.defenders
		);
	}
}

const SpaceCache = new Map();
function probability_space(attackers, defenders) {
	attackers = Math.min(3, attackers);
	defenders = Math.min(2, defenders);
	const key = `${attackers},${defenders}`;

	if (SpaceCache.has(key)) return SpaceCache.get(key);

	let W = 0, L = 0, T = 0;
	const dice = attackers + defenders;

	function* rollGenerator(n) {
		if (n === 0) {
			yield [];
			return;
		}
		for (let die = 1; die <= 6; die++) {
			for (const rest of rollGenerator(n - 1)) {
				yield [die, ...rest];
			}
		}
	}

	for (const roll of rollGenerator(dice)) {
		const attRolls = roll.slice(0, attackers).sort((a, b) => b - a);
		const defRolls = roll.slice(attackers).sort((a, b) => b - a);

		let attLosses = 0, defLosses = 0;
		const compare = Math.min(attackers, defenders);

		for (let i = 0; i < compare; i++) {
			if (attRolls[i] > defRolls[i]) defLosses++;
			else attLosses++;
		}

		if (defLosses > attLosses) W++;
		else if (attLosses > defLosses) L++;
		else T++;
	}

	const N = 6 ** dice;
	const space = {
		attackers,
		defenders,
		W, T, L, N,
		P_W: new Fraction(W, N),
		P_T: new Fraction(T, N),
		P_L: new Fraction(L, N)
	};

	SpaceCache.set(key, space);
	return space;
}

// Precompute spaces
for (let a = 1; a <= 3; a++) {
	for (let d = 1; d <= 2; d++) {
		probability_space(a, d);
	}
}
function constant_space_probability(start, end) {
	const space = probability_space(3, 2);
	const delta = start.subtract(end);

	if (delta.attackers < 0 || delta.defenders < 0) return new Fraction(0n);
	if (start.attackers >= 2 && start.defenders >= 2) {
		if ((delta.attackers + delta.defenders) % 2 !== 0) {
			return new Fraction(0n);
		}
	}

	const W_max = Math.floor(delta.defenders / 2);
	const L_max = Math.floor(delta.attackers / 2);
	const T_min = delta.attackers % 2;
	const T_max = 2 * Math.min(L_max, W_max) + T_min;

	let totalProb = new Fraction(0n);

	for (let T_edges = T_min; T_edges <= T_max; T_edges += 2) {
		const W_edges = W_max - Math.floor((T_edges - T_min) / 2);
		const L_edges = L_max - Math.floor((T_edges - T_min) / 2);
		const totalEdges = W_edges + L_edges + T_edges;

		const multinomial = FactorialCache.get(totalEdges) /
			(FactorialCache.get(W_edges) * FactorialCache.get(L_edges) *
			FactorialCache.get(T_edges));

		const pathProb = space.P_W.pow(W_edges)
			.mul(space.P_L.pow(L_edges))
			.mul(space.P_T.pow(T_edges));

		totalProb = totalProb.add(
			new Fraction(multinomial).mul(pathProb)
		);
	}

	return totalProb;
}
function computeProbability(start, end) {
	if (!start.isValid() || !start.hasEdges() || !end.isValid()) {
		return new Fraction(0n);
	}
	if (start.attackers < end.attackers || start.defenders < end.defenders) {
		return new Fraction(0n);
	}
	if (start.attackers === end.attackers && start.defenders === end.defenders) {
		return new Fraction(1n);
	}

	// Try fast path for 3v2 space
	if (end.hasEdges() && start.attackers >= 3 && start.defenders >= 2 &&
		end.attackers >= 3 && end.defenders >= 2) {
		return constant_space_probability(start, end);
	}

	// Dynamic programming fallback
	const reachProb = new Map();

	// Compute boundaries
	const boundaries = [];
	if (start.attackers >= 3 && start.defenders >= 2) {
		boundaries.push(new Node(3, 2));
		for (let d = 3; d <= start.defenders; d++) {
			boundaries.push(new Node(3, d));
		}
		for (let a = 4; a <= start.attackers; a++) {
			boundaries.push(new Node(a, 2));
		}
		if (start.attackers >= 4 && start.defenders >= 3) {
			boundaries.push(new Node(4, 3));
			for (let d = 4; d <= start.defenders; d++) {
				boundaries.push(new Node(4, d));
			}
			for (let a = 5; a <= start.attackers; a++) {
				boundaries.push(new Node(a, 3));
			}
		}
	}

	for (const boundary of boundaries) {
		if (boundary.attackers <= start.attackers &&
			boundary.defenders <= start.defenders) {
			reachProb.set(boundary.key(), constant_space_probability(start, boundary));
		}
	}

	if (boundaries.length === 0) {
		reachProb.set(start.key(), new Fraction(1n));
	}

	// Traverse in topological order
	for (let total = start.attackers + start.defenders;
		total >= end.attackers + end.defenders; total--) {
		for (let a = end.attackers; a <= start.attackers; a++) {
			const d = total - a;
			if (d < end.defenders || d > start.defenders) continue;

			const node = new Node(a, d);
			const prob = reachProb.get(node.key());
			if (!prob || prob.numerator === 0n) continue;
			if (!node.hasEdges()) continue;

			const space = probability_space(
				Math.min(3, node.attackers),
				Math.min(2, node.defenders)
			);

			const outcomes = [];
			const compare = Math.min(2, node.attackers, node.defenders);

			if (compare === 2) {
				outcomes.push([new Node(node.attackers, node.defenders - 2), space.P_W]);
				if (space.T > 0) {
					outcomes.push([new Node(node.attackers - 1, node.defenders - 1), space.P_T]);
				}
				outcomes.push([new Node(node.attackers - 2, node.defenders), space.P_L]);
			} else if (compare === 1) {
				outcomes.push([new Node(node.attackers, node.defenders - 1), space.P_W]);
				outcomes.push([new Node(node.attackers - 1, node.defenders), space.P_L]);
			}

			for (const [outcome, edgeProb] of outcomes) {
				if (!outcome.isValid()) continue;
				const isBoundary = boundaries.some(b =>
					b.attackers === outcome.attackers && b.defenders === outcome.defenders
				);
				if (!isBoundary) {
					const key = outcome.key();
					const current = reachProb.get(key) || new Fraction(0n);
					reachProb.set(key, current.add(prob.mul(edgeProb)));
				}
			}
		}
	}

	return reachProb.get(end.key()) || new Fraction(0n);
}

computeDistributionCache = new Map();
function computeDistribution(attackers, defenders) {
	const key = `${attackers},${defenders}`;
	if (computeDistributionCache.has(key)) {
		return computeDistributionCache.get(key);
	}

	const start = new Node(attackers, defenders);
	const distribution = Array(attackers + 1).fill(new Fraction(0n));

	for (let attackersLeft = 1; attackersLeft <= attackers; attackersLeft++) {
		const attackersLost = attackers - attackersLeft;
		distribution[attackersLost] = computeProbability(start, new Node(attackersLeft, 0));
	}

	for (let defendersLeft = 1; defendersLeft <= defenders; defendersLeft++) {
		distribution[attackers] = distribution[attackers].add(
			computeProbability(start, new Node(0, defendersLeft))
		);
	}

	computeDistributionCache.set(key, distribution);
	return distribution;
}

const Frozen = Object.freeze;
const NullProto = (obj) => Object.assign(Object.create(null), obj || {});
const FrozenNullProto = (obj) => Frozen(NullProto(obj));
const funcs = FrozenNullProto({
	computeProbability: (startA, startD, endA, endD) => {
		const start = new Node(startA, startD);
		const end = new Node(endA, endD);
		return computeProbability(start, end);
	},
	computeDistribution,
});

self.addEventListener('message', (e) => {
	const { func, args, id } = e.data;

	if (!(func in funcs)) {
		self.postMessage({ id, error: `Unrecognized function: "${func}"` });
		return;
	}

	try {
		const result = funcs[func]?.(...args);
		self.postMessage({ id, result });
	} catch (error) {
		self.postMessage({ id, error: error.message });
	}
});
