// Helpers for the benchmarks page. Every number shown on the page comes
// from website/src/data/benchmarks/comparison.json through these functions.

export interface Point {
	param: string | null;
	value: number | null;
	recall_at_k: number;
	batch_recall_at_k: number;
	qps: number;
	p50_ms: number;
	p99_ms: number;
	batch_qps: number;
}

export interface Result {
	name: string;
	library?: string;
	index?: string;
	threads: number;
	build_secs?: number;
	baseline_rss_mib?: number;
	peak_rss_mib?: number;
	points?: Point[];
	error?: string;
}

export interface Run {
	dataset: { name: string; dim: number; base: number; queries: number; k: number; seed: number };
	results: Result[];
}

export const label = (r: Result) => `${r.library} ${r.index}`;
export const isSrvdb = (r: Result) => r.library === 'SrvDB';
export const ok = (rs: Result[]) => rs.filter((r) => !r.error && r.points && r.points.length);
export const forThreads = (run: Run, t: number) => ok(run.results).filter((r) => r.threads === t);
export const failures = (run: Run) => run.results.filter((r) => r.error);

export function fmt(x: number, digits = 3): string {
	if (!Number.isFinite(x)) return 'n/a';
	if (Math.abs(x) >= 1000) return Math.round(x).toLocaleString('en-US');
	return Number(x.toPrecision(digits)).toString();
}

/** Fastest single-query point of a result that reaches the recall target. */
export function bestAt(r: Result, target: number): Point | undefined {
	const pts = (r.points ?? []).filter((p) => p.recall_at_k >= target);
	return pts.sort((a, b) => b.qps - a.qps)[0];
}

/** Highest recall point (used when the target is never reached). */
export function maxRecall(r: Result): Point | undefined {
	return [...(r.points ?? [])].sort((a, b) => b.recall_at_k - a.recall_at_k)[0];
}

export function runTitle(run: Run): string {
	return `dim ${run.dataset.dim}, n ${run.dataset.base.toLocaleString('en-US')}`;
}

/** Plain-words comparison of SrvDB against the fastest other library at a recall target. */
export function verdict(run: Run, threads: number, target: number): string[] {
	const rs = forThreads(run, threads);
	const mine = rs.filter(isSrvdb).map((r) => ({ r, p: bestAt(r, target) })).filter((x) => x.p);
	const theirs = rs.filter((r) => !isSrvdb(r)).map((r) => ({ r, p: bestAt(r, target) })).filter((x) => x.p);
	const pct = Math.round(target * 100);
	if (!mine.length) return [`No SrvDB index reached recall@10 of ${pct}% in this run.`];
	const m = mine.sort((a, b) => b.p!.qps - a.p!.qps)[0];
	const out: string[] = [];
	const best = theirs.sort((a, b) => b.p!.qps - a.p!.qps)[0];
	if (best) {
		const ratio = best.p!.qps / m.p!.qps;
		out.push(
			ratio > 1
				? `At recall@10 of at least ${pct}%, SrvDB loses on query speed: ${label(best.r)} answers ${fmt(ratio, 2)}x more queries per second than SrvDB's fastest index (${m.r.index}).`
				: `At recall@10 of at least ${pct}%, SrvDB's ${m.r.index} index is the fastest measured, ${fmt(1 / ratio, 2)}x ahead of ${label(best.r)}.`,
		);
		const slower = theirs.filter((x) => x.p!.qps < m.p!.qps).map((x) => label(x.r));
		if (slower.length && ratio > 1) out.push(`SrvDB ${m.r.index} is faster than: ${slower.join(', ')}.`);
	}
	const builds = rs.filter((r) => r.build_secs !== undefined).sort((a, b) => a.build_secs! - b.build_secs!);
	const rank = builds.findIndex((r) => r === m.r) + 1;
	if (rank) out.push(`Build time for SrvDB ${m.r.index} ranks ${rank} of ${builds.length} (it includes a durable persist() to disk; the others build in memory only).`);
	return out;
}
