// WCAG 2.1 contrast check for the key UI colour pairs in src/styles/tokens.css
// and every syntax token colour in src/code-theme.mjs against its code block
// background. Run from website/: npm run check:contrast
// (exits 1 if any pair is below 4.5:1). The UI pairs are listed by hand and
// must be updated when tokens.css changes.
import { codeColors } from '../src/code-theme.mjs';

const lum = (hex) => {
	const c = hex.match(/\w\w/g).map((h) => parseInt(h, 16) / 255);
	const l = c.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
	return 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2];
};
const ratio = (a, b) => {
	const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
	return (x + 0.05) / (y + 0.05);
};
const pairs = {
	light: [
		['body text on panel', '#111111', '#ffffff'],
		['muted text on panel', '#6b6b6b', '#ffffff'],
		['muted text on page', '#6b6b6b', '#f5f5f5'],
		['sidebar item on page', '#333333', '#f5f5f5'],
		['active item on pill', '#111111', '#ececec'],
		['muted text on pill', '#6b6b6b', '#ececec'],
		['button text on black', '#ffffff', '#111111'],
		['code text on code block', '#111111', '#fafafa'],
	],
	dark: [
		['body text on panel', '#ededed', '#111111'],
		['muted text on panel', '#a3a3a3', '#111111'],
		['muted text on page', '#a3a3a3', '#0a0a0a'],
		['sidebar item on page', '#d4d4d4', '#0a0a0a'],
		['active item on pill', '#fafafa', '#262626'],
		['muted text on pill', '#a3a3a3', '#262626'],
		['button text on white', '#111111', '#fafafa'],
		['code text on code block', '#ededed', '#171717'],
	],
};
for (const theme of ['light', 'dark']) {
	const { bg, ...tokens } = codeColors[theme];
	for (const [token, fg] of Object.entries(tokens)) {
		pairs[theme].push([`syntax ${token} on code`, fg, bg]);
	}
}
let fail = false;
for (const [theme, list] of Object.entries(pairs)) {
	for (const [name, fg, bg] of list) {
		const r = ratio(fg, bg);
		const ok = r >= 4.5;
		fail ||= !ok;
		console.log(`${theme.padEnd(5)} ${name.padEnd(24)} ${fg} on ${bg}  ${r.toFixed(2)}:1 ${ok ? 'AA' : 'FAIL'}`);
	}
}
process.exit(fail ? 1 : 0);
