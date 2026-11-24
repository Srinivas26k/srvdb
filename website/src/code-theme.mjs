// Expressive Code (code block) options, passed to Starlight in astro.config.mjs.
// Monochrome syntax themes: greys and weight for structure, one muted
// colour for string literals. scripts/contrast.mjs checks these colours
// against the code backgrounds, so keep the two in sync.

/** Build a monochrome TextMate theme from a small palette. */
function mono(name, type, c) {
	return {
		name,
		type,
		colors: { 'editor.background': c.bg, 'editor.foreground': c.fg },
		tokenColors: [
			{ settings: { foreground: c.fg } },
			{ scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: c.comment, fontStyle: 'italic' } },
			{ scope: ['keyword', 'storage', 'storage.type', 'storage.modifier', 'keyword.control', 'keyword.operator.word'], settings: { foreground: c.keyword, fontStyle: 'bold' } },
			{ scope: ['string', 'string.quoted', 'punctuation.definition.string', 'constant.character.escape'], settings: { foreground: c.string } },
			{ scope: ['constant.numeric', 'constant.language', 'support.constant', 'variable.other.constant'], settings: { foreground: c.muted } },
			{ scope: ['entity.name.function', 'support.function', 'meta.function-call', 'entity.name.type', 'support.type', 'entity.name.class'], settings: { foreground: c.fg } },
			{ scope: ['punctuation', 'keyword.operator', 'meta.brace'], settings: { foreground: c.muted } },
			{ scope: ['variable.parameter', 'variable.language'], settings: { foreground: c.fg, fontStyle: 'italic' } },
		],
	};
}

export const codeColors = {
	light: { bg: '#fafafa', fg: '#111111', comment: '#6b6b6b', keyword: '#111111', string: '#2f5d46', muted: '#4a4a4a' },
	dark: { bg: '#171717', fg: '#ededed', comment: '#a3a3a3', keyword: '#ffffff', string: '#9ccfb0', muted: '#c4c4c4' },
};

/** Options for Starlight's `expressiveCode` setting (plain, serializable). */
export const expressiveCode = {
	themes: [mono('srvdb-dark', 'dark', codeColors.dark), mono('srvdb-light', 'light', codeColors.light)],
	styleOverrides: {
		borderRadius: '8px',
		frames: {
			shadowColor: 'transparent',
			terminalTitlebarDotsOpacity: '0',
		},
	},
};
