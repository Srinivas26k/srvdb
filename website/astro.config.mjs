// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Deployed to GitHub Pages at https://srinivas26k.github.io/srvdb/
export default defineConfig({
	site: 'https://srinivas26k.github.io',
	base: '/srvdb',
	integrations: [
		starlight({
			title: 'SrvDB',
			description:
				'Embedded vector database for local and offline AI. Crash-safe, in-process, Rust core with Python bindings.',
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
				replacesTitle: true,
			},
			favicon: '/favicon.svg',
			head: [
				{ tag: 'link', attrs: { rel: 'icon', href: '/srvdb/favicon.ico', sizes: '48x48' } },
				{ tag: 'link', attrs: { rel: 'apple-touch-icon', href: '/srvdb/apple-touch-icon.png' } },
				{ tag: 'meta', attrs: { property: 'og:image', content: 'https://srinivas26k.github.io/srvdb/social-card.png' } },
				{ tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
				{ tag: 'meta', attrs: { name: 'keywords', content: 'SrvDB, embedded vector database, vector search, semantic search, RAG, local AI, offline AI, Python, Rust' } },
				// Structured data for search engines and AI answer engines
				{ tag: 'script', attrs: { type: 'application/ld+json' }, content: JSON.stringify({"@context": "https://schema.org", "@type": "SoftwareApplication", "name": "SrvDB", "alternateName": ["srvdb", "SrvDB vector database"], "description": "SrvDB is an open-source embedded vector database for Python and Rust. It stores embeddings on local disk for semantic search, RAG and AI memory, with no server and a crash-safe persist(). Not related to SurrealDB.", "applicationCategory": "DeveloperApplication", "operatingSystem": "Linux, macOS, Windows", "url": "https://srinivas26k.github.io/srvdb/", "downloadUrl": "https://pypi.org/project/srvdb/", "license": "https://www.apache.org/licenses/LICENSE-2.0", "programmingLanguage": ["Rust", "Python"], "keywords": "embedded vector database, vector search, semantic search, RAG, local AI, offline AI, embeddings, HNSW, Python, Rust", "author": {"@type": "Person", "name": "Srinivas Nampalli"}, "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}, "sameAs": ["https://github.com/Srinivas26k/srvdb", "https://pypi.org/project/srvdb/", "https://crates.io/crates/srvdb"]}) },
			],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/Srinivas26k/srvdb' },
			],
			editLink: { baseUrl: 'https://github.com/Srinivas26k/srvdb/edit/master/website/' },
			customCss: ['@fontsource/cinzel/600.css', '@fontsource/cinzel/700.css', './src/styles/eagle.css'],
			lastUpdated: true,
			sidebar: [
				{
					label: 'Start here',
					items: [
						{ label: 'Introduction', slug: 'start/introduction' },
						{ label: 'Installation', slug: 'start/installation' },
						{ label: 'Quickstart', slug: 'start/quickstart' },
						{ label: 'Use cases', slug: 'start/use-cases' },
					],
				},
				{
					label: 'Examples',
					items: [
						{ label: 'Live examples & cheat sheets', slug: 'examples/overview', badge: { text: 'Soon', variant: 'caution' } },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Build a local RAG app', slug: 'guides/local-rag' },
						{ label: 'Choosing an index mode', slug: 'guides/index-modes' },
						{ label: 'Durability and persist()', slug: 'guides/durability' },
						{ label: 'Delete, upsert and compact', slug: 'guides/delete-upsert-compact' },
						{ label: 'Threads and processes', slug: 'guides/concurrency' },
						{ label: 'Handling errors', slug: 'guides/errors' },
						{ label: 'Using SrvDB from Rust', slug: 'guides/rust' },
						{ label: 'Upgrading to 0.3', slug: 'guides/upgrading' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Python API', slug: 'reference/python-api' },
						{ label: 'Rust API', slug: 'reference/rust-api' },
						{ label: 'On-disk files', slug: 'reference/files' },
					],
				},
				{
					label: 'Project',
					items: [
						{ label: 'Architecture', slug: 'project/architecture' },
						{ label: 'Design decisions', slug: 'project/decisions' },
						{ label: 'Roadmap', slug: 'project/roadmap' },
						{ label: 'Benchmarks', slug: 'project/benchmarks', badge: { text: 'Soon', variant: 'caution' } },
						{ label: 'Integrations', slug: 'project/integrations', badge: { text: 'Soon', variant: 'caution' } },
						{ label: 'Contributing', slug: 'project/contributing' },
						{ label: 'FAQ', slug: 'project/faq' },
						{ label: 'Brand', slug: 'project/brand' },
					],
				},
			],
		}),
	],
});
