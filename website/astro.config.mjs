// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import industries from './src/industries-sidebar.mjs';
import { expressiveCode } from './src/code-theme.mjs';

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
				{ tag: 'script', attrs: { type: 'application/ld+json' }, content: JSON.stringify({"@context": "https://schema.org", "@type": "SoftwareApplication", "name": "SrvDB", "alternateName": ["srvdb", "SrvDB vector database"], "description": "SrvDB is an open-source embedded vector database for Python and Rust. It stores embeddings on local disk for semantic search, RAG and AI memory, with no server and a crash-safe save(). Not related to SurrealDB.", "applicationCategory": "DeveloperApplication", "operatingSystem": "Linux, macOS, Windows", "url": "https://srinivas26k.github.io/srvdb/", "downloadUrl": "https://pypi.org/project/srvdb/", "license": "https://www.apache.org/licenses/LICENSE-2.0", "programmingLanguage": ["Rust", "Python"], "keywords": "embedded vector database, vector search, semantic search, RAG, local AI, offline AI, embeddings, HNSW, Python, Rust", "author": {"@type": "Person", "name": "Srinivas Nampalli"}, "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}, "sameAs": ["https://github.com/Srinivas26k/srvdb", "https://pypi.org/project/srvdb/", "https://crates.io/crates/srvdb"]}) },
			],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/Srinivas26k/srvdb' },
			],
			editLink: { baseUrl: 'https://github.com/Srinivas26k/srvdb/edit/release/0.4/website/' },
			customCss: [
				'@fontsource/inter/400.css',
				'@fontsource/inter/600.css',
				'@fontsource/jetbrains-mono/400.css',
				'./src/styles/tokens.css',
				'./src/styles/layout.css',
				'./src/styles/diagrams.css',
			],
			components: {
				Header: './src/components/overrides/Header.astro',
				Sidebar: './src/components/overrides/Sidebar.astro',
				PageTitle: './src/components/overrides/PageTitle.astro',
				Footer: './src/components/overrides/Footer.astro',
			},
			expressiveCode,
			lastUpdated: true,
			sidebar: [
				{
					label: 'Get started',
					items: [
						{ label: 'Introduction', slug: 'start/introduction' },
						{ label: 'Installation', slug: 'start/installation' },
						{ label: 'Quickstart', slug: 'start/quickstart' },
						{ label: 'Use cases', slug: 'start/use-cases' },
					],
				},
				{
					label: 'Concepts',
					items: [
						{ label: 'Compare index modes', slug: 'concepts/comparison' },
						{ label: 'Flat (exact) search', slug: 'concepts/flat' },
						{ label: 'HNSW graph search', slug: 'concepts/hnsw' },
						{ label: 'IVF partitions', slug: 'concepts/ivf' },
						{ label: 'SQ8 scalar quantization', slug: 'concepts/sq8' },
						{ label: 'PQ product quantization', slug: 'concepts/pq' },
						{ label: 'RaBitQ, fast-scan, rerank', slug: 'concepts/rabitq' },
						{ label: 'Memory budget', slug: 'concepts/memory-budget' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Build a local RAG app', slug: 'guides/local-rag' },
						{ label: 'Filtered search', slug: 'guides/filtered-search' },
						{ label: 'Links between records', slug: 'guides/links' },
						{ label: 'Keyword and hybrid search', slug: 'guides/keyword-search' },
						{ label: 'LangChain and LlamaIndex', slug: 'guides/frameworks' },
						{ label: 'Choosing an index mode', slug: 'guides/index-modes' },
						{ label: 'RaBitQ low-bit search', slug: 'guides/rabitq' },
						{ label: 'Fit a memory budget', slug: 'guides/memory-budget' },
						{ label: 'Durability and save()', slug: 'guides/durability' },
						{ label: 'Delete, upsert and compact', slug: 'guides/delete-upsert-compact' },
						{ label: 'Threads and processes', slug: 'guides/concurrency' },
						{ label: 'Handling errors', slug: 'guides/errors' },
						{ label: 'Using SrvDB from Rust', slug: 'guides/rust' },
						{ label: 'Upgrading', slug: 'guides/upgrading' },
						{ label: 'Live examples & cheat sheets', slug: 'examples/overview', badge: { text: 'Soon', variant: 'caution' } },
					],
				},
				{
					label: 'Reference',
					items: [
						{
							label: 'Python API',
							collapsed: true,
							items: [
								{ label: 'Overview', slug: 'reference/python' },
								{ label: 'srvdb.create', slug: 'reference/python/create' },
								{ label: 'srvdb.open', slug: 'reference/python/open' },
								{ label: 'srvdb.delete', slug: 'reference/python/delete' },
								{ label: 'The db object', slug: 'reference/python/db' },
								{ label: 'db.add', slug: 'reference/python/db-add' },
								{ label: 'db.upsert', slug: 'reference/python/db-upsert' },
								{ label: 'db.delete', slug: 'reference/python/db-delete' },
								{ label: 'db.save', slug: 'reference/python/db-save' },
								{ label: 'db.close', slug: 'reference/python/db-close' },
								{ label: 'db.compact', slug: 'reference/python/db-compact' },
								{ label: 'db.search', slug: 'reference/python/db-search' },
								{ label: 'db.enable_keyword_search', slug: 'reference/python/db-enable-keyword-search' },
								{ label: 'db.search_batch', slug: 'reference/python/db-search-batch' },
								{ label: 'db.link', slug: 'reference/python/db-link' },
								{ label: 'db.unlink', slug: 'reference/python/db-unlink' },
								{ label: 'db.neighbors', slug: 'reference/python/db-neighbors' },
								{ label: 'db.get', slug: 'reference/python/db-get' },
								{ label: 'db.get_all_ids', slug: 'reference/python/db-get-all-ids' },
								{ label: 'db.count', slug: 'reference/python/db-count' },
								{ label: 'db.info', slug: 'reference/python/db-info' },
								{ label: 'db.memory_report', slug: 'reference/python/db-memory-report' },
								{ label: 'db.set_mode', slug: 'reference/python/db-set-mode' },
								{ label: 'db.set_ef_search', slug: 'reference/python/db-set-ef-search' },
								{ label: 'db.configure_ivf', slug: 'reference/python/db-configure-ivf' },
								{ label: 'db.train_ivf', slug: 'reference/python/db-train-ivf' },
								{ label: 'db.set_rabitq_rerank', slug: 'reference/python/db-set-rabitq-rerank' },
								{ label: 'LangChain SrvDBVectorStore', slug: 'reference/python/langchain-srvdbvectorstore' },
								{ label: 'LlamaIndex SrvDBVectorStore', slug: 'reference/python/llamaindex-srvdbvectorstore' },
								{ label: 'Recovery: srvdb.rebuild_checksums', slug: 'reference/python/rebuild-checksums' },
								{ label: 'Recovery: srvdb.drop_links', slug: 'reference/python/drop-links' },
							],
						},
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
						{ label: 'Benchmarks', slug: 'benchmarks' },
						{ label: 'Integrations', slug: 'project/integrations', badge: { text: 'Soon', variant: 'caution' } },
						{ label: 'Contributing', slug: 'project/contributing' },
						{ label: 'Community', slug: 'project/community' },
						{ label: 'Sponsor', slug: 'project/sponsor' },
						{ label: 'FAQ', slug: 'project/faq' },
						{ label: 'Brand', slug: 'project/brand' },
					],
				},
				// Industry pages last and collapsed, so the default sidebar reads
				// Get started, Guides, Reference, Project.
				{ label: 'Industries', collapsed: true, items: industries },
			],
		}),
	],
});
