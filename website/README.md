# SrvDB website

The documentation site at <https://srinivas26k.github.io/srvdb/>, built with
[Astro Starlight](https://starlight.astro.build/) and the SrvDB eagle theme
(`src/styles/eagle.css`).

```bash
cd website
npm ci
npm run dev        # http://localhost:4321/srvdb/
npm run build      # static site in dist/
```

Pages live in `src/content/docs/` as Markdown; the sidebar is defined in
`astro.config.mjs`. Every code snippet must run against the current release.
Pushing to `master` deploys the site (`.github/workflows/docs.yml`).
