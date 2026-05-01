(function () {
    const LIST_PATH = 'data/trends/trends-list.txt';
    const REPORT_PATH = (name) => `data/trends/${name}`;

    const $ = (id) => document.getElementById(id);

    function fetchText(path) {
        return fetch(DATA_CONFIG.getDataUrl(path), { cache: 'no-store' }).then((r) => {
            if (!r.ok) throw new Error(`${path} → ${r.status}`);
            return r.text();
        });
    }

    function fetchJson(path) {
        return fetch(DATA_CONFIG.getDataUrl(path), { cache: 'no-store' }).then((r) => {
            if (!r.ok) throw new Error(`${path} → ${r.status}`);
            return r.json();
        });
    }

    function clear(node) {
        while (node.firstChild) node.removeChild(node.firstChild);
    }

    function showMessage(node, text, type) {
        clear(node);
        const div = document.createElement('div');
        div.className = type === 'error' ? 'error-state' : 'empty-state';
        div.textContent = text;
        node.appendChild(div);
    }

    function truncate(s, n) {
        return s.length > n ? s.slice(0, n - 1) + '…' : s;
    }

    function paperChip(id, paperIndex) {
        const meta = paperIndex && paperIndex[id];
        const href = (meta && meta.abs) || `https://arxiv.org/abs/${id}`;
        const label = meta && meta.title ? `${id} · ${truncate(meta.title, 64)}` : id;
        const a = document.createElement('a');
        a.href = href;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.className = 'paper-chip';
        a.textContent = label;
        return a;
    }

    function appendSection(card, label, body) {
        if (!body) return;
        const lbl = document.createElement('span');
        lbl.className = 'section-label';
        lbl.textContent = label;
        card.appendChild(lbl);
        const p = document.createElement('p');
        p.textContent = body;
        card.appendChild(p);
    }

    function growthClass(g) {
        if (g >= 1.5) return 'growth-hot';
        if (g >= 1.1) return 'growth-warm';
        return '';
    }

    function confidenceClass(c) {
        const v = (c || '').toLowerCase();
        if (v === 'high') return 'confidence-high';
        if (v === 'medium') return 'confidence-medium';
        return 'confidence-low';
    }

    function renderCluster(c, paperIndex, opts) {
        const card = document.createElement('div');
        card.className = 'cluster-card';

        const head = document.createElement('div');
        head.className = 'cluster-head';
        const label = document.createElement('h3');
        label.className = 'cluster-label';
        label.textContent = c.label || `Cluster ${c.cluster_id}`;
        head.appendChild(label);

        const stats = document.createElement('div');
        stats.className = 'cluster-stats';

        const sizePill = document.createElement('span');
        sizePill.className = 'stat-pill';
        sizePill.textContent = `${c.size} papers`;
        stats.appendChild(sizePill);

        const growthPill = document.createElement('span');
        growthPill.className = 'stat-pill ' + growthClass(c.growth_ratio);
        growthPill.textContent = `growth ×${c.growth_ratio}`;
        stats.appendChild(growthPill);

        if (c.confidence) {
            const conf = document.createElement('span');
            conf.className = 'confidence-tag ' + confidenceClass(c.confidence);
            conf.textContent = c.confidence + ' confidence';
            stats.appendChild(conf);
        }

        head.appendChild(stats);
        card.appendChild(head);

        if (c.one_line) {
            const oneLine = document.createElement('p');
            oneLine.className = 'cluster-oneliner';
            oneLine.textContent = c.one_line;
            card.appendChild(oneLine);
        }

        if (c.keywords && c.keywords.length) {
            const kwWrap = document.createElement('div');
            kwWrap.className = 'keyword-list';
            c.keywords.forEach((k) => {
                const pill = document.createElement('span');
                pill.className = 'keyword-pill';
                pill.textContent = k;
                kwWrap.appendChild(pill);
            });
            card.appendChild(kwWrap);
        }

        if (opts.mode === 'gaps') {
            appendSection(card, 'Existing landscape', c.existing_landscape);
            appendSection(card, 'Research–industry gap', c.research_industry_gap);
            appendSection(card, 'Startup thesis', c.startup_thesis);
            appendSection(card, 'Why now', c.why_now);
            appendSection(card, 'Risks', c.risks);
        } else {
            appendSection(card, 'Research–industry gap', c.research_industry_gap);
            appendSection(card, 'Why now', c.why_now);
        }

        if (c.sample_paper_ids && c.sample_paper_ids.length) {
            const lbl = document.createElement('span');
            lbl.className = 'section-label';
            lbl.textContent = 'Representative papers';
            card.appendChild(lbl);
            const chips = document.createElement('div');
            chips.className = 'paper-chips';
            c.sample_paper_ids.forEach((id) => chips.appendChild(paperChip(id, paperIndex)));
            card.appendChild(chips);
        }

        return card;
    }

    function renderClusters(clusters, paperIndex, mount, mode) {
        clear(mount);
        if (!clusters || clusters.length === 0) {
            showMessage(mount, 'No clusters in this report.');
            return;
        }
        clusters.forEach((c) => mount.appendChild(renderCluster(c, paperIndex, { mode })));
    }

    function renderHeader(report) {
        $('reportDate').textContent = report.report_date || '—';
        $('paperCount').textContent = (report.paper_count || 0) + ' papers';
        $('windowDays').textContent = (report.window_days || 90) + 'd window';
        $('overview').textContent = report.overview || '';
    }

    function setupTabs() {
        document.querySelectorAll('.tab-button').forEach((btn) => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-button').forEach((b) => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
                btn.classList.add('active');
                const target = document.getElementById(btn.dataset.target);
                if (target) target.classList.add('active');
            });
        });
    }

    async function loadReport(name) {
        const trendsMount = $('trendsPanel');
        const gapsMount = $('gapsPanel');
        showMessage(trendsMount, 'Loading…');
        clear(gapsMount);
        try {
            const report = await fetchJson(REPORT_PATH(name));
            renderHeader(report);
            renderClusters(report.clusters, report.paper_index, trendsMount, 'trends');
            renderClusters(report.clusters, report.paper_index, gapsMount, 'gaps');
        } catch (e) {
            showMessage(trendsMount, 'Failed to load report: ' + e.message, 'error');
        }
    }

    async function init() {
        setupTabs();
        const select = $('reportSelect');
        try {
            const txt = await fetchText(LIST_PATH);
            const files = txt.split('\n').map((s) => s.trim()).filter((s) => s.endsWith('.json'));
            files.sort().reverse();
            if (files.length === 0) {
                showMessage($('trendsPanel'), 'No trend reports yet. The first one will appear after the next Sunday run.');
                return;
            }
            files.forEach((f) => {
                const opt = document.createElement('option');
                opt.value = f;
                opt.textContent = f.replace('.json', '');
                select.appendChild(opt);
            });
            select.addEventListener('change', () => loadReport(select.value));
            select.value = files[0];
            await loadReport(files[0]);
        } catch (e) {
            showMessage($('trendsPanel'), 'Could not load report list: ' + e.message, 'error');
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
