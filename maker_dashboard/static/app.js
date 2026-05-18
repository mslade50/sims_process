// Alpine.js component for the maker dashboard.
// Three tabs, each with its own ag-grid instance.

const grids = { proposals: null, open: null, exposures: null, fills: null, pnl: null };

// Default expiry = tomorrow 06:00 in the user's local timezone. The
// <input type="datetime-local"> control reads "YYYY-MM-DDTHH:MM" in local
// time, and new Date(localString) → unix seconds via getTime()/1000 picks
// up the local offset automatically. Leaves the user to confirm/edit before
// sending; clearing the field → null = true GTC.
function defaultExpiry() {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  d.setHours(6, 0, 0, 0);
  const pad = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function makerDashboard() {
  return {
    tab: "proposals",

    // Proposals state
    proposals: { rows: [], scanTs: null, staleMinutes: null },
    confirmSend: false,
    confirmToken: "",
    sending: false,
    sendResult: null,

    // Order expiry — global setting applied to every order in the next send.
    // Empty string = no expiry (true GTC). Local datetime string from the
    // <input type="datetime-local"> control.
    expiry: defaultExpiry(),

    // Open Orders state
    openOrders: { rows: [], fetchedTs: null, golfOnly: false },

    // Exposures state
    exposures: { rows: [], fetchedTs: null, golfOnly: true },

    // Fills state
    fills: { rows: [], fetchedTs: null, golfOnly: false },
    fillDetail: null,

    // PnL state
    pnl: {
      rows: [],
      totals: { realized: 0, unrealized: 0, fees: 0, net: 0, open_count: 0, settled_count: 0 },
      fetchedTs: null,
      fromCache: false,
      newSettledCount: 0,
      groupBy: "none",
      golfOnly: true,
      statusFilter: "all",  // 'all' | 'open' | 'settled'
      // Bumped on every grid filter/sort change so chip totals recompute
      // off the post-filter row set. Alpine tracks this reactively.
      filterTick: 0,
    },

    async initAll() {
      await this.loadProposals();
      // Background-load other tabs so first switch feels instant.
      this.loadOpenOrders();
      this.loadExposures();
      this.loadFills();
      this.loadPnL();
    },

    async switchTab(name) {
      this.tab = name;
      // Grids initialized in a hidden div get 0-height; rerender on first show.
      await this.$nextTick();
      if (name === "proposals") this.renderProposalsGrid();
      if (name === "open")      this.renderOpenGrid();
      if (name === "exposures") this.renderExposuresGrid();
      if (name === "fills")     this.renderFillsGrid();
      if (name === "pnl")       this.renderPnLGrid();
    },

    // ── Proposals ───────────────────────────────────────────────
    async loadProposals() {
      const r = await fetch("/api/proposals");
      if (!r.ok) { alert(`Failed to load proposals: ${r.status}`); return; }
      const data = await r.json();
      this.proposals.rows = data.rows || [];
      this.proposals.scanTs = data.scan_ts || null;
      this.proposals.staleMinutes = data.stale_minutes;
      this.renderProposalsGrid();
    },

    renderProposalsGrid() {
      const colDefs = [
        { headerName: "✓", field: "include", width: 60,
          cellRenderer: "agCheckboxCellRenderer", cellEditor: "agCheckboxCellEditor",
          editable: true, pinned: "left" },
        { headerName: "name", field: "title", width: 340, pinned: "left",
          tooltipField: "ticker",
          valueFormatter: p => p.value || p.data.ticker },
        { headerName: "player", field: "player", width: 160, pinned: "left" },
        { headerName: "mkt", field: "market", width: 80 },
        { headerName: "side", field: "side", width: 70,
          cellStyle: p => ({ color: p.value === "yes" ? "#4caf50" : "#e74c3c" }) },
        { headerName: "type", field: "post_type", width: 90 },
        { headerName: "bid", field: "best_bid", width: 80,
          valueFormatter: p => (p.value * 100).toFixed(1) + "c" },
        { headerName: "ask", field: "best_ask", width: 80,
          valueFormatter: p => (p.value * 100).toFixed(1) + "c" },
        { headerName: "post", field: "post_price", width: 80,
          valueFormatter: p => (p.value * 100).toFixed(1) + "c",
          cellStyle: { fontWeight: "600" } },
        { headerName: "sim", field: "sim_prob", width: 80,
          valueFormatter: p => (p.value * 100).toFixed(1) + "c" },
        { headerName: "edge", field: "edge_pp", width: 80,
          valueFormatter: p => p.value.toFixed(1) + "%" },
        { headerName: "kelly%", field: "kelly_f", width: 80,
          valueFormatter: p => (p.value * 100).toFixed(1) + "%" },
        { headerName: "kelly qty", field: "kelly_contracts", width: 100,
          cellStyle: { color: "#888" } },
        { headerName: "edit qty", field: "edit_contracts", width: 100,
          editable: true,
          cellStyle: { fontWeight: "600", backgroundColor: "#2a2f38" },
          valueParser: p => {
            const n = parseInt(p.newValue, 10);
            return isNaN(n) ? 0 : Math.max(0, n);
          } },
        { headerName: "$ stake", field: "stake_dollars", width: 100,
          valueGetter: p => p.data.edit_contracts * p.data.post_price,
          valueFormatter: p => "$" + p.value.toFixed(2) },
      ];
      this.bindGrid("proposals", "#grid-proposals", colDefs, this.proposals.rows,
        () => { this.proposals.rows = [...this.proposals.rows]; });
    },

    approvedRows() {
      if (!grids.proposals) return this.proposals.rows.filter(r => r.include && r.edit_contracts > 0);
      const out = [];
      grids.proposals.forEachNode(n => {
        if (n.data.include && n.data.edit_contracts > 0) out.push(n.data);
      });
      return out;
    },
    approvedCount() { return this.approvedRows().length; },
    approvedContracts() {
      return this.approvedRows().reduce((s, r) => s + Number(r.edit_contracts || 0), 0);
    },
    approvedDollars() {
      return this.approvedRows().reduce(
        (s, r) => s + Number(r.edit_contracts || 0) * Number(r.post_price || 0), 0);
    },
    biggestPlayer() {
      const by = {};
      for (const r of this.approvedRows()) {
        const k = r.player;
        by[k] = (by[k] || 0) + Number(r.edit_contracts || 0) * Number(r.post_price || 0);
      }
      const top = Object.entries(by).sort((a, b) => b[1] - a[1])[0];
      return top ? `${top[0]} $${top[1].toFixed(0)}` : "—";
    },

    resetAllSizes() {
      if (!grids.proposals) return;
      grids.proposals.forEachNode(n =>
        n.setDataValue("edit_contracts", n.data.kelly_contracts));
      this.proposals.rows = [...this.proposals.rows];
    },
    setAllInclude(v) {
      if (!grids.proposals) return;
      grids.proposals.forEachNode(n => n.setDataValue("include", v));
      this.proposals.rows = [...this.proposals.rows];
    },

    // Local-datetime input → unix seconds. Returns null when the field is
    // blank, which the server interprets as no expiry / true GTC.
    expirySeconds() {
      if (!this.expiry) return null;
      const ms = new Date(this.expiry).getTime();
      if (!Number.isFinite(ms)) return null;
      return Math.floor(ms / 1000);
    },

    async sendOrders() {
      this.sending = true;
      try {
        const r = await fetch("/api/send", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rows: this.approvedRows(),
            expiration_ts: this.expirySeconds(),
          }),
        });
        const data = await r.json();
        if (!r.ok) { alert(`Send failed: ${data.error || r.status}`); return; }
        this.sendResult = data;
        this.confirmSend = false;
        this.confirmToken = "";
      } catch (e) {
        alert(`Send error: ${e}`);
      } finally {
        this.sending = false;
      }
    },

    // ── Open Orders ──────────────────────────────────────────────
    async loadOpenOrders() {
      const r = await fetch("/api/open-orders");
      if (!r.ok) { alert(`Failed to load open orders: ${r.status}`); return; }
      const data = await r.json();
      this.openOrders.rows = data.rows || [];
      this.openOrders.fetchedTs = data.fetched_ts || new Date().toISOString();
      this.renderOpenGrid();
    },

    visibleOpenOrders() {
      return this.openOrders.golfOnly
        ? this.openOrders.rows.filter(r => r.is_golf)
        : this.openOrders.rows;
    },
    openTotalContracts() {
      return this.visibleOpenOrders().reduce((s, r) => s + Number(r.remaining_contracts || 0), 0);
    },
    openTotalDollars() {
      return this.visibleOpenOrders().reduce(
        (s, r) => s + Number(r.remaining_contracts || 0) * Number(r.price_dollars || 0), 0);
    },
    openScriptCount() {
      return this.visibleOpenOrders().filter(r => r.placed_by === "script").length;
    },
    openManualCount() {
      return this.visibleOpenOrders().filter(r => r.placed_by === "manual").length;
    },

    renderOpenGrid() {
      const colDefs = [
        { headerName: "name", field: "title", width: 380, pinned: "left",
          tooltipField: "ticker",
          valueFormatter: p => p.value || p.data.ticker,
          cellStyle: p => p.data.is_golf ? null : { color: "#888" } },
        { headerName: "side", field: "side", width: 70,
          cellStyle: p => ({ color: p.value === "yes" ? "#4caf50" : "#e74c3c" }) },
        { headerName: "price", field: "price_dollars", width: 90,
          valueFormatter: p => (p.value * 100).toFixed(1) + "c",
          cellStyle: { fontWeight: "600" }, sort: null },
        { headerName: "contracts", field: "remaining_contracts", width: 110,
          sort: "desc", sortIndex: 0,
          cellStyle: { fontWeight: "600", textAlign: "right" } },
        { headerName: "filled / initial", field: "fill_progress", width: 140,
          valueGetter: p => `${p.data.fill_count} / ${p.data.initial_count}` },
        { headerName: "$ value", field: "value_dollars", width: 100,
          valueGetter: p => p.data.remaining_contracts * p.data.price_dollars,
          valueFormatter: p => "$" + p.value.toFixed(2) },
        { headerName: "placed by", field: "placed_by", width: 100,
          cellStyle: p => ({ color: p.value === "script" ? "#4a90e2" : "#f5a623" }) },
        { headerName: "scope", field: "scope", width: 90,
          cellStyle: p => ({ color: p.value === "golf" ? "#4caf50" : "#888" }) },
        { headerName: "created", field: "created_time", width: 200,
          valueFormatter: p => p.value ? p.value.replace("T", " ").substring(0, 19) : "" },
        { headerName: "order_id", field: "order_id", width: 280,
          cellStyle: { color: "#888", fontFamily: "monospace", fontSize: "11px" } },
      ];
      this.bindGrid("open", "#grid-open", colDefs, this.visibleOpenOrders(), null);
    },

    // ── Exposures ────────────────────────────────────────────────
    async loadExposures() {
      const r = await fetch("/api/positions");
      if (!r.ok) { alert(`Failed to load positions: ${r.status}`); return; }
      const data = await r.json();
      this.exposures.rows = data.rows || [];
      this.exposures.fetchedTs = data.fetched_ts || new Date().toISOString();
      this.renderExposuresGrid();
    },

    visibleExposures() {
      return this.exposures.golfOnly
        ? this.exposures.rows.filter(r => r.is_golf)
        : this.exposures.rows;
    },
    exposuresAtRisk() {
      return this.visibleExposures().reduce(
        (s, r) => s + Number(r.exposure_dollars || 0), 0);
    },
    exposuresMaxGain() {
      return this.visibleExposures().reduce(
        (s, r) => s + Number(r.max_gain_dollars || 0), 0);
    },
    exposuresNetPnl() {
      return this.visibleExposures().reduce(
        (s, r) => s + Number(r.net_pnl_dollars || 0), 0);
    },
    exposuresTotalContracts() {
      return this.visibleExposures().reduce(
        (s, r) => s + Number(r.contracts || 0), 0);
    },

    renderExposuresGrid() {
      const colDefs = [
        { headerName: "name", field: "title", width: 380, pinned: "left",
          tooltipField: "ticker",
          valueFormatter: p => p.value || p.data.ticker,
          cellStyle: p => p.data.is_golf ? null : { color: "#888" } },
        { headerName: "side", field: "side", width: 70,
          cellStyle: p => ({ color: p.value === "yes" ? "#4caf50" : "#e74c3c", fontWeight: "600" }) },
        { headerName: "contracts", field: "contracts", width: 110,
          cellStyle: { fontWeight: "600", textAlign: "right" } },
        { headerName: "avg cost", field: "avg_cost_dollars", width: 100,
          valueFormatter: p => (p.value * 100).toFixed(1) + "c" },
        { headerName: "$ at risk", field: "exposure_dollars", width: 110,
          sort: "desc", sortIndex: 0,
          valueFormatter: p => "$" + Number(p.value).toFixed(2),
          cellStyle: { fontWeight: "600", color: "#f5a623" } },
        { headerName: "$ max gain", field: "max_gain_dollars", width: 110,
          valueFormatter: p => "$" + Number(p.value).toFixed(2),
          cellStyle: { color: "#4caf50" } },
        { headerName: "realized P&L", field: "net_pnl_dollars", width: 110,
          valueFormatter: p => "$" + Number(p.value).toFixed(2),
          cellStyle: p => ({
            color: p.value > 0 ? "#4caf50" : (p.value < 0 ? "#e74c3c" : "#888"),
            fontWeight: "600"
          }) },
        { headerName: "fees", field: "fees_paid_dollars", width: 80,
          valueFormatter: p => "$" + Number(p.value).toFixed(2),
          cellStyle: { color: "#888" } },
        { headerName: "resting", field: "resting_orders_count", width: 90,
          cellStyle: p => ({ color: p.value > 0 ? "#4a90e2" : "#888" }) },
        { headerName: "last updated", field: "last_updated_ts", width: 200,
          valueFormatter: p => p.value ? p.value.replace("T", " ").substring(0, 19) : "",
          cellStyle: { color: "#888" } },
      ];
      this.bindGrid("exposures", "#grid-exposures", colDefs, this.visibleExposures(), null);
    },

    // ── Fills ────────────────────────────────────────────────────
    async loadFills() {
      const r = await fetch("/api/fills");
      if (!r.ok) { alert(`Failed to load fills: ${r.status}`); return; }
      const data = await r.json();
      this.fills.rows = data.rows || [];
      this.fills.fetchedTs = data.fetched_ts || new Date().toISOString();
      this.renderFillsGrid();
    },

    visibleFills() {
      return this.fills.golfOnly
        ? this.fills.rows.filter(r => r.is_golf)
        : this.fills.rows;
    },
    fillsTotalContracts() {
      return this.visibleFills().reduce((s, r) => s + Number(r.count || 0), 0);
    },
    fillsTotalDollars() {
      return this.visibleFills().reduce(
        (s, r) => s + Number(r.count || 0) * Number(r.fill_price_dollars || 0), 0);
    },
    fillsMakerCount() {
      return this.visibleFills().filter(r => !r.is_taker).length;
    },
    fillsTakerCount() {
      return this.visibleFills().filter(r => r.is_taker).length;
    },

    renderFillsGrid() {
      const colDefs = [
        { headerName: "time", field: "created_time", width: 200, sort: "desc", sortIndex: 0,
          valueFormatter: p => p.value ? p.value.replace("T", " ").substring(0, 19) : "" },
        { headerName: "name", field: "title", width: 380,
          tooltipField: "ticker",
          valueFormatter: p => p.value || p.data.ticker,
          cellStyle: p => p.data.is_golf ? null : { color: "#888" } },
        { headerName: "side", field: "side", width: 70,
          cellStyle: p => ({ color: p.value === "yes" ? "#4caf50" : "#e74c3c" }) },
        { headerName: "action", field: "action", width: 80 },
        { headerName: "count", field: "count", width: 90,
          cellStyle: { fontWeight: "600", textAlign: "right" } },
        { headerName: "fill price", field: "fill_price_dollars", width: 100,
          valueFormatter: p => (p.value * 100).toFixed(1) + "c",
          cellStyle: { fontWeight: "600" } },
        { headerName: "$ cost", field: "fill_dollars", width: 100,
          valueGetter: p => p.data.count * p.data.fill_price_dollars,
          valueFormatter: p => "$" + p.value.toFixed(2) },
        { headerName: "is_taker", field: "is_taker", width: 90,
          cellStyle: p => ({ color: p.value ? "#e74c3c" : "#4caf50", fontWeight: "600" }),
          valueFormatter: p => p.value ? "TAKER" : "maker" },
        { headerName: "order_id", field: "order_id", width: 280,
          cellStyle: { color: "#888", fontFamily: "monospace", fontSize: "11px" } },
      ];
      this.bindGrid("fills", "#grid-fills", colDefs, this.visibleFills(), null);
      // Wire double-click to open detail modal
      if (grids.fills) {
        grids.fills.addEventListener("rowDoubleClicked", (e) => {
          this.fillDetail = e.data;
        });
      }
    },

    // ── PnL ──────────────────────────────────────────────────────
    async loadPnL(forceRefresh = false) {
      const url = forceRefresh ? "/api/pnl?refresh=1" : "/api/pnl";
      const r = await fetch(url);
      if (!r.ok) { alert(`Failed to load PnL: ${r.status}`); return; }
      const data = await r.json();
      this.pnl.rows = data.rows || [];
      this.pnl.totals = data.totals || this.pnl.totals;
      this.pnl.fetchedTs = data.fetched_ts || new Date().toISOString();
      this.pnl.fromCache = !!data.from_cache;
      this.pnl.newSettledCount = data.new_settled_count || 0;
      this.renderPnLGrid();
    },

    // After applying golf/status filters, return the per-position rows.
    // pnlGroupedRows() applies the group-by aggregation on top.
    visiblePnL() {
      let rows = this.pnl.rows;
      if (this.pnl.golfOnly) rows = rows.filter(r => r.is_golf);
      if (this.pnl.statusFilter === "open")    rows = rows.filter(r => r.is_open);
      if (this.pnl.statusFilter === "settled") rows = rows.filter(r => !r.is_open);
      return rows;
    },

    // Group-aware row set fed to AG Grid. Aggregates per group into one
    // summary row so we don't need Enterprise row-grouping.
    pnlGroupedRows() {
      const gb = this.pnl.groupBy;
      const rows = this.visiblePnL();
      if (!gb || gb === "none") return rows;
      const groupKey = r => {
        if (gb === "is_open") return r.is_open ? "open" : "settled";
        return r[gb] ?? "(blank)";
      };
      const buckets = new Map();
      for (const r of rows) {
        const k = groupKey(r);
        if (!buckets.has(k)) {
          buckets.set(k, {
            _group: k,
            n_positions: 0,
            contracts: 0,
            realized_pnl_dollars: 0,
            unrealized_pnl_dollars: 0,
            fees_paid_dollars: 0,
            net_pnl_dollars: 0,
            exposure_dollars: 0,
            invested_dollars: 0,
            n_open: 0,
            n_settled: 0,
          });
        }
        const b = buckets.get(k);
        b.n_positions += 1;
        b.contracts += Number(r.contracts || 0);
        b.realized_pnl_dollars += Number(r.realized_pnl_dollars || 0);
        b.unrealized_pnl_dollars += Number(r.unrealized_pnl_dollars || 0);
        b.fees_paid_dollars += Number(r.fees_paid_dollars || 0);
        b.net_pnl_dollars += Number(r.net_pnl_dollars || 0);
        b.exposure_dollars += Number(r.exposure_dollars || 0);
        b.invested_dollars += Number(r.invested_dollars || 0);
        if (r.is_open) b.n_open += 1; else b.n_settled += 1;
      }
      return Array.from(buckets.values())
        .sort((a, b) => b.net_pnl_dollars - a.net_pnl_dollars);
    },

    // Post-grid-filter row set. When ag-grid has applied column filters,
    // walk the displayed nodes; otherwise fall back to the pre-grid
    // (golf/status) filter result. Reading filterTick makes Alpine
    // re-evaluate whenever the grid signals a filter/sort change.
    pnlGridRows() {
      void this.pnl.filterTick;
      const g = grids.pnl;
      if (g && typeof g.forEachNodeAfterFilter === "function") {
        const out = [];
        g.forEachNodeAfterFilter(n => { if (n.data) out.push(n.data); });
        return out;
      }
      return this.pnlGroupedRows();
    },

    // Totals are derived from the currently-visible (post-grid-filter)
    // row set so column filters AND golf/status filters change the chips.
    pnlTotal(field) {
      const map = { net: "net_pnl_dollars",
                    realized: "realized_pnl_dollars",
                    unrealized: "unrealized_pnl_dollars",
                    fees: "fees_paid_dollars",
                    invested: "invested_dollars" };
      const col = map[field] || field;
      return this.pnlGridRows().reduce((s, r) => s + Number(r[col] || 0), 0);
    },
    pnlROI() {
      const inv = this.pnlTotal("invested");
      return inv > 0 ? (this.pnlTotal("net") / inv) * 100 : 0;
    },
    // Position counts on grouped rows aggregate n_open/n_settled fields
    // (per-bucket); on flat rows fall back to scanning is_open.
    pnlOpenCount() {
      return this.pnlGridRows().reduce(
        (s, r) => s + (r.n_open != null ? Number(r.n_open) : (r.is_open ? 1 : 0)), 0);
    },
    pnlSettledCount() {
      return this.pnlGridRows().reduce(
        (s, r) => s + (r.n_settled != null ? Number(r.n_settled) : (r.is_open ? 0 : 1)), 0);
    },

    renderPnLGrid() {
      const groupBy = this.pnl.groupBy;
      const grouped = groupBy && groupBy !== "none";

      const pnlCell = p => ({
        color: (Number(p.value) || 0) > 0 ? "#4caf50"
             : (Number(p.value) || 0) < 0 ? "#e74c3c" : "#888",
        fontWeight: "600",
      });
      const dollarFmt = p => p.value != null
        ? "$" + Number(p.value).toFixed(2) : "";

      let colDefs;
      if (grouped) {
        // Aggregated view: one row per group.
        const groupHeader = ({
          player_code: "golfer",
          market_type: "market type",
          event_code: "tournament",
          side: "side",
          is_open: "status",
        })[groupBy] || "group";
        colDefs = [
          { headerName: groupHeader, field: "_group", width: 200, pinned: "left",
            cellStyle: { fontWeight: "600" } },
          { headerName: "# pos", field: "n_positions", width: 75,
            cellStyle: { textAlign: "right", color: "#aaa" } },
          { headerName: "open/settled", field: "open_settled", width: 120,
            valueGetter: p => `${p.data.n_open} / ${p.data.n_settled}` },
          { headerName: "contracts", field: "contracts", width: 100,
            cellStyle: { textAlign: "right" } },
          { headerName: "exposure", field: "exposure_dollars", width: 100,
            valueFormatter: dollarFmt, cellStyle: { color: "#f5a623" } },
          { headerName: "realized", field: "realized_pnl_dollars", width: 110,
            valueFormatter: dollarFmt, cellStyle: pnlCell },
          { headerName: "unrealized", field: "unrealized_pnl_dollars", width: 110,
            valueFormatter: dollarFmt, cellStyle: pnlCell },
          { headerName: "fees", field: "fees_paid_dollars", width: 90,
            valueFormatter: dollarFmt, cellStyle: { color: "#888" } },
          { headerName: "net", field: "net_pnl_dollars", width: 110,
            sort: "desc", sortIndex: 0,
            valueFormatter: dollarFmt,
            cellStyle: p => ({ ...pnlCell(p), fontWeight: "700" }) },
        ];
      } else {
        // Flat per-position view.
        colDefs = [
          { headerName: "name", field: "title", width: 320, pinned: "left",
            tooltipField: "ticker",
            valueFormatter: p => p.value || p.data?.ticker,
            cellStyle: p => p.data?.is_golf ? null : { color: "#888" } },
          { headerName: "golfer", field: "player_code", width: 80 },
          { headerName: "market", field: "market_type", width: 80 },
          { headerName: "event", field: "event_code", width: 80 },
          // Human-readable event name parsed from the title. Two shapes:
          //   "RBC Heritage: Will Brian Harman finish top 10?"   → "RBC Heritage"
          //   "Will Robert MacIntyre win the RBC Heritage?"      → "RBC Heritage"
          { headerName: "event name", field: "event_name", width: 180,
            valueGetter: p => {
              const t = p.data?.title || "";
              if (!t) return "";
              const colon = t.indexOf(":");
              if (colon > 0) return t.slice(0, colon).trim();
              const m = t.match(/win the (.+?)\??$/i);
              return m ? m[1].trim() : "";
            } },
          { headerName: "side", field: "side", width: 60,
            cellStyle: p => p.value === "yes"
              ? { color: "#4caf50", fontWeight: "600" }
              : (p.value === "no" ? { color: "#e74c3c", fontWeight: "600" } : null) },
          { headerName: "status", field: "is_open", width: 80,
            valueFormatter: p => p.value ? "open" : "settled",
            cellStyle: p => p.value ? { color: "#4a90e2" } : { color: "#888" } },
          { headerName: "contracts", field: "contracts", width: 95,
            cellStyle: { textAlign: "right" } },
          { headerName: "avg cost", field: "avg_cost_dollars", width: 85,
            valueFormatter: p => p.value ? (p.value * 100).toFixed(1) + "c" : "" },
          { headerName: "mark", field: "mark_dollars", width: 75,
            valueFormatter: p => p.value ? (p.value * 100).toFixed(1) + "c" : "",
            cellStyle: { color: "#aaa" } },
          { headerName: "realized", field: "realized_pnl_dollars", width: 100,
            valueFormatter: dollarFmt, cellStyle: pnlCell },
          { headerName: "unrealized", field: "unrealized_pnl_dollars", width: 105,
            valueFormatter: dollarFmt, cellStyle: pnlCell },
          { headerName: "fees", field: "fees_paid_dollars", width: 75,
            valueFormatter: dollarFmt, cellStyle: { color: "#888" } },
          { headerName: "net", field: "net_pnl_dollars", width: 100,
            sort: "desc", sortIndex: 0,
            valueFormatter: dollarFmt,
            cellStyle: p => ({ ...pnlCell(p), fontWeight: "700" }) },
          { headerName: "return %", field: "return_pct", width: 85,
            valueFormatter: p => p.value != null
              ? Number(p.value).toFixed(1) + "%" : "",
            cellStyle: pnlCell },
        ];
      }
      this.bindGrid("pnl", "#grid-pnl", colDefs, this.pnlGroupedRows(), null);
      // Bump filterTick on any row-set change so the chip totals recompute
      // off the post-filter visible rows. We listen on multiple events
      // because column filters fire filterChanged, but quick-filter / sort
      // / data-change all surface through modelUpdated.
      if (grids.pnl && typeof grids.pnl.addEventListener === "function") {
        const evts = ["filterChanged", "modelUpdated"];
        if (this._pnlListener) {
          for (const e of evts) grids.pnl.removeEventListener(e, this._pnlListener);
        }
        this._pnlListener = () => { this.pnl.filterTick++; };
        for (const e of evts) grids.pnl.addEventListener(e, this._pnlListener);
      }
    },

    // ── Grid binding helper ──────────────────────────────────────
    bindGrid(name, selector, colDefs, rowData, onCellValueChanged, extraOptions) {
      const div = document.querySelector(selector);
      if (!div) return;
      if (grids[name]) {
        grids[name].setGridOption("columnDefs", colDefs);
        grids[name].setGridOption("rowData", rowData);
        // Apply any grouping-related options on re-bind so the user can
        // change group-by without a full re-create.
        if (extraOptions) {
          for (const [k, v] of Object.entries(extraOptions)) {
            grids[name].setGridOption(k, v);
          }
        }
        return;
      }
      const options = {
        columnDefs: colDefs,
        rowData: rowData,
        defaultColDef: { sortable: true, resizable: true, filter: true },
        animateRows: false,
        rowHeight: 28,
        headerHeight: 32,
        ...(extraOptions || {}),
      };
      if (onCellValueChanged) options.onCellValueChanged = onCellValueChanged;
      grids[name] = agGrid.createGrid(div, options);
    },
  };
}
