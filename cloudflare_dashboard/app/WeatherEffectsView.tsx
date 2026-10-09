"use client";

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { Area, Bar, CartesianGrid, ComposedChart, Line, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EmptyState, Kpi, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { useDashboardData } from "./data";
import { etTime } from "./lib";
import {
  COMPONENT_HINTS, COMPONENT_LABELS, filterPlayers, modelName, windSourceText, forecastAgeHours, hhmm, parseWeather, signedFixed, sortPlayers, toHour, weatherCheckpoint,
  type HourlyPoint, type PlayerRow, type SortKey, type Wave, type WeatherDoc,
} from "./weather-rules";

type IndexEvent = { event_uid: string; name: string; tour?: string; date_start?: string; course?: string; weather_key?: string | null; weather_run?: string | null; weather_as_of?: string | null; explain_run?: string | null };
type PublishIndex = { events?: IndexEvent[] };

const WIND = "var(--wx-wind)";
const GUST = "var(--wx-gust)";
const TEMP = "var(--wx-temp)";
const RAIN = "var(--wx-rain)";
const WAVE_SHADES = ["var(--wave-am-fill)", "var(--wave-pm-fill)"];

const tourWord = (tour: string) => ({ pga: "PGA Tour", euro: "DP World Tour" } as Record<string, string>)[tour.toLowerCase()] ?? tour.toUpperCase();

export const weatherKey = (event: IndexEvent) => event.weather_key || `golfprice/weather/${event.event_uid.replace(/:/g, "_")}/latest.json`;

/* ------------------------------------------------------------------ small pieces */
function Arrow({ cx, cy, dir }: { cx?: number; cy?: number; dir: number | null }) {
  if (cx === undefined || cy === undefined || dir === null) return <g />;
  // meteorological direction is where the wind comes FROM; the arrow points where it blows TO
  return (
    <g transform={`translate(${cx},${cy}) rotate(${dir + 180})`}>
      <path d="M0,-14 L6,5 L0,1 L-6,5 Z" fill={WIND} stroke="var(--background)" strokeWidth={1} />
    </g>
  );
}

/** "early" as "Early wave"; a wave already named something else keeps its name; missing names become "Wave 2". */
const waveName = (wave: string, index: number): string => {
  const w = wave.trim();
  if (!w) return `Wave ${index + 1}`;
  return /wave/i.test(w) ? w : `${w.replace(/^./, (c) => c.toUpperCase())} wave`;
};

const compass = (deg: number) => ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round((((deg % 360) + 360) % 360) / 45) % 8];

type Row = HourlyPoint & { windBand: [number, number] | null; tempBand: [number, number] | null };
const band = (lo: number | null, hi: number | null): [number, number] | null => (lo !== null && hi !== null ? [lo, hi] : null);

function WxTooltip({ active, payload, unit }: { active?: boolean; payload?: Array<{ payload: Row }>; unit: "wind" | "temp" | "rain" }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="wx-tooltip">
      <strong>{hhmm(p.hour)}</strong>
      {unit === "wind" && (
        <>
          <span>Wind {p.wind?.toFixed(1) ?? "n/a"} mph{p.wind_p10 !== null && p.wind_p90 !== null ? ` (likely range ${p.wind_p10.toFixed(1)} to ${p.wind_p90.toFixed(1)})` : ""}</span>
          {p.gust !== null && <span>Gusts {p.gust.toFixed(1)} mph</span>}
          {p.dir !== null && <span>Blowing from the {compass(p.dir)}</span>}
        </>
      )}
      {unit === "temp" && <span>Temperature {p.temp?.toFixed(1) ?? "n/a"}°F{p.temp_p10 !== null && p.temp_p90 !== null ? ` (likely range ${p.temp_p10.toFixed(1)} to ${p.temp_p90.toFixed(1)})` : ""}</span>}
      {unit === "rain" && <span>Rain {p.rain?.toFixed(2) ?? "n/a"} mm</span>}
    </div>
  );
}

function HourlyCharts({ points, waves }: { points: HourlyPoint[]; waves: Wave[] }) {
  const rows: Row[] = points.map((p) => ({ ...p, windBand: band(p.wind_p10, p.wind_p90), tempBand: band(p.temp_p10, p.temp_p90) }));
  const hours = rows.map((r) => r.hour);
  const domain: [number, number] = [Math.floor(Math.min(...hours)), Math.ceil(Math.max(...hours))];
  const windows = waves.filter((w) => w.window);
  const shading = windows.map((w, i) => (
    <ReferenceArea key={`${w.wave}-${i}`} x1={w.window![0]} x2={w.window![1]} fill={WAVE_SHADES[i % 2]} stroke="none" ifOverflow="hidden"
      label={{ value: waveName(w.wave, i), position: "insideTop", fill: "var(--muted)", fontSize: 10 }} />
  ));
  const axis = <XAxis type="number" dataKey="hour" domain={domain} allowDecimals={false} stroke="var(--muted)" tickFormatter={(v) => `${String(v).padStart(2, "0")}:00`} />;
  const margin = { top: 14, right: 12, bottom: 4, left: 0 };
  const hasWind = rows.some((r) => r.wind !== null);
  const hasTemp = rows.some((r) => r.temp !== null);
  const hasRain = rows.some((r) => r.rain !== null);
  return (
    <div className="wx-charts">
      {hasWind && (
        <div>
          <h3 className="inputs-h3">Wind and gusts (mph). The shaded band is the likely range; arrows point where the wind blows.</h3>
          <div className="wx-chart" role="img" aria-label="Hourly wind forecast with uncertainty band and tee windows">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={margin}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                {axis}
                <YAxis stroke="var(--muted)" width={34} />
                <Tooltip content={<WxTooltip unit="wind" />} />
                {shading}
                <Area dataKey="windBand" stroke="none" fill={WIND} fillOpacity={0.2} isAnimationActive={false} />
                <Line dataKey="gust" stroke={GUST} strokeDasharray="4 3" dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls />
                <Line dataKey="wind" stroke={WIND} strokeWidth={2.5} isAnimationActive={false} connectNulls
                  dot={(props: { cx?: number; cy?: number; payload?: Row; index?: number }) => <Arrow key={props.index} cx={props.cx} cy={props.cy} dir={props.payload?.dir ?? null} />} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      {hasTemp && (
        <div>
          <h3 className="inputs-h3">Temperature (°F). The shaded band is the likely range.</h3>
          <div className="wx-chart short" role="img" aria-label="Hourly temperature forecast with uncertainty band">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={margin}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                {axis}
                <YAxis stroke="var(--muted)" width={34} domain={["auto", "auto"]} />
                <Tooltip content={<WxTooltip unit="temp" />} />
                {shading}
                <Area dataKey="tempBand" stroke="none" fill={TEMP} fillOpacity={0.2} isAnimationActive={false} />
                <Line dataKey="temp" stroke={TEMP} strokeWidth={2.5} dot={false} isAnimationActive={false} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      {hasRain && (
        <div>
          <h3 className="inputs-h3">Rain (mm per hour)</h3>
          <div className="wx-chart short" role="img" aria-label="Hourly rain forecast">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={margin}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                {axis}
                <YAxis stroke="var(--muted)" width={34} />
                <Tooltip content={<WxTooltip unit="rain" />} />
                {shading}
                <Bar dataKey="rain" fill={RAIN} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

/** Expected strokes vs the field with its p10-p90 band, drawn on one shared symmetric scale. */
function RangeBar({ rel, lo, hi, scale }: { rel: number | null; lo: number | null; hi: number | null; scale: number }) {
  if (rel === null) return <div className="wx-range empty" />;
  const pos = (v: number) => `${Math.max(0, Math.min(100, 50 + (v / scale) * 50))}%`;
  const a = lo ?? rel;
  const b = hi ?? rel;
  return (
    <div className="wx-range" aria-hidden="true">
      <i className="wx-zero" />
      <i className="wx-band" style={{ left: pos(Math.min(a, b)), width: `calc(${pos(Math.max(a, b))} - ${pos(Math.min(a, b))})` }} />
      <i className="wx-dot" style={{ left: pos(rel) }} />
    </div>
  );
}

function WaveCards({ waves, scale }: { waves: Wave[]; scale: number }) {
  if (!waves.length) return <p className="inputs-muted">No tee waves yet because the tee sheet is not out.</p>;
  const diff = waves.length >= 2 && waves[0].rel !== null && waves[1].rel !== null ? waves[1].rel - waves[0].rel : null;
  return (
    <div>
      <div className="wx-waves">
        {waves.map((w, i) => (
          <div className="wx-wave" key={`${w.wave}-${i}`}>
            <span className="eyebrow">{waveName(w.wave, i)}{w.window ? `, tee times ${hhmm(w.window[0])} to ${hhmm(w.window[1])}` : w.windowText ? `, tee times ${w.windowText}` : ""}</span>
            <strong className={w.rel === null ? "" : w.rel > 0 ? "negative" : "positive"} title="Expected strokes per round compared with the field average. Positive means a harder draw.">{signedFixed(w.rel)} strokes</strong>
            <small>{w.rel_p10 !== null && w.rel_p90 !== null ? `Likely range ${signedFixed(w.rel_p10)} to ${signedFixed(w.rel_p90)}. ` : ""}{w.mean_wind !== null ? `Wind ${w.mean_wind.toFixed(1)} mph` : ""}{w.mean_temp !== null ? `${w.mean_wind !== null ? ", " : ""}${w.mean_temp.toFixed(0)}°F` : ""}</small>
            <RangeBar rel={w.rel} lo={w.rel_p10} hi={w.rel_p90} scale={scale} />
          </div>
        ))}
      </div>
      {diff !== null && <p className="inputs-note">Difference between the waves ({waveName(waves[1].wave, 1)} minus {waveName(waves[0].wave, 0)}): <b>{signedFixed(diff)}</b> strokes per round. Positive means the second wave plays harder.</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ player table */
const TEE_ROUNDS = [1, 2];
function PlayersTable({ doc }: { doc: WeatherDoc }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "total12", dir: "asc" });
  const [open, setOpen] = useState<number | null>(null);
  const [limit, setLimit] = useState(40);
  // A round gets a column only if at least one player has something in it (rounds 3 and 4 have no tee sheet early in the week).
  const rounds = useMemo(() => {
    const seen = new Set<number>(TEE_ROUNDS);
    doc.players.forEach((p) => Object.keys(p.rounds).forEach((r) => seen.add(Number(r))));
    return [...seen].filter((r) => r <= 4).sort((a, b) => a - b);
  }, [doc.players]);
  const teeRounds = rounds.filter((r) => doc.players.some((p) => p.rounds[r]?.tee_time));
  const effectRounds = rounds.filter((r) => doc.players.some((p) => p.rounds[r] && (p.rounds[r].mean_off_rel ?? p.rounds[r].mean_off_strokes) !== null));
  const showSd = doc.players.some((p) => p.sdMax !== null);
  const columnCount = 2 + teeRounds.length + effectRounds.length + 1 + (showSd ? 1 : 0);
  const rows = useMemo(() => sortPlayers(filterPlayers(doc.players, query), sort.key, sort.dir), [doc.players, query, sort]);
  const shown = query ? rows : rows.slice(0, limit);

  const toggleSort = (key: SortKey) => setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: key === "name" ? "asc" : "asc" }));
  const head = (key: SortKey, label: string, hint?: string) => (
    <th title={hint} aria-sort={sort.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" className="th-sort" onClick={() => toggleSort(key)}>{label}{sort.key === key ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}</button>
    </th>
  );

  if (!doc.players.length) return <p className="inputs-muted">No per-player effects yet because tee times are not published. Only the whole-field effect is shown above.</p>;
  return (
    <div>
      <div className="wx-search">
        <Search size={16} aria-hidden="true" />
        <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player" aria-label="Search player" />
        <span className="inputs-muted">{rows.length} of {doc.players.length}</span>
      </div>
      <div className="table-scroll wx-table">
        <table aria-label="Weather effect by player">
          <thead>
            <tr>
              <th aria-label="Expand" />
              {head("name", "Player")}
              {teeRounds.map((r) => <th key={`t${r}`} title={`Tee time in round ${r}. (10) means the player starts on the 10th tee.`}><span className="th-text">R{r} tee</span></th>)}
              {effectRounds.map((r) => head(`r${r}` as SortKey, `R${r} vs field`, `Strokes the weather and tee time are expected to add (positive) or save (negative) in round ${r}, compared with the average player in the field.`))}
              {head("total12", "R1+R2", "The weather effect over rounds 1 and 2 combined, in strokes. Negative is a helpful draw.")}
              {showSd && head("sd", "Spread", "How much the weather widens (above 1) or narrows (below 1) the range of this player's possible scores.")}
            </tr>
          </thead>
          <tbody>
            {shown.map((p) => <PlayerRows key={p.dg_id} p={p} rounds={rounds} teeRounds={teeRounds} effectRounds={effectRounds} showSd={showSd} cols={columnCount} open={open === p.dg_id} onToggle={() => setOpen(open === p.dg_id ? null : p.dg_id)} />)}
            {shown.length === 0 && <tr><td colSpan={columnCount}><span className="inputs-muted">No player matches.</span></td></tr>}
          </tbody>
        </table>
      </div>
      {!query && rows.length > limit && <button type="button" className="inputs-button" onClick={() => setLimit((l) => l + 60)}>Show more ({rows.length - limit} left)</button>}
    </div>
  );
}

const titleCaseKey = (key: string) => key.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
const effectClass = (v: number | null) => (v === null ? "" : v > 0.005 ? "negative" : v < -0.005 ? "positive" : "");

function PlayerRows({ p, rounds, teeRounds, effectRounds, showSd, cols, open, onToggle }: { p: PlayerRow; rounds: number[]; teeRounds: number[]; effectRounds: number[]; showSd: boolean; cols: number; open: boolean; onToggle: () => void }) {
  void rounds;
  const names = Object.keys(p.rounds).flatMap((r) => Object.keys(p.rounds[Number(r)].components));
  const comps = [...new Set(names)].filter((c) => c !== "tod");
  const hasSd = Object.values(p.rounds).some((r) => r.sd_mult !== null);
  return (
    <>
      <tr className={open ? "active-row" : undefined}>
        <td><button type="button" className="th-sort" aria-expanded={open} aria-label={`${open ? "Collapse" : "Expand"} ${p.name}`} onClick={onToggle}>{open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}</button></td>
        <td className="wx-name">{p.name}</td>
        {teeRounds.map((r) => <td key={`t${r}`}>{p.rounds[r]?.tee_time ? hhmm(toHour(p.rounds[r].tee_time)) + (p.rounds[r].start_hole === 10 ? " (10)" : "") : "not set"}</td>)}
        {effectRounds.map((r) => { const v = p.rounds[r] ? (p.rounds[r].mean_off_rel ?? p.rounds[r].mean_off_strokes) : null; return <td key={`e${r}`} className={effectClass(v)}>{signedFixed(v)}</td>; })}
        <td className={effectClass(p.total12)}><b>{signedFixed(p.total12)}</b></td>
        {showSd && <td>{p.sdMax === null ? "n/a" : p.sdMax.toFixed(3)}</td>}
      </tr>
      {open && (
        <tr className="wx-detail">
          <td colSpan={cols}>
            <div className="table-scroll">
              <table aria-label={`Weather components for ${p.name}`}>
                <thead>
                  <tr>
                    <th>Round</th>
                    <th title="Strokes added (positive) or saved (negative) compared with the average player in the field.">Vs field</th>
                    <th title="Strokes added or saved compared with calm, average conditions.">Vs calm conditions</th>
                    {comps.map((c) => <th key={c} title={COMPONENT_HINTS[c]}>{COMPONENT_LABELS[c] ?? titleCaseKey(c)}</th>)}
                    <th title="Strokes added or saved by the time of day the player tees off.">Time of day</th>
                    {hasSd && <th title="How much the weather widens (above 1) or narrows (below 1) the range of possible scores.">Spread</th>}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(p.rounds).map(Number).sort((a, b) => a - b).map((r) => {
                    const pr = p.rounds[r];
                    return (
                      <tr key={r}>
                        <td>R{r}</td>
                        <td>{signedFixed(pr.mean_off_rel)}</td>
                        <td>{signedFixed(pr.mean_off_strokes)}</td>
                        {comps.map((c) => <td key={c}>{signedFixed(pr.components[c] ?? null)}</td>)}
                        <td>{signedFixed(pr.tod_strokes)}</td>
                        {hasSd && <td>{pr.sd_mult === null ? "n/a" : pr.sd_mult.toFixed(3)}</td>}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="inputs-muted">Strokes per round; positive means more strokes (worse for the player). The columns on the right are the pieces that add up to the total.</p>
          </td>
        </tr>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ one event */
export function CheckpointBanner({ event, docAsOf }: { event: IndexEvent; docAsOf: string }) {
  const c = weatherCheckpoint(event, docAsOf);
  const headline = c.state === "mismatch" ? "Weather and prices are out of step." : c.weatherAsOf ? `Weather updated ${etTime(c.weatherAsOf)}.` : "Weather update time unknown.";
  return (
    <div className={`inputs-note${c.state === "match" ? "" : " accent"}`} role={c.state === "mismatch" ? "alert" : "status"} data-state={c.state}>
      <strong>{headline}</strong>
      <span> {c.message}</span>
    </div>
  );
}

export function WeatherBody({ doc, now }: { doc: WeatherDoc; now: number }) {
  const [roundPick, setRound] = useState<number | null>(null);
  const round = roundPick !== null && doc.rounds.includes(roundPick) ? roundPick : doc.rounds[0] ?? 1;
  const points = doc.forecast.hourly.filter((h) => h.round === round);
  const waves = doc.waves.filter((w) => w.round === round);
  const field = doc.field.find((f) => f.round === round);
  const age = forecastAgeHours(doc.forecast.issued_at, now);
  const scale = Math.max(0.2, ...doc.waves.flatMap((w) => [w.rel, w.rel_p10, w.rel_p90]).filter((v): v is number => v !== null).map(Math.abs)) * 1.1;
  const v = doc.venue;
  return (
    <div className="wx-body">
      {doc.synthetic && <p className="inputs-note accent">Example data: this forecast is invented to preview the layout and is not a real forecast.</p>}
      <Panel eyebrow="Venue" title={v.name || doc.event_uid}>
        <div className="kpi-grid">
          {v.class && <Kpi label="Course type" value={v.class.replace(/^./, (c) => c.toUpperCase())} detail="How exposed the course is to weather" hint="The course's weather profile, for example links or desert, which sets how much wind and temperature matter." />}
          {v.wind_slope !== null && <Kpi label="Wind sensitivity" value={`${v.wind_slope.toFixed(3)} strokes`} detail={`Per mph of wind, per round${windSourceText(v.wind_slope_source) ? `. ${windSourceText(v.wind_slope_source)}` : ""}`} tone="accent" hint="How many extra strokes a player loses per round for each additional mph of average wind at this course." />}
          {doc.forecast.issued_at && <Kpi label="Forecast issued" value={etTime(doc.forecast.issued_at)} detail={age === null ? undefined : age < 1 ? "Under an hour old when this page loaded" : `${age.toFixed(1)} hours old when this page loaded`} tone={age !== null && age > 12 ? "negative" : "neutral"} hint="When the weather services last published the forecast the model used." />}
          {doc.forecast.models.length > 0 && <Kpi label="Forecast sources" value={String(doc.forecast.models.length)} detail={doc.forecast.models.map(modelName).join(", ")} hint="The weather models averaged together to make this forecast." />}
        </div>
        {doc.notes.filter((n) => !(doc.synthetic && /synthetic/i.test(n))).length > 0 && <ul className="wx-notes">{doc.notes.filter((n) => !(doc.synthetic && /synthetic/i.test(n))).map((n, i) => <li key={i} className="inputs-note">{n}</li>)}</ul>}
        {doc.as_of && <p className="inputs-muted">Weather prepared {etTime(doc.as_of)}.</p>}
      </Panel>

      {doc.rounds.length > 0 && (
        <SegmentedControl label="Round" value={String(round)} onChange={(x) => setRound(Number(x))} options={doc.rounds.map((r) => ({ value: String(r), label: `R${r}` }))} />
      )}

      <Panel eyebrow={`Round ${round}`} title="Forecast by hour">
        {points.length ? <HourlyCharts points={points} waves={waves} /> : <EmptyState title="No hourly forecast for this round" detail="This round is too far away for the forecast to reach yet." />}
      </Panel>

      <Panel eyebrow={`Round ${round}`} title="Wave comparison and field effect">
        <WaveCards waves={waves} scale={scale} />
        <div className="kpi-grid wx-field">
          {field && field.common_strokes !== null && <Kpi label="Whole-field shift" value={`${signedFixed(field.common_strokes)} strokes`} detail="How much weather moves scoring for everyone" hint="The change in the average score for the whole field because of the weather. Positive means scores run higher." />}
          {field && field.cut_line_shift !== null && <Kpi label="Cut-line shift" value={`${signedFixed(field.cut_line_shift)} strokes`} detail="Expected move of the projected cut" hint="How far the weather is expected to move the projected cut score." />}
        </div>
        {field && field.cut_line_shift === null && <p className="inputs-muted">No cut-line shift for this round: the cut is only projected once round 2 is in view.</p>}
      </Panel>

      <Panel eyebrow="All players" title="Weather effect by player (rounds 1 and 2)">
        <PlayersTable doc={doc} />
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ page */
export function WeatherEffectsView() {
  const { data: index, loading } = useDashboardData<PublishIndex>("golfprice/index.json");
  const events = useMemo(() => index?.events ?? [], [index]);
  const [pick, setPick] = useState("");
  const [now] = useState(() => Date.now());
  const event = events.find((e) => e.event_uid === pick) ?? events.find((e) => e.weather_key) ?? events[0];
  const { data, loading: docLoading, error } = useDashboardData<unknown>(event ? weatherKey(event) : "golfprice/none.json");
  const doc = useMemo(() => parseWeather(data), [data]);

  const intro = (
    <PageIntro
      eyebrow="Course conditions"
      title="Weather effects"
      description="What the forecast says hour by hour, and how much it helps or hurts each player given the course and tee time."
      controls={events.length > 0 ? (
        <label className="select-control">
          <span>Event</span>
          <select value={event?.event_uid ?? ""} onChange={(e) => setPick(e.target.value)}>
            {events.map((e) => <option key={e.event_uid} value={e.event_uid}>{e.name}{e.tour ? ` (${tourWord(e.tour)})` : ""}</option>)}
          </select>
        </label>
      ) : undefined}
    />
  );
  if (loading) return <div>{intro}<LoadingState label="Loading published events" /></div>;
  if (!event) return <div>{intro}<EmptyState title="No golfprice run is published yet" detail="Weather appears here once a run has been published with a forecast." /></div>;
  if (docLoading) return <div>{intro}<LoadingState label="Loading weather" /></div>;
  if (!doc) {
    return (
      <div>
        {intro}
        <EmptyState
          title={`No weather published for ${event.name}`}
          detail={error ? "This run has no weather forecast yet. It is added once a forecast is available, so check back after the next run." : "The weather information could not be read."}
        />
      </div>
    );
  }
  const check = weatherCheckpoint(event, doc.as_of);
  return (
    <div>
      {intro}
      <CheckpointBanner event={event} docAsOf={doc.as_of} />
      <WeatherBody key={event.event_uid} doc={doc} now={now} />
      <details className="tech-details">
        <summary>Technical details</summary>
        <p className="inputs-muted">Weather run {check.weatherRun ?? "not recorded"}; price run {check.priceRun ?? "not recorded"}.</p>
      </details>
    </div>
  );
}
