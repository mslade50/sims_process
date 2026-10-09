/** Published historical profiles. Null metrics remain unknown, never population average. */
export const PROFILE_AXES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"] as const;
export type ProfileAxis = typeof PROFILE_AXES[number];
export type Photo = { url: string | null; source: string | null; source_name?:string; status?: string };
export type PgaBenchmark = {schema_version:string;status:string;value:number|null;shrunk_value?:number|null;shrink_stale?:boolean|null;shrink_weight_on_data?:number|null;shrink_prior_tour?:string|null;observed_mean?:number|null;raw_adjusted_sg_mean?:number|null;decomposition?:{tour_weights?:Array<{tour:string;n_rounds:number;weight_fraction:number}>;historical_mean_interval?:{lower:number;upper:number;level?:number;note?:string}};n_rounds:number;n_events:number;n_eff:number;effective_sample_size?:number;weight_on_data:number;last_observation:string|null;as_of:string;reference_id:string|null;unit:string;note?:string;method:{window_days:number;half_life_days:number;prior_rounds:number;min_rounds:number;min_events:number;min_n_eff:number}};
export type PgaBenchmarkReference = {id:string;status:string;label:string;population:string;year:number;rounds:number[];adjusted_sg_mean:number|null;n_rounds:number;n_players:number;n_events:number;available_after?:string;method?:unknown;regular_tick?:{unshrunk?:number|null;shrunk?:number|null;as_of?:string}};
export type ProfileEntry = {pga_benchmark?:PgaBenchmark; profile_id?: string; dg_id: number | null; name: string; aliases?: string[]; tours?: string[]; country?: string; country_code?: string; amateur?: boolean; last_observation?: string; n_rounds?: number; status: string; available_axes?: number | string[]; photo?: Photo; profile_key: string; search?: string };
export const playerId = (p: ProfileEntry): string => p.profile_id ?? String(p.dg_id);
export type Reference = { id: string; population: string; n_players: number; min_rounds: number; mean: number | null; sd: number | null; as_of: string; higher_is_better: boolean };
export type Metric = { key: ProfileAxis; label: string; unit: string; value: number | null; estimate: number | null; z: number | null; n: number | null; n_eff: number | null; reference: string; shrinkage: { prior_rounds: number; weight_on_data: number } | null; last_observation: string | null; status: string };
export type HistoricalTrait = {value?:number;raw_value?:number;unit?:string;percentile?:number|null;band?:string;interval_90?:number[]|null;shrinkage_weight?:number;reference_players?:number;label_status?:string;band_uncertain?:boolean};
export type Archetype = {label?:string;description?:string;status?:string;reason?:string;caveat?:string;notes?:string[];sample?:{rounds?:number;events?:number;round_selection?:string};reliability?:{status?:string;evidence?:string;reason?:string;cut_maker?:unknown};volatility?:HistoricalTrait|null;relative_hot_round_upside?:HistoricalTrait|null;tail_balance?:HistoricalTrait|null};
export type PlayerProfile = { deep_key?:string; schema_version: string; as_of: string; identity: ProfileEntry; coverage: {status: string}; photo?: Photo; radar: Metric[]; references: Record<string, Reference>; pga_benchmark?:PgaBenchmark; pga_benchmark_reference?:PgaBenchmarkReference; archetype?: Archetype | string | null; provenance?: unknown };
export type ProfileCatalog = { schema_version: string; as_of: string; players: ProfileEntry[]; references: Record<string, Reference>; pga_benchmark_reference?:PgaBenchmarkReference; provenance?: unknown; catalog_key?: string };
export const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
export const signedZ = (v: unknown, digits = 2): string => finite(v) ? `${v > 0 ? "+" : ""}${v.toFixed(digits)}` : "—";
export function profileKey(key: string): string {
  if (key.startsWith("golfprice/player_profiles/") && !key.includes("..")) return key;
  if (/^(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+\.json$/.test(key)) return `golfprice/player_profiles/${key}`;
  return "golfprice/player_profiles/unavailable.json";
}
export function parseCatalog(raw: unknown): ProfileCatalog | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as ProfileCatalog;
  return d.schema_version === "player_profiles.v1" && Array.isArray(d.players) ? {...d, players: d.players.filter(p => ((Number.isSafeInteger(p.dg_id) && p.dg_id! > 0) || (p.dg_id === null && /^historical-[a-z0-9]+$/.test(p.profile_id ?? ""))) && typeof p.name === "string" && typeof p.profile_key === "string")} : null;
}
export function parseProfile(raw: unknown, id: string | number): PlayerProfile | null {
  if (!raw || typeof raw !== "object") return null;
  const batch = raw as {schema_version?:string;profiles?:Record<string,unknown>};
  if(batch.schema_version === "player_profiles.batch.v1") return parseProfile(batch.profiles?.[String(id)],id);
  const d = raw as PlayerProfile;
  return d.schema_version === "player_profiles.v1" && d.identity && playerId(d.identity) === String(id) && Array.isArray(d.radar) ? d : null;
}
export function profileMatches(p: ProfileEntry, query: string): boolean {
  const fold = (s: string) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  const haystack = fold([p.name, ...(p.aliases ?? []), p.country, ...(p.tours ?? []), String(p.dg_id)].join(" "));
  return fold(query).split(/\s+/).every(word => haystack.includes(word));
}
/** Coverage first, then recent observation and alphabetic identity; do not mutate source order. */
export function sortProfiles(players:ProfileEntry[]):ProfileEntry[] {
  const rank=(p:ProfileEntry)=>p.status==="complete"?0:p.status==="partial"?1:2;
  const time=(p:ProfileEntry)=>p.last_observation && Number.isFinite(Date.parse(p.last_observation)) ? Date.parse(p.last_observation) : -Infinity;
  return [...players].sort((a,b)=>rank(a)-rank(b) || (time(a)===time(b)?0:time(b)-time(a)) || a.name.localeCompare(b.name) || playerId(a).localeCompare(playerId(b)));
}
/** Calendar dates ("2026-08-30") print as plain dates; full timestamps read in US Eastern time ("Oct 8, 12:10 PM ET"). */
export function shortProfileDate(value:string|undefined):string {
  if(!value || !Number.isFinite(Date.parse(value))) return "Unavailable";
  if(/^\d{4}-\d{2}-\d{2}$/.test(value)) return new Date(`${value}T00:00:00Z`).toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric",timeZone:"UTC"});
  return `${new Intl.DateTimeFormat("en-US",{timeZone:"America/New_York",month:"short",day:"numeric",hour:"numeric",minute:"2-digit"}).format(new Date(value))} ET`;
}
export function selectWeeklyEvent<T extends {event_uid:string;tour?:string;date_start?:string;player_ids?:number[]}>(events:T[],explicit:string|null,player:string|null,fields:Array<{event_uid:string;player_ids:number[]}> = []):T|undefined {
  const requested=events.find(e=>e.event_uid===explicit); if(requested) return requested;
  const recent=[...events].sort((a,b)=>(b.date_start ?? "").localeCompare(a.date_start ?? ""));
  if(!explicit && player && /^\d+$/.test(player)) {const containing=recent.find(e=>(e.player_ids ?? fields.find(f=>f.event_uid===e.event_uid)?.player_ids ?? []).includes(Number(player))); if(containing) return containing;}
  return recent.find(e=>e.tour?.toLowerCase()==="pga") ?? recent[0];
}
export function profileHref(id: string | number, source = "/players"): string {
  const url = new URL(source, "https://dashboard.invalid");
  url.pathname = "/players"; url.searchParams.set("player", String(id)); url.searchParams.delete("vs");
  return `${url.pathname}${url.search}${url.hash}`;
}
/** Offset signed Z into a positive radius; zero is the middle ring. Shared extent for overlays. */
export function radarExtent(profiles: PlayerProfile[]): number {
  return Math.max(3, ...profiles.flatMap(p => p.radar.map(m => finite(m.z) ? Math.ceil(Math.abs(m.z)) : 0)));
}
export function radarRadius(z: number, extent: number): number { return (z + extent) / (2 * extent); }
export function completeRadar(p: PlayerProfile): boolean { return PROFILE_AXES.every(key => finite(p.radar.find(m => m.key === key)?.z)); }
export function sameReference(a: PlayerProfile, b: PlayerProfile): boolean {
  return PROFILE_AXES.every(key => {const x = a.radar.find(m => m.key === key); const y = b.radar.find(m => m.key === key); return !!x?.reference && x.reference === y?.reference && JSON.stringify(a.references?.[key]) === JSON.stringify(b.references?.[key]);});
}
export function reconciliation(base: number | null, final: number | null, components: Record<string, number>): number | null {
  return finite(base) && finite(final) && Object.values(components).every(finite) ? final - base - Object.values(components).reduce((a,b) => a+b,0) : null;
}
export function savedSkill(row: unknown): { base: number | null; final: number | null; sum: number | null; residual: number | null; families: Record<string, unknown>; weather: Record<string, unknown> } {
  const obj = (v: unknown): Record<string, unknown> => v && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : {};
  const ch = obj(obj(row).challenger), bd = obj(ch.breakdown), hist = obj(obj(row).history);
  const value = (v: unknown) => finite(v) ? v : null;
  const final = value(ch.mu), sum = value(bd.sum_of_components);
  return {base:value(hist.chl_mu_untouched_by_location) ?? value(bd.chl_total),final,sum,residual:final !== null && sum !== null ? final-sum : null,families:obj(bd.chl_families),weather:obj(ch.weather)};
}
export type SavedCheckpoint = {event_uid:string;kind:string;run:string;as_of:string|null};
const record = (v:unknown):Record<string,unknown> => v && typeof v === "object" && !Array.isArray(v) ? v as Record<string,unknown> : {};
const sameTime = (a:unknown,b:unknown):boolean => typeof a === "string" && typeof b === "string" && Number.isFinite(Date.parse(a)) && Date.parse(a) === Date.parse(b);
export function matchesExplainCheckpoint(raw:unknown, expected:SavedCheckpoint|null):boolean {
  const d=record(raw);return !!expected && d.event_uid===expected.event_uid && d.kind===expected.kind && d.run===expected.run && sameTime(d.as_of,expected.as_of);
}
export function matchesInputCheckpoint(raw:unknown, expected:SavedCheckpoint|null):boolean {
  const d=record(raw),live=record(d.live),prov=record(d.provenance);
  // live.run_key is a content hash, while the published run is the official folder name.
  const liveStamp=typeof live.as_of === "string" && Number.isFinite(Date.parse(live.as_of)) ? new Date(live.as_of).toISOString().replace(/[-:]/g,"").replace(/\.\d{3}Z$/,"Z") : null;
  const liveRun=Number.isInteger(live.after_round) && liveStamp ? `R${live.after_round}${live.mid_round ? "m" : ""}_${liveStamp}` : null;
  const run=expected?.kind === "live" ? liveRun : typeof prov.run_dir === "string" ? prov.run_dir.replace(/\\/g,"/").split("/").at(-1) : null;
  const asof=expected?.kind === "live" ? live.as_of : d.generated_for_as_of;
  return !!expected && d.schema === "golfprice.model_inputs.v1" && record(d.event).event_uid===expected.event_uid && d.kind===expected.kind && run===expected.run && sameTime(asof,expected.as_of);
}
export function matchingInputRun(event:{event_uid:string;explain_run?:string|null;runs?:Array<{kind:string;run:string;as_of?:string|null;key:string}>},selected?:{kind:string;run:string;as_of:string|null}):({key:string}&SavedCheckpoint)|null {
  const row=event.runs?.find(r=>selected ? r.kind===selected.kind && r.run===selected.run && sameTime(r.as_of,selected.as_of) : `${r.kind}_${r.run}`===event.explain_run);
  return row && row.as_of ? {...row,as_of:row.as_of,event_uid:event.event_uid} : null;
}

/** Saved model skill on the PGA scale (challenger.mu_tour = mu + field_offset) for one player, read from a checkpoint-matched model_inputs document. Live checkpoints use mu_live + the pre-event field offset (no active-field recompute yet); the pre-event value is kept. Null when the document lacks the needed numbers, so callers fall back to the field-centred display. */
export type PgaSkill = {value:number;offset:number|null;live:boolean;preEvent:number|null};
export function savedPgaSkill(inputs:unknown,id:number|null):PgaSkill|null {
  const d=record(inputs);if(id===null || !Array.isArray(d.players)) return null;
  const row=d.players.map(record).find(p=>p.dg_id===id);if(!row) return null;
  const tour=record(row.challenger).mu_tour, offsetRaw=record(record(d.event).field_strength).field_offset;
  const offset=finite(offsetRaw) ? offsetRaw : null, preEvent=finite(tour) ? tour : null;
  if(d.kind==="live"){const mu=record(row.live).mu_live;return finite(mu) && offset!==null ? {value:mu+offset,offset,live:true,preEvent} : null;}
  return preEvent!==null ? {value:preEvent,offset,live:false,preEvent} : null;
}
const offsetText=(v:number|null)=>v===null ? "unavailable" : `${v>0 ? "+" : ""}${v.toFixed(3)}`;
/** One wording for the secondary card and the hero chip. */
export function pgaSkillWords(s:PgaSkill):{sub:string;title:string} {
  return s.live
    ? {sub:`Live skill plus the pre-event field offset (${offsetText(s.offset)}, how this field compares with a normal PGA field)`,title:`Live skill plus the pre-event field offset. The pre-event value was ${offsetText(s.preEvent)}. The offset has not been recomputed for the players still active.`}
    : {sub:`The model's saved skill plus this field's offset to the PGA scale (${offsetText(s.offset)}, how this field compares with a normal PGA field)`,title:`The model's saved skill plus this field's offset to the PGA scale (${offsetText(s.offset)}).`};
}

/** The pre-shrinkage value, kept for the Method disclosure. Only supported, explicitly referenced published ratings count. */
export function unshrunkValue(rating:PgaBenchmark|undefined,reference:PgaBenchmarkReference|undefined):number|null {
  return rating?.schema_version === "player_benchmark.v1" && rating.status === "available" && reference?.status === "available" && rating.reference_id === reference.id && finite(rating.value) ? rating.value : null;
}
/** The headline: the shrunk value when published, otherwise the unshrunk one (see isShrunk). */
export function benchmarkValue(rating:PgaBenchmark|undefined,reference:PgaBenchmarkReference|undefined):number|null {
  const raw=unshrunkValue(rating,reference);
  return raw!==null && finite(rating?.shrunk_value) ? rating.shrunk_value : raw;
}
/** True when the headline is the shrunk value; false means the publisher has not yet sent one (small "not yet shrunk" note). */
export function isShrunk(rating:PgaBenchmark|undefined,reference:PgaBenchmarkReference|undefined):boolean {
  return unshrunkValue(rating,reference)!==null && finite(rating?.shrunk_value);
}
export const NOT_YET_SHRUNK = "Raw average; not adjusted for small samples.";
export function fieldBenchmark(ids:number[],players:ProfileEntry[],reference:PgaBenchmarkReference|undefined,checkpoint?:string|null):{mean:number|null;covered:number;total:number} {
  const unique=[...new Set(ids)];const byId=new Map(players.filter(p=>p.dg_id!==null).map(p=>[p.dg_id,p]));
  const values=unique.map(id=>checkpointBenchmarkValue(byId.get(id)?.pga_benchmark,reference,checkpoint)).filter((v):v is number=>v!==null);
  // Full coverage is required: a partial-field average can systematically omit weaker players.
  return {mean:unique.length>0 && values.length===unique.length ? values.reduce((a,b)=>a+b,0)/values.length : null,covered:values.length,total:unique.length};
}
export function relativeBenchmark(value:number|null,field:{mean:number|null}):number|null {return finite(value) && finite(field.mean) ? value-field.mean : null;}

/** Current history cannot be displayed as known before its last observation or fixed reference. */
export function checkpointBenchmarkValue(rating:PgaBenchmark|undefined,reference:PgaBenchmarkReference|undefined,checkpoint?:string|null):number|null {
  const value=benchmarkValue(rating,reference);if(!checkpoint || value===null) return value;
  const cutoff=Date.parse(checkpoint), last=Date.parse(rating?.last_observation ?? ""), anchor=Date.parse(reference?.available_after ?? "");
  return Number.isFinite(cutoff) && Number.isFinite(last) && Number.isFinite(anchor) && last+86400000<cutoff && anchor<cutoff ? value : null;
}

/** One saved estimator for the entire active field; never substitute pre-event values into live gaps. */
export function savedFieldSkill(doc:{kind:string;players:Array<{id:number;withdrawn?:boolean;mu:number|null;live?:{mu_live:number|null}|null}>}|null,id:number|null):{value:number|null;covered:number;total:number} {
  const active=doc?.players.filter(p=>!p.withdrawn) ?? [];const selected=active.find(p=>p.id===id);
  const metric=(p:typeof active[number])=>doc?.kind==="live" ? p.live?.mu_live : p.mu;
  const values=active.map(metric).filter(finite);const value=selected ? metric(selected) : null;
  return {value:values.length===active.length && values.length>0 && finite(value) ? value-values.reduce((a,b)=>a+b,0)/values.length : null,covered:values.length,total:active.length};
}
