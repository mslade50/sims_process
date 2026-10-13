import {finite, type HistoryEvent} from "./player-deep-rules.ts";

/** Event-by-event table: columns, color scale, shot aggregation and sorting (pure, tested). */
export type ShotRoundRow={event_id:string;round:number;date?:string;metrics:Record<string,{total:number;n_shots:number}|undefined>};
/** mean/sd: PGA golfers' multi-year averages (spread between golfers). eventMean/eventSd: PGA golfers' single-event results (what one table row is), preferred for shading when published. */
export type ShotRef=Record<string,{mean?:number|null;sd?:number|null;higher?:boolean;eventMean?:number|null;eventSd?:number|null}>;
export function shotReferenceOf(metrics?:Array<{key:string;reference_mean?:number|null;reference_sd?:number|null;higher_is_better?:boolean;event_reference_mean?:number|null;event_reference_sd?:number|null}>):ShotRef{
  const out:ShotRef={};for(const m of metrics??[]){
    out[m.key]={mean:m.reference_mean,sd:m.reference_sd,higher:m.higher_is_better};
    if(finite(m.event_reference_mean)&&finite(m.event_reference_sd)){out[m.key].eventMean=m.event_reference_mean;out[m.key].eventSd=m.event_reference_sd;}
  }
  return out;
}
export type SgKey="sg_total"|"sg_ott"|"sg_app"|"sg_arg"|"sg_putt"|"t2g";
export const SG_COLUMNS:Array<{key:SgKey;label:string;help:string;range:number}>=[
  {key:"sg_total",label:"Total",range:3,help:"Total strokes gained per round against the PGA average round (0 = average), adjusted for field strength; the four categories use the same baseline. Blank when the source has no adjusted total (mostly feeder tours)."},
  {key:"sg_ott",label:"OTT",range:1.5,help:"Off the tee: strokes gained per round on tee shots against the PGA average round (2017–2025), same baseline as Total. Tour events without that baseline show the week's-field figure in italics."},
  {key:"sg_app",label:"APP",range:1.5,help:"Approach: strokes gained per round on approach shots against the PGA average round (2017–2025). Italic = compared with that week's field instead (older or non-PGA data)."},
  {key:"sg_arg",label:"ARG",range:1,help:"Around the green: strokes gained per round on chips, pitches and bunker shots against the PGA average round (2017–2025). Italic = compared with that week's field instead."},
  {key:"sg_putt",label:"PUTT",range:1.5,help:"Putting: strokes gained per round on the greens against the PGA average round (2017–2025). Italic = compared with that week's field instead."},
  {key:"t2g",label:"T2G",range:2.5,help:"Tee to green: off the tee + approach + around the green, per round, against the PGA average round (2017–2025). Italic = compared with that week's field instead."},
];
export type ShotKey="tee_distance"|"tee_accuracy"|"app_great"|"app_poor"|"prox_fairway"|"putt_short_make"|"app_gir"|"short_scramble"|"score_birdie"|"tee_penalty"|"radar_club_speed"|"radar_ball_speed"|"radar_smash"|"radar_launch"|"radar_spin"|"radar_apex";
export type ShotUnit="yd"|"pct"|"ft"|"mph"|"deg"|"rpm"|"ratio";
/** descriptive = launch-monitor ball flight: shown plainly, never shaded as good or bad. */
export const SHOT_COLUMNS:Array<{key:ShotKey;label:string;unit:ShotUnit;help:string;min:number;descriptive?:boolean}>=[
  {key:"tee_distance",label:"Drive",unit:"yd",min:8,help:"Average tee-shot distance on par 4 and 5 holes (all clubs), yards."},
  {key:"tee_accuracy",label:"Fwy %",unit:"pct",min:8,help:"Share of par 4/5 tee shots that finished in the fairway."},
  {key:"prox_fairway",label:"Prox",unit:"ft",min:5,help:"Average distance from the hole after approach shots played from the fairway, in feet. Lower is better."},
  {key:"app_great",label:"Great app",unit:"pct",min:8,help:"Share of approach shots that gained at least half a stroke on the average player."},
  {key:"app_poor",label:"Poor app",unit:"pct",min:8,help:"Share of approach shots that lost at least half a stroke. Lower is better."},
  {key:"putt_short_make",label:"Putts <5 ft",unit:"pct",min:8,help:"Share of putts from inside 5 feet that were holed."},
  {key:"app_gir",label:"GIR",unit:"pct",min:18,help:"Greens in regulation: share of holes with the ball on the green in regulation. Depends on the course and holes played."},
  {key:"short_scramble",label:"Scr %",unit:"pct",min:6,help:"Scrambling: share of missed greens where par or better was still made."},
  {key:"score_birdie",label:"Birdie+",unit:"pct",min:18,help:"Share of holes finished under par."},
  {key:"tee_penalty",label:"Tee pen",unit:"pct",min:8,help:"Share of par 4/5 tee shots followed by a penalty stroke (water, out of bounds, lost ball). Lower is better."},
  {key:"radar_club_speed",label:"Club mph",unit:"mph",min:5,descriptive:true,help:"Average club speed on par 4/5 tee shots where the launch monitor recorded it. Describes the swing, not skill."},
  {key:"radar_ball_speed",label:"Ball mph",unit:"mph",min:5,descriptive:true,help:"Average ball speed on par 4/5 tee shots at events with radar. Describes the shot, not skill."},
  {key:"radar_smash",label:"Smash",unit:"ratio",min:5,descriptive:true,help:"Average ball speed divided by club speed, where club speed was recorded."},
  {key:"radar_launch",label:"Launch°",unit:"deg",min:5,descriptive:true,help:"Average vertical launch angle in degrees on par 4/5 tee shots."},
  {key:"radar_spin",label:"Spin rpm",unit:"rpm",min:5,descriptive:true,help:"Average launch spin in revolutions per minute on par 4/5 tee shots."},
  {key:"radar_apex",label:"Apex ft",unit:"ft",min:5,descriptive:true,help:"Average peak height of the flight in feet on par 4/5 tee shots."},
];
export type EventRow={id:string;date:string;name:string;course?:string;tour:string;major:boolean;finish?:string;rounds:number;score_to_par:number|null;
  sg:Record<SgKey,number|null>;/** Per category: "pga" = against the fixed PGA average round (same as Total); "field" = fallback, relative to that week's field. */sgBasis:Partial<Record<SgKey,"pga"|"field">>;shots:Record<ShotKey,{value:number|null;n:number;z:number|null;avg:number|null;basis:"event"|"player"|null}>;event:HistoryEvent};
const sum3=(a:number|null|undefined,b:number|null|undefined,c:number|null|undefined)=>finite(a)&&finite(b)&&finite(c)?a+b+c:null;
type CatSource=Partial<Record<"sg_ott"|"sg_app"|"sg_arg"|"sg_putt"|"sg_t2g"|"sg_ott_pga"|"sg_app_pga"|"sg_arg_pga"|"sg_putt_pga"|"sg_t2g_pga",number|null>>;
/** A category value on the PGA-average baseline when published, else the field-relative number (basis says which). T2G falls back to OTT+APP+ARG. */
export function categoryValue(src:CatSource,key:"sg_ott"|"sg_app"|"sg_arg"|"sg_putt"|"t2g"):{value:number|null;basis:"pga"|"field"|null}{
  if(key==="t2g"){const p=src.sg_t2g_pga;if(finite(p))return {value:p,basis:"pga"};
    const f=sum3(src.sg_ott,src.sg_app,src.sg_arg);return {value:f,basis:f===null?null:"field"};}
  const p=src[`${key}_pga` as const];if(finite(p))return {value:p,basis:"pga"};
  const f=src[key];return finite(f)?{value:f,basis:"field"}:{value:null,basis:null};
}
export function aggregateShots(eventId:string,rounds:number[],rows:ShotRoundRow[]|undefined,ref:ShotRef):EventRow["shots"]{
  const want=new Set(rounds),mine=(rows??[]).filter(r=>r.event_id===eventId&&want.has(r.round));
  const out={} as EventRow["shots"];
  for(const c of SHOT_COLUMNS){
    let total=0,n=0;for(const r of mine){const m=r.metrics[c.key];if(m&&finite(m.total)&&finite(m.n_shots)&&m.n_shots>0){total+=m.total;n+=m.n_shots;}}
    const value=n>=c.min?total/n:null,x=ref[c.key];
    // A table row is one event, so it is shaded against PGA golfers' single-event results; the multi-year spread is only a fallback.
    const useEvent=finite(x?.eventMean)&&finite(x?.eventSd)&&x.eventSd>0,mean=useEvent?x?.eventMean:x?.mean,sd=useEvent?x?.eventSd:x?.sd;
    const z=!c.descriptive&&finite(value)&&finite(mean)&&finite(sd)&&sd>0?(value-mean)/sd*(x?.higher===false?-1:1):null;
    out[c.key]={value,n,z,avg:!c.descriptive&&finite(mean)?mean:null,basis:!c.descriptive&&finite(mean)&&finite(sd)&&sd>0?(useEvent?"event":"player"):null};
  }
  return out;
}
export function buildRows(events:HistoryEvent[],anchor:number|null|undefined,shotRows:ShotRoundRow[]|undefined,ref:ShotRef):EventRow[]{
  return events.map(e=>{
    const total=e.sg_basis==="source_adjusted"&&finite(e.sg_total)&&finite(anchor)?e.sg_total-anchor:null;
    const cat={sg_ott:categoryValue(e,"sg_ott"),sg_app:categoryValue(e,"sg_app"),sg_arg:categoryValue(e,"sg_arg"),sg_putt:categoryValue(e,"sg_putt"),t2g:categoryValue(e,"t2g")};
    const rounds=(e.round_data??[]).map(r=>r.round??r.round_num).filter(finite);
    return {id:e.id,date:e.date,name:e.name,course:e.course,tour:e.tour,major:e.major===true,finish:e.finish,rounds:e.round_data?.length??e.rounds??0,
      score_to_par:finite(e.score_to_par)?e.score_to_par:null,
      sg:{sg_total:total,sg_ott:cat.sg_ott.value,sg_app:cat.sg_app.value,sg_arg:cat.sg_arg.value,sg_putt:cat.sg_putt.value,t2g:cat.t2g.value},
      sgBasis:Object.fromEntries(Object.entries(cat).flatMap(([k,c])=>c.basis?[[k,c.basis]]:[])) as EventRow["sgBasis"],
      shots:aggregateShots(e.id,rounds,shotRows,ref),event:e};
  });
}
/** Finish text to a sortable number: "1"→1, "T5"→5, cut/withdrawn/disqualified sort last. */
export function finishRank(f:string|undefined|null):number|null{
  if(!f)return null;const m=/^T?(\d+)$/i.exec(f.trim());if(m)return Number(m[1]);
  return /^(cut|mc|wd|dq|w\/d|dnf)/i.test(f.trim())?10000:null;
}
/** Diverging background: green for good, red for bad, centered on 0, saturating at ±range. Alpha-only so it works on any theme. */
export function divergingBackground(v:number|null|undefined,range:number,max=40):string|undefined{
  if(!finite(v)||!finite(range)||range<=0)return undefined;
  const pct=Math.round(Math.min(1,Math.abs(v)/range)*max);if(pct<1)return undefined;
  return `color-mix(in srgb, var(${v>0?"--positive":"--negative"}) ${pct}%, transparent)`;
}
export type SortKey="date"|"name"|"finish"|"rounds"|"score_to_par"|SgKey|ShotKey;
export function sortValue(r:EventRow,key:SortKey):number|string|null{
  if(key==="date")return r.date;if(key==="name")return r.name.toLowerCase();if(key==="finish")return finishRank(r.finish);
  if(key==="rounds")return r.rounds;if(key==="score_to_par")return r.score_to_par;
  if(key in r.sg)return r.sg[key as SgKey];return r.shots[key as ShotKey]?.value??null;
}
export function sortRows(rows:EventRow[],key:SortKey,dir:"asc"|"desc"):EventRow[]{
  const sign=dir==="asc"?1:-1;
  return [...rows].sort((a,b)=>{const av=sortValue(a,key),bv=sortValue(b,key);
    if(av===null&&bv===null)return b.date.localeCompare(a.date);if(av===null)return 1;if(bv===null)return -1;
    const c=typeof av==="string"?av.localeCompare(bv as string):(av as number)-(bv as number);
    return c*sign||b.date.localeCompare(a.date);});
}
export function formatShot(unit:ShotUnit,v:number|null):string{if(!finite(v))return "—";if(unit==="pct")return `${Math.round(v*100)}%`;if(unit==="yd"||unit==="rpm")return `${Math.round(v)}`;if(unit==="mph"||unit==="deg")return v.toFixed(1);if(unit==="ratio")return v.toFixed(2);return `${v.toFixed(0)}′`;}
