/** Shot-fingerprint groups: every axis is a published shot metric and sits in the category it belongs to. */
export const FINGERPRINT_GROUPS:Record<string,string[]>={
  "Driving":["tee_distance","tee_accuracy","tee_dispersion","tee_sg","tee_great","tee_poor","tee_penalty"],
  "Approach":["app_50_100","app_100_150","app_150_200","app_200_plus","app_great","app_poor","prox_fairway","prox_rough","app_gir","app_penalty"],
  "Short game":["arg_sand","arg_rough","arg_fairway","short_scramble","short_updown","short_sand_save"],
  "Putting":["putt_0_5","putt_5_10","putt_10_20","putt_20_plus","putt_short_make","putt_lag_leave"],
  "Best & worst":["tee_great","app_great","putt_short_make","score_birdie","tee_poor","app_poor","putt_lag_leave","score_bogey_avoid","score_three_putt_avoid"],
  "Radar":["radar_club_speed","radar_ball_speed","radar_smash","radar_launch","radar_spin","radar_spin_axis_abs","radar_apex","radar_apex_range"],
};
/** Radar spokes describe ball flight, never skill: further out only means a bigger number than the PGA average. */
export const DESCRIPTIVE_GROUPS=new Set(["Radar"]);
export const AXIS_LABEL:Record<string,string>={
  tee_distance:"Distance",tee_accuracy:"Fairways hit",tee_dispersion:"Straightness",tee_sg:"Tee shot value",tee_great:"Great drives",tee_poor:"Avoids bad drives",tee_penalty:"Avoids tee penalties",
  app_50_100:"50–100 yds",app_100_150:"100–150 yds",app_150_200:"150–200 yds",app_200_plus:"200+ yds",app_great:"Great approaches",app_poor:"Avoids bad approaches",prox_fairway:"Close from fairway",prox_rough:"Close from rough",app_gir:"Greens hit",app_penalty:"Avoids approach penalties",
  arg_sand:"Bunker shots",arg_rough:"Chips from rough",arg_fairway:"Chips from fairway",short_scramble:"Scrambling",short_updown:"Up-and-downs",short_sand_save:"Sand saves",
  putt_0_5:"Under 5 ft",putt_5_10:"5–10 ft",putt_10_20:"10–20 ft",putt_20_plus:"20+ ft",putt_short_make:"Short putts holed",putt_lag_leave:"Lag putts left close",
  score_birdie:"Birdies or better",score_bogey_avoid:"Avoids bogeys",score_three_putt_avoid:"Avoids 3-putts",
  radar_club_speed:"Club speed",radar_ball_speed:"Ball speed",radar_smash:"Smash factor",radar_launch:"Launch angle",radar_spin:"Spin rate",radar_spin_axis_abs:"Curve (spin axis)",radar_apex:"Apex height",radar_apex_range:"Distance to apex",
};
/** Names for the numbers listed under the chart where the spoke label reads as the opposite (avoid-style labels). */
export const VALUE_LABEL:Record<string,string>={
  tee_poor:"Bad-drive rate",app_poor:"Bad-approach rate",tee_penalty:"Tee-shot penalty rate",app_penalty:"Approach penalty rate",
  score_bogey_avoid:"Bogey-free holes",score_three_putt_avoid:"Holes without a 3-putt",
};
export const valueLabel=(key:string):string=>VALUE_LABEL[key]??AXIS_LABEL[key]??key;
export const AXIS_HELP:Record<string,string>={
  tee_distance:"Average tee-shot distance on par 4 and 5 holes, all clubs. Longer is better.",
  tee_accuracy:"Share of par 4/5 tee shots that finish in the fairway.",
  tee_dispersion:"How far tee shots finish left or right of where the rest of the field's tee shots landed on the same hole. Smaller is better, so further out the chart is straighter.",
  tee_sg:"Average strokes gained per tee shot on par 4s and 5s. Penalty and drop records are not included.",
  tee_great:"Share of tee shots that gained at least half a stroke on the average player.",
  tee_poor:"Share of tee shots that lost at least half a stroke. Penalty and drop records are not included. Fewer is better, so further out is a lower rate.",
  tee_penalty:"Share of par 4/5 tee shots followed by a penalty stroke (water, out of bounds, lost ball, unplayable). Fewer is better, so further out is a lower rate.",
  app_50_100:"Strokes gained per approach shot from 50–100 yards.",app_100_150:"Strokes gained per approach shot from 100–150 yards.",
  app_150_200:"Strokes gained per approach shot from 150–200 yards.",app_200_plus:"Strokes gained per approach shot from 200+ yards.",
  app_great:"Share of approach shots that gained at least half a stroke.",app_poor:"Share of approach shots that lost at least half a stroke. Fewer is better.",
  prox_fairway:"Average distance from the hole after approach shots played from the fairway. Closer is better.",
  prox_rough:"Average distance from the hole after approach shots played from the rough. Closer is better.",
  app_gir:"Share of holes where the ball was on the green in regulation (two shots on a par 4). Depends on the courses and holes played.",
  app_penalty:"Share of approach shots followed by a penalty stroke, such as water. Fewer is better, so further out is a lower rate.",
  arg_sand:"Strokes gained per greenside bunker shot.",arg_rough:"Strokes gained per chip or pitch from the rough around the green.",arg_fairway:"Strokes gained per chip or pitch from the fairway around the green.",
  short_scramble:"Share of holes where the green was missed in regulation but the player still made par or better.",
  short_updown:"Share of holes with a chip, pitch or bunker shot around the green where the player holed out within two shots (that shot plus one putt, or holed).",
  short_sand_save:"Share of holes with a greenside bunker shot where the player holed out within two shots.",
  putt_0_5:"Strokes gained per putt from under 5 feet.",putt_5_10:"Strokes gained per putt from 5–10 feet.",putt_10_20:"Strokes gained per putt from 10–20 feet.",putt_20_plus:"Strokes gained per putt from 20+ feet.",
  putt_short_make:"Share of putts from under 5 feet that were holed.",putt_lag_leave:"Average distance left after putts from 20+ feet. Smaller is better, a proxy for avoiding three-putts.",
  score_birdie:"Share of holes finished a stroke or more under par. Depends on the courses and holes played.",
  score_bogey_avoid:"Share of holes finished at par or better, so bogey or worse is avoided.",
  score_three_putt_avoid:"Share of holes with at least one putt that took one or two putts, so no three-putt.",
  radar_club_speed:"Average club speed on par 4/5 tee shots, where the launch monitor recorded it. Describes the swing, not skill.",
  radar_ball_speed:"Average ball speed off the club on par 4/5 tee shots. Describes the shot, not skill.",
  radar_smash:"Ball speed divided by club speed: how efficiently the strike transfers speed. Only where club speed was recorded.",
  radar_launch:"Average vertical launch angle, in degrees above horizontal.",
  radar_spin:"Average spin rate at launch, in revolutions per minute.",
  radar_spin_axis_abs:"Average size of the spin axis tilt in degrees, ignoring direction. A bigger number means more curve; left or right is not shown.",
  radar_apex:"Average peak height of the ball flight, in feet.",
  radar_apex_range:"Average distance from the tee to the highest point of the flight, in feet. Carry and flight time are not published.",
};
/** What each group cannot show yet (shown as a note, never drawn). */
export const GROUP_MISSING:Record<string,string>={
  "Driving":"which side of the fairway tee shots miss on (the published shot data has no verified left/right convention).",
  "Approach":"",
  "Short game":"",
  "Putting":"",
  "Best & worst":"",
  "Radar":"carry distance and flight time (the radar data has no trajectories and its flight-time units are unverified).",
};
/** Measures whose sample is whole holes rather than shots. */
export const HOLE_BASED=new Set(["app_gir","short_scramble","short_updown","short_sand_save","score_birdie","score_bogey_avoid","score_three_putt_avoid"]);
export const sampleNoun=(key:string):string=>HOLE_BASED.has(key)?"holes":"shots";
/** Plain-language names of the strongest spokes (no raw metric keys or published jargon). */
export function standoutText(standouts:Array<{key:string}>):string{return standouts.map(m=>(AXIS_LABEL[m.key]??m.key).toLowerCase()).join(" · ");}
type RadarMetric={key:string;n_shots?:number;n_rounds?:number;n_events?:number};
/** One honest coverage line, computed from the (possibly filtered) metrics so it follows the selected rounds. */
export function radarCoverageLine(metrics:RadarMetric[]):string{
  const radar=metrics.find(m=>m.key==="radar_ball_speed"),tee=metrics.find(m=>m.key==="tee_distance");
  const n=radar?.n_shots??0,total=tee?.n_shots??0;
  if(!n)return "No radar was recorded in these rounds. Radar only exists at some PGA events, from 2023.";
  const share=total>0?` of ${total.toLocaleString("en-US")} par 4/5 tee shots (${Math.round(Math.min(1,n/total)*100)}%)`:"";
  const ev=radar?.n_events?`, across ${radar.n_events} event${radar.n_events===1?"":"s"}`:"";
  return `Radar was recorded on ${n.toLocaleString("en-US")} tee shots${share}${ev}. It is measured at only some events, so this is not the whole picture.`;
}
export type AxisPlacement={x:number;y:number;anchor:"start"|"middle"|"end"};
export function axisAngle(i:number,n:number):number{return (i*360/n-90)*Math.PI/180;}
export function labelPlacement(i:number,n:number,cx:number,cy:number,radius:number):AxisPlacement{
  const a=axisAngle(i,n),c=Math.cos(a),s=Math.sin(a);
  return {x:cx+c*radius,y:cy+s*radius+(s>.3?9:s<-.3?-2:4),anchor:c>.3?"start":c<-.3?"end":"middle"};
}
