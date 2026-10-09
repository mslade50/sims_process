/** Shot-fingerprint groups: every axis is a published shot metric and sits in the category it belongs to. */
export const FINGERPRINT_GROUPS:Record<string,string[]>={
  "Driving":["tee_distance","tee_accuracy","tee_dispersion","tee_sg","tee_great","tee_poor"],
  "Approach":["app_50_100","app_100_150","app_150_200","app_200_plus","app_great","app_poor","prox_fairway","prox_rough"],
  "Short game":["arg_sand","arg_rough","arg_fairway"],
  "Putting":["putt_0_5","putt_5_10","putt_10_20","putt_20_plus","putt_short_make","putt_lag_leave"],
  "Best & worst shots":["tee_great","app_great","putt_short_make","tee_poor","app_poor","putt_lag_leave"],
};
export const AXIS_LABEL:Record<string,string>={
  tee_distance:"Distance",tee_accuracy:"Fairways hit",tee_dispersion:"Straightness",tee_sg:"Tee shot value",tee_great:"Great drives",tee_poor:"Avoids bad drives",
  app_50_100:"50–100 yds",app_100_150:"100–150 yds",app_150_200:"150–200 yds",app_200_plus:"200+ yds",app_great:"Great approaches",app_poor:"Avoids bad approaches",prox_fairway:"Close from fairway",prox_rough:"Close from rough",
  arg_sand:"Bunker shots",arg_rough:"Chips from rough",arg_fairway:"Chips from fairway",
  putt_0_5:"Under 5 ft",putt_5_10:"5–10 ft",putt_10_20:"10–20 ft",putt_20_plus:"20+ ft",putt_short_make:"Short putts holed",putt_lag_leave:"Lag putts left close",
};
export const AXIS_HELP:Record<string,string>={
  tee_distance:"Average tee-shot distance on par 4 and 5 holes, all clubs. Longer is better.",
  tee_accuracy:"Share of par 4/5 tee shots that finish in the fairway.",
  tee_dispersion:"How far tee shots finish left or right of where the rest of the field's tee shots landed on the same hole. Smaller is better, so further out the chart is straighter.",
  tee_sg:"Average strokes gained per tee shot on par 4s and 5s. Penalty and drop records are not included.",
  tee_great:"Share of tee shots that gained at least half a stroke on the average player.",
  tee_poor:"Share of tee shots that lost at least half a stroke. Penalty and drop records are not included. Fewer is better, so further out is a lower rate.",
  app_50_100:"Strokes gained per approach shot from 50–100 yards.",app_100_150:"Strokes gained per approach shot from 100–150 yards.",
  app_150_200:"Strokes gained per approach shot from 150–200 yards.",app_200_plus:"Strokes gained per approach shot from 200+ yards.",
  app_great:"Share of approach shots that gained at least half a stroke.",app_poor:"Share of approach shots that lost at least half a stroke. Fewer is better.",
  prox_fairway:"Average distance from the hole after approach shots played from the fairway. Closer is better.",
  prox_rough:"Average distance from the hole after approach shots played from the rough. Closer is better.",
  arg_sand:"Strokes gained per greenside bunker shot.",arg_rough:"Strokes gained per chip or pitch from the rough around the green.",arg_fairway:"Strokes gained per chip or pitch from the fairway around the green.",
  putt_0_5:"Strokes gained per putt from under 5 feet.",putt_5_10:"Strokes gained per putt from 5–10 feet.",putt_10_20:"Strokes gained per putt from 10–20 feet.",putt_20_plus:"Strokes gained per putt from 20+ feet.",
  putt_short_make:"Share of putts from under 5 feet that were holed.",putt_lag_leave:"Average distance left after putts from 20+ feet. Smaller is better, a proxy for avoiding three-putts.",
};
/** Stats the shot export does not publish yet (shown as a note, never drawn). */
export const GROUP_MISSING:Record<string,string>={
  "Driving":"Penalty and out-of-bounds rate off the tee is not published yet.",
  "Approach":"Greens in regulation and approach penalty rate are not published yet.",
  "Short game":"Up-and-down (scrambling) rate is not published yet.",
  "Putting":"",
  "Best & worst shots":"Scrambling, bogey avoidance and birdie conversion need hole-level scoring that is not published yet.",
};
export type AxisPlacement={x:number;y:number;anchor:"start"|"middle"|"end"};
export function axisAngle(i:number,n:number):number{return (i*360/n-90)*Math.PI/180;}
export function labelPlacement(i:number,n:number,cx:number,cy:number,radius:number):AxisPlacement{
  const a=axisAngle(i,n),c=Math.cos(a),s=Math.sin(a);
  return {x:cx+c*radius,y:cy+s*radius+(s>.3?9:s<-.3?-2:4),anchor:c>.3?"start":c<-.3?"end":"middle"};
}
