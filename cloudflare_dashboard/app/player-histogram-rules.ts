/** Round-outcome histogram: half-stroke bins drawn edge to edge as one continuous stepped outline. */
export const HIST_BIN=.5;
export function stepPath(bins:Array<{share:number}>,x:(v:number)=>number,y:(p:number)=>number,low=-8,step=HIST_BIN,close=true):string{
  let d=`M${x(low)} ${y(0)}`;
  bins.forEach((bn,i)=>{d+=` L${x(low+i*step)} ${y(bn.share)} L${x(low+(i+1)*step)} ${y(bn.share)}`;});
  return d+(close?` L${x(low+bins.length*step)} ${y(0)} Z`:"");
}
