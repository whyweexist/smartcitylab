export function Gauge({score}:{score:number}){
  const pct = Math.max(0, Math.min(100, score))
  const color = pct>=70 ? "#2ecc71" : pct>=45 ? "#f1c40f" : "#e74c3c"
  const r=54, circ=2*Math.PI*r, off=circ*(1-pct/100)
  return (
    <div style={{position:"relative", width:140, height:140}}>
      <svg width={140} height={140} viewBox="0 0 120 120">
        <circle cx={60} cy={60} r={r} stroke="#e6e8eb" strokeWidth={12} fill="none"/>
        <circle cx={60} cy={60} r={r} stroke={color} strokeWidth={12} fill="none"
          strokeDasharray={circ} strokeDashoffset={off} strokeLinecap="round"
          transform="rotate(-90 60 60)" style={{transition:"stroke-dashoffset 0.8s ease"}}/>
        <text x={60} y={62} textAnchor="middle" fontSize={26} fontWeight={700} fill="#2c3e50">{Math.round(pct)}</text>
        <text x={60} y={78} textAnchor="middle" fontSize={10} fill="#7f8c8d">/ 100</text>
      </svg>
    </div>
  )
}
