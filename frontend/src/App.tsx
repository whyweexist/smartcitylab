import { useEffect, useState, useCallback } from 'react'
import { analyzeImage, fetchHistory, imageUrl } from './api/client'
/// <reference types="vite/client" />
import { Gauge } from './components/Gauge'
import './styles.css'

type Issue={type:string; severity:string; confidence:number; explanation:string; evidence:any}
type Analysis={id:string; quality_score:number; quality_label:string; confidence:number; issues:Issue[]; image_stats:any; image_metadata:any; model_version:string; inference_ms:number; image_url:string; timestamp?:string}

function labelClass(l:string){
  if(l==="ACCEPTABLE") return "badge-acceptable"
  if(l==="POTENTIALLY_DEFECTIVE") return "badge-defective"
  return "badge-degraded"
}

export default function App(){
  const [tab,setTab]=useState<"analyze"|"history">("analyze")
  const [file,setFile]=useState<File|null>(null)
  const [preview,setPreview]=useState<string|null>(null)
  const [loading,setLoading]=useState(false)
  const [result,setResult]=useState<Analysis|null>(null)
  const [error,setError]=useState<string|null>(null)
  const [history,setHistory]=useState<any[]>([])
  const [hLoading,setHLoading]=useState(false)

  const loadHistory=useCallback(async()=>{
    setHLoading(true)
    try{ const d=await fetchHistory(30,0); setHistory(d.items||[]) }catch(e){console.error(e)}
    setHLoading(false)
  },[])

  useEffect(()=>{ if(tab==="history") loadHistory() },[tab, loadHistory])

  const onFile=(f:File)=>{
    setFile(f); setResult(null); setError(null)
    const url=URL.createObjectURL(f); setPreview(url)
  }

  const onDrop=(e:React.DragEvent)=>{
    e.preventDefault(); (e.target as HTMLElement).classList.remove("drag")
    const f=e.dataTransfer.files?.[0]; if(f) onFile(f)
  }

  const doAnalyze=async()=>{
    if(!file) return
    setLoading(true); setError(null)
    try{
      const r=await analyzeImage(file)
      setResult(r)
      // refresh history in background
    }catch(e:any){
      let msg=e.message
      try{ const j=JSON.parse(msg); msg=j.detail||msg }catch{}
      setError(msg)
    }finally{setLoading(false)}
  }

  return (
    <div>
      <div className="topbar">
        <div style={{maxWidth:1120, margin:"0 auto", padding:"14px 18px", display:"flex", justifyContent:"space-between", alignItems:"center"}}>
          <div style={{display:"flex", gap:12, alignItems:"center"}}>
            <div style={{width:36,height:36,borderRadius:12, background:"#5b7cff", display:"grid", placeItems:"center", color:"white", fontWeight:800}}>AI</div>
            <div>
              <div style={{fontWeight:700}}>AI Image Quality & Defect Detection</div>
              <div style={{fontSize:12, color:"#7f8c8d"}}>Hybrid CV + ML · CPU inference · Explainable</div>
            </div>
          </div>
          <div style={{display:"flex", gap:8}}>
            <button className="btn" style={{background: tab==="analyze" ? "#5b7cff":undefined, color: tab==="analyze"?"white":undefined}} onClick={()=>setTab("analyze")}>Analyze</button>
            <button className="btn" style={{background: tab==="history" ? "#5b7cff":undefined, color: tab==="history"?"white":undefined}} onClick={()=>setTab("history")}>History</button>
            <a className="btn" style={{textDecoration:"none", color:"#2c3e50"}} href="/docs" target="_blank">API Docs</a>
          </div>
        </div>
      </div>

      <div style={{maxWidth:1120, margin:"24px auto", padding:"0 18px"}}>
        {tab==="analyze" && (
          <>
            <div className="grid">
              <div className="neo" style={{padding:22}}>
                <h3 style={{margin:"0 0 8px"}}>Upload image</h3>
                <p style={{margin:"0 0 14px", color:"#7f8c8d", fontSize:13}}>Supports JPG, PNG, WEBP, BMP, TIFF · Max 10 MB · Corrupt files are rejected</p>

                <div className="drop neo-inset" onDragOver={e=>{e.preventDefault(); (e.currentTarget as HTMLElement).classList.add("drag")}} onDragLeave={e=> (e.currentTarget as HTMLElement).classList.remove("drag")} onDrop={onDrop} onClick={()=>document.getElementById("fileinp")?.click()}>
                  <input id="fileinp" type="file" accept="image/*" style={{display:"none"}} onChange={e=>{ const f=e.target.files?.[0]; if(f) onFile(f)}}/>
                  <div style={{fontSize:36}}>🖼️</div>
                  <div style={{fontWeight:600, marginTop:6}}>Drag & drop or click to select</div>
                  <div style={{fontSize:12, color:"#7f8c8d"}}>Preview appears instantly</div>
                  {file && <div style={{marginTop:10, fontSize:13, fontWeight:600}}>{file.name} · {(file.size/1024).toFixed(1)} KB</div>}
                </div>

                {preview && (
                  <div style={{marginTop:16}}>
                    <div style={{fontWeight:600, marginBottom:8}}>Preview</div>
                    <img src={preview} style={{width:"100%", maxHeight:360, objectFit:"contain", borderRadius:14, background:"#fff", padding:8, boxShadow:"inset 4px 4px 8px #d1d9e6"}}/>
                  </div>
                )}

                <button className="btn btn-primary" disabled={!file || loading} onClick={doAnalyze} style={{marginTop:16, width:"100%"}}>
                  {loading ? "Analyzing…" : "Analyze Image"}
                </button>
                {error && <div style={{marginTop:12, padding:12, borderRadius:12, background:"#fdedec", color:"#c0392b", fontSize:13}}>{error}</div>}
                {loading && <div style={{marginTop:12, fontSize:13, color:"#7f8c8d"}}>Running CV features → ML inference → scoring…</div>}
              </div>

              <div>
                {!result && !loading && (
                  <div className="neo" style={{padding:22, textAlign:"center", color:"#7f8c8d"}}>
                    <div style={{fontSize:28}}>✨</div>
                    <div style={{fontWeight:600, color:"#2c3e50", marginTop:6}}>No analysis yet</div>
                    <div style={{fontSize:13}}>Select an image and click Analyze to see quality score, issues, and explanations.</div>
                    <div style={{marginTop:14, display:"grid", gap:8, textAlign:"left"}}>
                      <div className="stat">Detects: blur · underexposure · overexposure · noise · severe degradation · potential defect</div>
                      <div className="stat">Score 0–100 + label ACCEPTABLE / DEGRADED / POTENTIALLY_DEFECTIVE</div>
                      <div className="stat">Explainability via engineered CV features + RF importances</div>
                    </div>
                  </div>
                )}
                {result && (
                  <div className="neo" style={{padding:18}}>
                    <div style={{display:"flex", gap:16, alignItems:"center"}}>
                      <Gauge score={result.quality_score}/>
                      <div>
                        <div className={`badge ${labelClass(result.quality_label)}`}>{result.quality_label}</div>
                        <div style={{fontSize:28, fontWeight:800, marginTop:6}}>{result.quality_score.toFixed(1)}<span style={{fontSize:14, fontWeight:400, color:"#7f8c8d"}}> /100</span></div>
                        <div style={{fontSize:12, color:"#7f8c8d"}}>Confidence {result.confidence} · {result.inference_ms} ms · model {result.model_version}</div>
                        <div style={{fontSize:12, color:"#7f8c8d"}}>{result.image_metadata.width}×{result.image_metadata.height} · {result.image_metadata.original_filename}</div>
                      </div>
                    </div>

                    <div style={{marginTop:16}}>
                      <div style={{fontWeight:700, marginBottom:8}}>Detected issues ({result.issues.length || "none"})</div>
                      {result.issues.length===0 && <div className="stat" style={{background:"#e8f8f0"}}>No issue above threshold — image looks acceptable.</div>}
                      <div style={{display:"grid", gap:10}}>
                        {result.issues.map((iss, i)=>(
                          <div key={i} className="issue">
                            <div style={{width:8, height:8, borderRadius:999, marginTop:6, background: iss.severity==="high" ? "#e74c3c" : iss.severity==="medium" ? "#f39c12" : "#2ecc71"}}/>
                            <div style={{flex:1}}>
                              <div style={{display:"flex", gap:8, alignItems:"center", flexWrap:"wrap"}}>
                                <strong style={{textTransform:"capitalize"}}>{iss.type.replace("_"," ")}</strong>
                                <span className="badge" style={{background:"#eef2f7", fontSize:11}}>{iss.severity}</span>
                                <span style={{fontSize:12, color:"#7f8c8d"}}>conf {iss.confidence}</span>
                              </div>
                              <div style={{fontSize:13, color:"#34495e", marginTop:4}}>{iss.explanation}</div>
                              {iss.evidence && Object.keys(iss.evidence).length>0 && (
                                <div style={{fontSize:11, color:"#7f8c8d", marginTop:6, display:"flex", gap:8, flexWrap:"wrap"}}>
                                  {Object.entries(iss.evidence).map(([k,v])=> <span key={k} style={{background:"#f8fafc", padding:"4px 8px", borderRadius:8}}>{k}: {String(v)}</span>)}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{marginTop:16}}>
                      <div style={{fontWeight:600, marginBottom:6}}>Image statistics</div>
                      <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:8}}>
                        {["laplacian_var","brightness_mean","brightness_std","noise_est","edge_density","saturation_mean"].map(k=>(
                          <div key={k} className="stat" style={{fontSize:12}}>
                            <div style={{color:"#7f8c8d", fontSize:11}}>{k}</div>
                            <div style={{fontWeight:600, fontSize:13}}>{Number(result.image_stats[k] ?? 0).toFixed(3)}</div>
                          </div>
                        ))}
                      </div>
                      <details style={{marginTop:10}}>
                        <summary style={{cursor:"pointer", fontSize:13, color:"#5b7cff"}}>Show all features</summary>
                        <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:6, marginTop:8, fontSize:11}}>
                          {Object.entries(result.image_stats).map(([k,v])=>(
                            <div key={k} style={{display:"flex", justifyContent:"space-between", background:"#fff", padding:"6px 8px", borderRadius:8}}>
                              <span style={{color:"#7f8c8d"}}>{k}</span><strong>{Number(v).toFixed(2)}</strong>
                            </div>
                          ))}
                        </div>
                      </details>
                    </div>

                    {result.image_url && (
                      <div style={{marginTop:14}}>
                        <div style={{fontWeight:600, marginBottom:6}}>Stored image</div>
                        <img src={imageUrl(result.image_url)} style={{width:"100%", maxHeight:220, objectFit:"contain", borderRadius:12, background:"#fff", padding:6}}/>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="neo" style={{padding:16, marginTop:18, fontSize:12, color:"#7f8c8d"}}>
              <strong>How scoring works:</strong> 100 minus weighted penalty of ML issue probabilities + anomaly score (weights: blur 18, under/over 15, noise 14, severe 22, defect 20). Labels via thresholds on score and anomaly. Not a perceptual MOS — an application quality estimate.
              &nbsp;· SLO: CPU inference ~20–60 ms. &nbsp;· Model: MultiOutput RandomForest (180 trees) + IsolationForest on clean features.
            </div>
          </>
        )}

        {tab==="history" && (
          <div className="neo" style={{padding:18}}>
            <div style={{display:"flex", justifyContent:"space-between", alignItems:"center"}}>
              <h3 style={{margin:0}}>History</h3>
              <button className="btn" onClick={loadHistory}>Refresh</button>
            </div>
            {hLoading && <div style={{padding:18, color:"#7f8c8d"}}>Loading…</div>}
            {!hLoading && history.length===0 && (
              <div style={{padding:32, textAlign:"center", color:"#7f8c8d"}}>
                <div style={{fontSize:28}}>📭</div>
                <div>Empty history — analyze an image first.</div>
              </div>
            )}
            <div style={{display:"grid", gap:10, marginTop:14}}>
              {history.map((h:any)=>(
                <div key={h.id} className="history-card" onClick={async()=>{
                  const d=await fetch(`${(import.meta as any).env?.VITE_API_URL || ""}/api/v1/analysis/${h.id}`).then(r=>r.json())
                  setResult(d); setTab("analyze"); window.scrollTo({top:0, behavior:"smooth"})
                }}>
                  <img src={imageUrl(h.image_url)} style={{width:64, height:64, objectFit:"cover", borderRadius:10, background:"#eef2f7"}}/>
                  <div style={{flex:1}}>
                    <div style={{display:"flex", gap:8, alignItems:"center"}}>
                      <span className={`badge ${labelClass(h.quality_label)}`}>{h.quality_label}</span>
                      <strong>{h.quality_score}</strong>
                      <span style={{fontSize:12, color:"#7f8c8d"}}>{h.width}×{h.height}</span>
                    </div>
                    <div style={{fontSize:12, color:"#7f8c8d"}}>{h.original_filename} · {new Date(h.timestamp).toLocaleString()}</div>
                    <div style={{fontSize:11, color:"#7f8c8d"}}>{h.issues.map((i:any)=>i.type).join(", ") || "no issues"}</div>
                  </div>
                  <div style={{fontSize:12, color:"#5b7cff", fontWeight:600}}>View →</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
