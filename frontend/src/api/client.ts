const BASE = (import.meta as any).env?.VITE_API_URL || ""

export async function analyzeImage(file: File) {
  const fd = new FormData()
  fd.append("file", file)
  const res = await fetch(`${BASE}/api/v1/analyze`, { method: "POST", body: fd })
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || `HTTP ${res.status}`)
  }
  return res.json()
}
export async function fetchHistory(limit=20, offset=0) {
  const res = await fetch(`${BASE}/api/v1/history?limit=${limit}&offset=${offset}`)
  if (!res.ok) throw new Error("history failed")
  return res.json()
}
export async function fetchDetail(id: string) {
  const res = await fetch(`${BASE}/api/v1/analysis/${id}`)
  if (!res.ok) throw new Error("detail failed")
  return res.json()
}
export async function fetchHealth(){
  const res = await fetch(`${BASE}/api/v1/health`)
  return res.json()
}
export function imageUrl(path: string){
  if(path.startsWith("http")) return path
  return `${BASE}${path}`
}
