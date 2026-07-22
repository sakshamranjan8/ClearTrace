import {
  type CSSProperties,
  type FormEvent,
  useEffect,
  useMemo,
  useState,
} from "react";

type View = "overview" | "sources" | "exposure" | "assistant" | "reports";
type User = { user_id: string; email: string; display_name: string; created_at: string };
type Point = { timestamp: string; hour_offset: number; aqi: number; category: string };
type Forecast = {
  location: string;
  lat: number;
  lon: number;
  hourly: Point[];
  nearest_stations: Array<{ station_name: string; distance_km: number; blend_weight: number }>;
  is_stale?: boolean;
};
type Summary = { avg_aqi: number; peak_aqi: number; peak_time: string; category: string };
type Indicator = {
  category: string;
  label: string;
  icon: string;
  strength: string;
  feature_count: number;
  nearest_distance_km: number;
  confidence_label: string;
  evidence: string[];
};
type Report = {
  report_id: string;
  category_guess: string;
  description: string;
  reporter_name?: string;
  status: string;
  created_at: string;
  distance_m?: number;
  upvotes: number;
  has_image: boolean;
  image_url?: string;
  user_id?: string;
};

const API = (import.meta.env.VITE_CLEARTRACE_API_URL || "http://127.0.0.1:8002/api").replace(/\/$/, "");
const LOCATIONS = [
  { name: "Connaught Place, Delhi", lat: 28.6315, lon: 77.2167 },
  { name: "Anand Vihar, Delhi", lat: 28.6469, lon: 77.316 },
  { name: "R K Puram, Delhi", lat: 28.5633, lon: 77.1869 },
  { name: "Punjabi Bagh, Delhi", lat: 28.674, lon: 77.131 },
  { name: "Lodhi Road, Delhi", lat: 28.5918, lon: 77.2273 },
  { name: "Dwarka Sector 8, Delhi", lat: 28.571, lon: 77.0719 },
];
const NAV: Array<[View, string, string]> = [
  ["overview", "⌂", "Citizen Dashboard"],
  ["sources", "◎", "Source Intelligence"],
  ["exposure", "✣", "Exposure Planner"],
  ["assistant", "☵", "AQI Chatbot"],
  ["reports", "◇", "Citizen Reports"],
];
const SAMPLE_VALUES = [156,149,144,139,136,142,154,171,188,204,213,207,198,187,178,169,161,155,150,147,151,158,165,171];

function category(aqi: number) {
  if (aqi <= 50) return "Good";
  if (aqi <= 100) return "Satisfactory";
  if (aqi <= 200) return "Moderate";
  if (aqi <= 300) return "Poor";
  if (aqi <= 400) return "Very Poor";
  return "Severe";
}

function previewForecast(location = LOCATIONS[0]): Forecast {
  const start = new Date();
  start.setMinutes(0, 0, 0);
  return {
    location: location.name,
    lat: location.lat,
    lon: location.lon,
    hourly: SAMPLE_VALUES.map((aqi, i) => ({
      timestamp: new Date(start.getTime() + (i + 1) * 3600000).toISOString(),
      hour_offset: i + 1,
      aqi,
      category: category(aqi),
    })),
    nearest_stations: [
      { station_name: "Mandir Marg", distance_km: 1.8, blend_weight: .44 },
      { station_name: "ITO", distance_km: 2.9, blend_weight: .32 },
      { station_name: "Lodhi Road", distance_km: 4.1, blend_weight: .24 },
    ],
  };
}

function summarize(forecast: Forecast): Summary {
  const average = forecast.hourly.reduce((sum, point) => sum + point.aqi, 0) / forecast.hourly.length;
  const peak = forecast.hourly.reduce((best, point) => point.aqi > best.aqi ? point : best);
  return { avg_aqi: Math.round(average), peak_aqi: Math.round(peak.aqi), peak_time: peak.timestamp, category: category(average) };
}

async function request<T>(path: string, options: RequestInit = {}, token?: string | null): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (!response.ok) {
    let message = "Something went wrong.";
    try { message = (await response.json()).detail || message; } catch { /* keep fallback */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

const time = (value: string) => new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
const dateTime = (value: string) => new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
const initials = (name: string) => name.split(" ").slice(0,2).map((word) => word[0]?.toUpperCase()).join("");

function Chart({ points }: { points: Point[] }) {
  const values = points.map((p) => p.aqi);
  const low = Math.min(...values, 0);
  const high = Math.max(...values, 300);
  const path = points.map((p, i) => {
    const x = i / Math.max(points.length - 1, 1) * 1000;
    const y = 225 - (p.aqi - low) / Math.max(high - low, 1) * 185;
    return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return <div className="chart" aria-label="24-hour AQI forecast">
    <div className="bands"><i/><i/><i/><i/></div>
    <svg viewBox="0 0 1000 245" preserveAspectRatio="none">
      <defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#08a997" stopOpacity=".4"/><stop offset="1" stopColor="#08a997" stopOpacity="0"/></linearGradient></defs>
      <path d={`${path} L1000,240 L0,240 Z`} fill="url(#fill)"/>
      <path d={path} fill="none" stroke="#047d71" strokeWidth="5"/>
    </svg>
    <div className="chart-labels">{points.filter((_, i) => i % 5 === 0).map((p) => <span key={p.timestamp}>{time(p.timestamp)}</span>)}</div>
  </div>;
}

function AuthModal({ mode: initial, close, done }: { mode: "login"|"register"; close: () => void; done: (user: User, token: string) => void }) {
  const [mode, setMode] = useState(initial);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(""); setBusy(true);
    const form = new FormData(event.currentTarget);
    try {
      const payload = await request<{user: User; token: string}>(`/auth/${mode}`, {
        method: "POST",
        body: JSON.stringify({
          email: form.get("email"), password: form.get("password"),
          ...(mode === "register" ? { display_name: form.get("display_name") } : {}),
        }),
      });
      done(payload.user, payload.token);
    } catch (err) { setError(err instanceof Error ? err.message : "Unable to continue."); }
    finally { setBusy(false); }
  }
  return <div className="modal-backdrop" onMouseDown={close}>
    <section className="auth-modal" onMouseDown={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
      <button className="modal-close" onClick={close} aria-label="Close">×</button>
      <div className="auth-mark"><i/></div><span className="eyebrow">CLEARTRACE CITIZEN NETWORK</span>
      <h2>{mode === "login" ? "Welcome back." : "Create your account."}</h2>
      <p>{mode === "login" ? "Sign in to submit, view, and verify citizen pollution reports." : "Registration is optional. Every air-intelligence feature remains public."}</p>
      <form onSubmit={submit}>
        {mode === "register" && <label>Full name<input name="display_name" minLength={2} autoComplete="name" required/></label>}
        <label>Email address<input name="email" type="email" autoComplete="email" required/></label>
        <label>Password<input name="password" type="password" minLength={mode === "register" ? 8 : 1} autoComplete={mode === "register" ? "new-password" : "current-password"} required/></label>
        {mode === "register" && <small>Use 8+ characters with at least one letter and one number.</small>}
        {error && <div className="form-error">{error}</div>}
        <button className="primary wide" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
      </form>
      <button className="link-button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>{mode === "login" ? "New here? Create an account" : "Already registered? Sign in"}</button>
    </section>
  </div>;
}

function ReportImage({ report, token }: { report: Report; token: string }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!report.image_url) return;
    const controller = new AbortController();
    const origin = API.endsWith("/api") ? API.slice(0, -4) : API;
    fetch(`${origin}${report.image_url}`, { headers: { Authorization: `Bearer ${token}` }, signal: controller.signal })
      .then((response) => response.ok ? response.blob() : null)
      .then((blob) => blob && setUrl(URL.createObjectURL(blob))).catch(() => undefined);
    return () => controller.abort();
  }, [report.image_url, token]);
  return url ? <img src={url} alt={report.category_guess}/> : <div className="photo-placeholder">{report.category_guess[0]?.toUpperCase()}</div>;
}

export default function App() {
  const [view, setView] = useState<View>("overview");
  const [location, setLocation] = useState(LOCATIONS[0]);
  const [forecast, setForecast] = useState(() => previewForecast());
  const [summary, setSummary] = useState(() => summarize(previewForecast()));
  const [indicators, setIndicators] = useState<Indicator[]>([]);
  const [status, setStatus] = useState<"connecting"|"live"|"preview">("connecting");
  const [user, setUser] = useState<User|null>(null);
  const [token, setToken] = useState<string|null>(null);
  const [authMode, setAuthMode] = useState<"login"|"register"|null>(null);
  const [advisory, setAdvisory] = useState<Record<string, unknown>|null>(null);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState([{ role: "assistant", content: "Hi — I’m ClearTrace. Ask me about today’s forecast, nearby source indicators, or safer outdoor timing." }]);
  const [reports, setReports] = useState<Report[]>([]);
  const [mine, setMine] = useState<Report[]>([]);
  const [reportTab, setReportTab] = useState<"nearby"|"submit"|"mine">("nearby");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem("cleartrace_token");
    if (stored) request<{user: User}>("/auth/me", {}, stored).then(({user}) => { setUser(user); setToken(stored); }).catch(() => localStorage.removeItem("cleartrace_token"));
  }, []);

  useEffect(() => {
    let active = true; setStatus("connecting");
    const qs = new URLSearchParams({ latitude: String(location.lat), longitude: String(location.lon), location: location.name });
    Promise.all([
      request<{forecast: Forecast; summary: Summary}>(`/forecast?${qs}`),
      request<{indicators: Indicator[]}>(`/source-indicators?latitude=${location.lat}&longitude=${location.lon}&radius_km=5`),
    ]).then(([fc, src]) => {
      if (!active) return; setForecast(fc.forecast); setSummary(fc.summary); setIndicators(src.indicators); setStatus("live");
    }).catch(() => {
      if (!active) return; const sample = previewForecast(location); setForecast(sample); setSummary(summarize(sample)); setIndicators([
        {category:"traffic",label:"Road traffic",icon:"↗",strength:"High",feature_count:14,nearest_distance_km:.18,confidence_label:"Mapped contextual record",evidence:["Major-road corridor within 0.18 km","14 eligible mapped segments nearby"]},
        {category:"industrial",label:"Industrial activity",icon:"▥",strength:"Medium",feature_count:3,nearest_distance_km:2.4,confidence_label:"Official or strongly verified record",evidence:["Nearest mapped industrial feature is 2.4 km away"]},
        {category:"waste",label:"Waste sites",icon:"◫",strength:"Low",feature_count:1,nearest_distance_km:4.7,confidence_label:"Mapped contextual record",evidence:["One eligible mapped feature within 5 km"]},
      ]); setStatus("preview");
    });
    return () => { active = false; };
  }, [location]);

  useEffect(() => { if (view === "reports" && token) void loadReports(token); }, [view, token, location]);
  const peakTime = useMemo(() => time(summary.peak_time), [summary.peak_time]);

  function authenticated(nextUser: User, nextToken: string) {
    localStorage.setItem("cleartrace_token", nextToken); setUser(nextUser); setToken(nextToken); setAuthMode(null); setView("reports");
  }
  async function signOut() {
    if (token) await request("/auth/logout", {method:"POST"}, token).catch(() => undefined);
    localStorage.removeItem("cleartrace_token"); setUser(null); setToken(null); setView("overview");
  }
  function useLocation() {
    navigator.geolocation?.getCurrentPosition(({coords}) => setLocation({name:"Your current location",lat:coords.latitude,lon:coords.longitude}));
  }
  async function loadReports(activeToken: string) {
    try {
      const [nearby, own] = await Promise.all([
        request<{reports: Report[]}>(`/reports/nearby?latitude=${location.lat}&longitude=${location.lon}&radius_m=10000`,{},activeToken),
        request<{reports: Report[]}>("/reports/mine",{},activeToken),
      ]); setReports(nearby.reports); setMine(own.reports);
    } catch (err) { setNotice(err instanceof Error ? err.message : "Unable to load reports."); }
  }
  async function plan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); const form = new FormData(event.currentTarget);
    try { setAdvisory(await request("/exposure-advisory",{method:"POST",body:JSON.stringify({location:location.name,latitude:location.lat,longitude:location.lon,start_index:Number(form.get("start_index")),duration_hours:Number(form.get("duration_hours")),sensitivity_group:form.get("sensitivity_group"),activity_level:form.get("activity_level")})})); }
    catch(err) { setAdvisory({error:err instanceof Error?err.message:"Advisory unavailable."}); } finally { setBusy(false); }
  }
  async function ask(question: string) {
    if (!question.trim() || busy) return; setMessages((all) => [...all,{role:"user",content:question}]); setBusy(true);
    try { const reply = await request<{answer:string}>("/chat",{method:"POST",body:JSON.stringify({question,location:location.name,latitude:location.lat,longitude:location.lon,user_category:"adult"})}); setMessages((all)=>[...all,{role:"assistant",content:reply.answer}]); }
    catch(err) { setMessages((all)=>[...all,{role:"assistant",content:err instanceof Error?err.message:"Assistant unavailable."}]); } finally { setBusy(false); }
  }
  async function submitReport(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (!token) return setAuthMode("login"); const form = new FormData(event.currentTarget); form.set("latitude",String(location.lat)); form.set("longitude",String(location.lon)); setNotice("Submitting report…");
    try { await request("/reports",{method:"POST",body:form},token); event.currentTarget.reset(); setNotice("Report submitted for community verification."); setReportTab("mine"); await loadReports(token); }
    catch(err) { setNotice(err instanceof Error?err.message:"Unable to submit report."); }
  }
  async function vote(id: string) {
    if (!token) return; try { await request(`/reports/${id}/vote`,{method:"POST",body:JSON.stringify({vote_type:"upvote"})},token); setNotice("Verification recorded — thank you."); await loadReports(token); }
    catch(err) { setNotice(err instanceof Error?err.message:"Unable to verify report."); }
  }

  const Overview = () => <>
    <section className="hero">
      <div><span className="hero-label">DELHI AIR INTELLIGENCE DASHBOARD</span><h1>Your next 24 hours,<br/>made clearer.</h1><p>Personalized exposure guidance powered by hyper-local forecasts for Delhi NCR.</p><div className="hero-chips"><span>● {status === "live" ? "Live model" : status === "connecting" ? "Connecting" : "Preview mode"}</span><span>{location.lat.toFixed(3)}, {location.lon.toFixed(3)}</span></div></div>
      <div className="air-orb"><i/><i/><i/></div>
    </section>
    {status === "preview" && <div className="connection-note">Live services are offline, so the dashboard is showing clearly labelled preview data. Start the Python services to switch automatically.</div>}
    <section className="metrics">
      {[ ["24H AVERAGE",summary.avg_aqi,"AQI",summary.category], ["FORECAST PEAK",summary.peak_aqi,"AQI",category(summary.peak_aqi)], ["PEAK TIME",peakTime,"","Daily surge"] ].map(([label,value,unit,badge]) => <article className="metric" key={label}><span className="eyebrow dark">{label}</span><div>{value} <small>{unit}</small></div><b className={`pill ${String(badge).toLowerCase().replaceAll(" ","-")}`}>{badge}</b></article>)}
    </section>
    <section className="panel chart-panel"><div className="section-head"><div><span className="eyebrow">MODEL FORECAST</span><h2>Air quality trend</h2><p>Hourly prediction across the next 24 hours</p></div><button className="icon-button" onClick={() => { const csv=["timestamp,aqi,category",...forecast.hourly.map(p=>`${p.timestamp},${p.aqi},${p.category}`)].join("\n"); const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv]));a.download="cleartrace-forecast.csv";a.click(); }}>↓</button></div><Chart points={forecast.hourly}/></section>
    <section className="lower-grid"><article className="panel station-panel"><span className="eyebrow">SPATIAL BLEND</span><h3>Nearest monitoring stations</h3>{forecast.nearest_stations.map((s)=><div className="station" key={s.station_name}><div><strong>{s.station_name}</strong><span>{s.distance_km.toFixed(1)} km away</span></div><i><em style={{width:`${s.blend_weight*100}%`}}/></i><b>{Math.round(s.blend_weight*100)}%</b></div>)}</article><article className="panel quick-card"><span className="eyebrow">PLAN AHEAD</span><h3>Turn data into a safer plan.</h3><p>Compare your timing, activity, and sensitivity against the exact forecast window.</p><button className="primary" onClick={()=>setView("exposure")}>Plan outdoor exposure →</button></article></section>
  </>;

  const Sources = () => <>
    <section className="page-intro"><span className="eyebrow">NEARBY CONTEXT, NOT APPORTIONMENT</span><h1>What could be influencing the air around you?</h1><p>Evidence-led proximity signals from mapped roads, industry, waste, power, and construction context.</p></section>
    <section className="source-layout"><div className="panel map-card"><div className="map-line"><span>5 km intelligence radius</span><strong>{location.name}</strong></div><div className="map"><i className="road one"/><i className="road two"/><i className="road three"/><i className="ring one"/><i className="ring two"/><b className="you">YOU</b>{indicators.slice(0,4).map((item,i)=><span className={`marker m${i+1}`} key={item.category}>{item.icon}</span>)}</div><div className="map-line"><span>Exact-coordinate geometry proximity</span><span>Wind adjustment: not included</span></div></div><div className="source-list">{indicators.map(item=><article className="source-card" key={item.category}><div className="source-icon">{item.icon}</div><div><header><h3>{item.label}</h3><b className={`signal ${item.strength.toLowerCase()}`}>{item.strength}</b></header><p>{item.evidence[0]}</p><footer><span>{item.feature_count} mapped features</span><span>{item.confidence_label}</span></footer></div></article>)}</div></section>
    <div className="method-note"><strong>Interpretation boundary:</strong> these are nearby source indicators, not estimated pollution-contribution percentages. Live activity, wind, and pollutant chemistry are not yet included.</div>
  </>;

  const Exposure = () => {
    const a = advisory as {error?:string;headline?:string;action_level?:string;action_color?:string;peak_aqi?:number;mean_aqi?:number;category?:string;health_message?:string;recommendations?:string[];method_note?:string}|null;
    return <><section className="page-intro"><span className="eyebrow">PERSONAL EXPOSURE PLANNER</span><h1>Choose a better time to step outside.</h1><p>ClearTrace evaluates the actual forecast hours you select—without inventing a medical risk score.</p></section><section className="planner-grid"><form className="panel planner-form" onSubmit={plan}><div><h2>Your outdoor plan</h2><p>Adjust the details below to generate guidance.</p></div><label>Planned start<select name="start_index">{forecast.hourly.map((p,i)=><option value={i} key={p.timestamp}>{time(p.timestamp)} · AQI {Math.round(p.aqi)}</option>)}</select></label><label>Time outdoors<select name="duration_hours"><option value="1">1 hour</option><option value="2">2 hours</option><option value="4">4 hours</option><option value="6">6 hours</option></select></label><label>Who is this for?<select name="sensitivity_group"><option>General population</option><option>Child or teenager</option><option>Older adult (65+)</option><option>Pregnant</option><option>Asthma or COPD</option><option>Heart condition</option></select></label><label>Planned activity<select name="activity_level"><option>Light activity (walking / commuting)</option><option>Moderate activity (brisk walk / cycling)</option><option>Strenuous activity (running / outdoor sport)</option></select></label><button className="primary wide" disabled={busy}>{busy?"Analysing forecast…":"Build my exposure plan"}</button></form><article className="advisory" style={{"--risk":a?.action_color||"#ef4444"} as CSSProperties}>{!a?<div className="empty"><span>✦</span><h2>Your advisory will appear here.</h2><p>We’ll use the forecast peak inside your chosen window, plus activity and sensitivity.</p></div>:a.error?<div className="empty"><span>!</span><h2>Advisory unavailable</h2><p>{a.error}</p></div>:<><div className="advisory-badges"><b>{a.action_level}</b><b>{a.category} air</b></div><h2>{a.headline}</h2><p>{a.health_message}</p><div className="advisory-metrics"><div><b>{a.mean_aqi}</b><span>Mean AQI</span></div><div><b>{a.peak_aqi}</b><span>Peak AQI</span></div></div><ul>{a.recommendations?.map(r=><li key={r}><span>✓</span>{r}</li>)}</ul><small>{a.method_note}</small></>}</article></section></>;
  };

  const Assistant = () => <section className="assistant-layout"><aside><span className="eyebrow">GROUNDED AQI ASSISTANT</span><h1>Ask the air,<br/>not the internet.</h1><p>Answers are grounded in forecasts, mapped-source context, verified reports, and advisory documents.</p><div className="context-chips"><span>24h forecast</span><span>Source indicators</span><span>Health guidance</span><span>Verified reports</span></div><small>ClearTrace provides general guidance, not diagnosis or emergency advice.</small></aside><div className="chat"><header><b>CT</b><div><strong>ClearTrace Assistant</strong><span>● Grounded and ready</span></div></header><div className="messages">{messages.map((m,i)=><p className={m.role} key={i}>{m.content}</p>)}{busy&&<p className="assistant typing">•••</p>}</div><div className="prompt-row">{["Is it safe to jog tomorrow morning?","Why might AQI be high here?","What should a school do today?"].map(q=><button key={q} onClick={()=>ask(q)}>{q}</button>)}</div><form onSubmit={(e)=>{e.preventDefault();const input=e.currentTarget.elements.namedItem("question") as HTMLInputElement;const q=input.value;input.value="";void ask(q);}}><input name="question" placeholder={`Ask about air near ${location.name}…`}/><button>↑</button></form></div></section>;

  const Reports = () => {
    if (!user || !token) return <section className="gate"><div className="gate-art"><span>◇</span><i/><i/></div><span className="eyebrow">PROTECTED CITIZEN NETWORK</span><h1>Evidence works better with accountability.</h1><p>Sign in to view nearby reports, submit a pollution source, and verify evidence. Forecasts, source analysis, exposure guidance, and the chatbot stay open to everyone.</p><div><button className="primary" onClick={()=>setAuthMode("login")}>Sign in to reports</button><button className="secondary" onClick={()=>setAuthMode("register")}>Create account</button></div><small>No external auth provider or API key is required.</small></section>;
    const visible = reportTab === "mine" ? mine : reports;
    return <><section className="page-intro reports-head"><div><span className="eyebrow">COMMUNITY EVIDENCE</span><h1>Citizen report network</h1><p>Submit local evidence and help verify reports near {location.name}.</p></div><button className="primary" onClick={()=>setReportTab("submit")}>＋ New report</button></section><nav className="tabs"><button className={reportTab==="nearby"?"active":""} onClick={()=>setReportTab("nearby")}>Nearby verification <span>{reports.length}</span></button><button className={reportTab==="submit"?"active":""} onClick={()=>setReportTab("submit")}>Submit report</button><button className={reportTab==="mine"?"active":""} onClick={()=>setReportTab("mine")}>My activity <span>{mine.length}</span></button></nav>{notice&&<div className="notice">{notice}<button onClick={()=>setNotice("")}>×</button></div>}{reportTab==="submit"?<section className="panel report-form"><div><span className="eyebrow">FIELD EVIDENCE</span><h2>Report a visible pollution source.</h2><p>Add a clear description and optional photo. Your selected location is attached automatically.</p><div className="location-card"><strong>{location.name}</strong><span>{location.lat.toFixed(5)}, {location.lon.toFixed(5)}</span></div></div><form onSubmit={submitReport}><label>Source category<select name="category"><option value="traffic">Traffic</option><option value="construction">Construction</option><option value="waste_burning">Waste burning</option><option value="dust">Road dust</option><option value="industry">Industrial activity</option><option value="other">Other visible source</option></select></label><label>What did you observe?<textarea name="description" minLength={8} maxLength={1000} rows={5} required placeholder="Describe what is visible, when it started, and useful context…"/></label><label className="upload">Optional photo<input name="image" type="file" accept="image/jpeg,image/png"/><span>JPEG or PNG · maximum 5 MB</span></label><button className="primary wide">Submit for verification</button></form></section>:visible.length===0?<section className="panel empty-reports"><span>◇</span><h2>{reportTab==="mine"?"You haven’t submitted a report yet.":"No reports found near this location."}</h2><p>Citizen evidence will appear here as the network grows.</p>{reportTab==="mine"&&<button className="primary" onClick={()=>setReportTab("submit")}>Create first report</button>}</section>:<section className="report-grid">{visible.map(r=><article className="report-card" key={r.report_id}><div className="report-photo">{r.has_image?<ReportImage report={r} token={token}/>:<div className="photo-placeholder">{r.category_guess[0]?.toUpperCase()}</div>}<b className={`report-status ${r.status}`}>{r.status}</b></div><div className="report-body"><span className="eyebrow">{r.category_guess.replaceAll("_"," ")}</span><h3>{r.description}</h3><div className="report-meta"><span>By {r.reporter_name||"Citizen"}</span><span>{dateTime(r.created_at)}</span>{r.distance_m!==undefined&&<span>{r.distance_m<1000?`${Math.round(r.distance_m)} m`:`${(r.distance_m/1000).toFixed(1)} km`} away</span>}</div><footer><span><b>{r.upvotes}</b> verifications</span>{reportTab==="nearby"&&r.user_id!==user.user_id&&<button onClick={()=>vote(r.report_id)}>✓ Verify evidence</button>}</footer></div></article>)}</section>}</>;
  };

  return <div className="shell">
    <aside className="sidebar"><div className="brand"><div className="brand-mark"><i/><i/></div>ClearTrace</div><nav>{NAV.map(([id,icon,label])=><button className={view===id?"active":""} onClick={()=>setView(id)} key={id}><b>{icon}</b>{label}{id==="reports"&&!user&&<small>LOCKED</small>}</button>)}</nav><div className="sidebar-bottom"><label>FORECAST LOCATION<select value={location.name} onChange={(e)=>{const next=LOCATIONS.find(x=>x.name===e.target.value);if(next)setLocation(next);}}>{!LOCATIONS.some(x=>x.name===location.name)&&<option>{location.name}</option>}{LOCATIONS.map(x=><option key={x.name}>{x.name}</option>)}</select></label><button onClick={useLocation}>⌖ Use my location</button></div><footer><span>DELHI AIR INTELLIGENCE</span><small>Prototype v2.5</small></footer></aside>
    <main><header className="topbar"><div className={`service ${status}`}><i/>{status==="live"?"Live intelligence":status==="connecting"?"Connecting services":"Preview data"}</div>{user?<div className="user"><div><span>Welcome back,</span><strong>{user.display_name}</strong></div><b>{initials(user.display_name)}</b><button onClick={signOut}>Sign out</button></div>:<div className="auth-actions"><button onClick={()=>setAuthMode("login")}>Sign in</button><button onClick={()=>setAuthMode("register")}>Create account</button></div>}</header><div className="content">{view==="overview"?<Overview/>:view==="sources"?<Sources/>:view==="exposure"?<Exposure/>:view==="assistant"?<Assistant/>:<Reports/>}</div></main>
    {authMode&&<AuthModal mode={authMode} close={()=>setAuthMode(null)} done={authenticated}/>} 
  </div>;
}
