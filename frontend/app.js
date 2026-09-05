/* ============================================================
   AI JOB FINDER — frontend controller
   Talks to the SSE endpoint, renders live agent progress,
   and draws the final job report.
   All user/job text is rendered via textContent (never innerHTML)
   except the server-generated markdown report, which is trusted
   backend output and rendered through DOMPurify-free lightweight
   sanitization.
   ============================================================ */
"use strict";

const $ = (id) => document.getElementById(id);

const PIPELINE_STEPS = [
  { id: "understand", label: "UNDERSTAND" },
  { id: "plan", label: "SEARCH PLAN" },
  { id: "search", label: "WEB SEARCH" },
  { id: "analyze", label: "ANALYZE" },
  { id: "verify", label: "VERIFY" },
  { id: "report", label: "REPORT" },
];

let resultData = null;

/* ------------------------------------------------------------- init */
document.addEventListener("DOMContentLoaded", () => {
  buildPipeline();
  bindLanding();
  bindResults();
  checkHealth();
});

async function checkHealth() {
  const pill = $("landing-api-pill");
  try {
    const res = await fetch("/api/health");
    const json = await res.json();
    if (json.openrouter_key_configured) {
      pill.textContent = "OPENROUTER READY";
      pill.classList.add("ok");
    } else {
      pill.textContent = "OPENROUTER KEY MISSING";
      pill.classList.add("err");
    }
  } catch {
    pill.textContent = "SERVICES UNREACHABLE";
    pill.classList.add("err");
  }
}

/* ---------------------------------------------------------- landing */
function bindLanding() {
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    startSearch();
  });
  $("btn-new").addEventListener("click", resetAll);
}

function startSearch() {
  const role = $("role").value.trim();
  const location = $("location").value.trim();
  const experience = $("experience").value.trim();
  const skills = $("skills").value.trim();
  const salary = $("salary").value.trim();
  const work_mode = $("work_mode").value.trim();

  const err = $("form-error");
  err.classList.add("hidden");
  if (!role) {
    err.textContent = "Please enter a job role.";
    err.classList.remove("hidden");
    return;
  }

  $("view-landing").classList.add("hidden");
  $("view-research").classList.remove("hidden");
  $("view-results").classList.add("hidden");

  clearResearch();
  const query = role + " · " + (location || "anywhere") + " · " + (experience || "any");
  $("search-query").textContent = query;

  runSearch({ role, location, experience, skills, salary, work_mode });
}

async function runSearch(payload) {
  let response;
  try {
    response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    showError("Could not reach the Job Finder backend.", String(err));
    return;
  }
  if (!response.ok) {
    let detail = "HTTP " + response.status;
    try { detail = (await response.text()).slice(0, 300) || detail; } catch {}
    showError("The search request was rejected.", detail);
    return;
  }

  try {
    const stream = sseStream(response);
    for await (const item of stream) {
      handleEvent(item);
    }
  } catch (err) {
    const msg = (err && err.message) ? err.message : String(err);
    showError("The job search was interrupted.", msg);
  }
}

async function* sseStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        const data = line.slice(6);
        if (data) yield JSON.parse(data);
      }
    }
  }
}

/* ---------------------------------------------------------- events */
function handleEvent(evt) {
  switch (evt.type) {
    case "activity":
      onActivity(evt.kind, evt.message);
      break;
    case "progress":
      onProgress(evt.percent, evt.label);
      break;
    case "step":
      onStepPulse(evt.index, evt.total, evt.label);
      break;
    case "result":
      onResult(evt.payload);
      break;
    case "error":
      showError(evt.message, evt.detail || "");
      break;
    default:
      break;
  }
}

function onProgress(percent, label) {
  const pct = Math.min(100, Math.max(0, percent));
  $("progress-fill").style.width = pct + "%";
  $("progress-pct").textContent = pct + "%";
  if (label) $("objective").textContent = label;
}

function onStepPulse(index, total, label) {
  if (label) $("objective").textContent = label;
  const node = document.querySelector(`.map-node[data-step="${label.toLowerCase()}"]`);
  if (node) setNode(node, "investigating");
}

function onActivity(kind, message) {
  const feed = $("activity-feed");
  const item = document.createElement("div");
  item.className = "feed-item " + (kind || "info");
  const ico = document.createElement("span");
  ico.className = "fi-ico";
  ico.textContent = { ok: "✓", info: "→", warn: "▲", action: "⚡" }[kind] || "·";
  const txt = document.createElement("span");
  txt.className = "fi-txt";
  txt.textContent = message;
  item.appendChild(ico);
  item.appendChild(txt);

  const prev = feed.lastElementChild;
  feed.insertBefore(item, prev && prev.classList.contains("feed-pending") ? prev : null);
  feed.scrollTop = feed.scrollHeight;

  // mark the pipeline step completed when a phase is done
  if (kind === "ok") {
    const action = message.toLowerCase();
    if (action.includes("understood")) markDone("understand");
    else if (action.includes("search plan")) markDone("plan");
    else if (action.includes("jobs collected")) markDone("search");
    else if (action.includes("market analysis")) markDone("analyze");
    else if (action.includes("verified")) markDone("verify");
    else if (action.includes("report generated")) markDone("report");
  }
}

function markDone(stepId) {
  const node = document.querySelector(`.map-node[data-step="${stepId}"]`);
  if (node) setNode(node, "completed");
}

/* ---------------------------------------------------------------- map */
function buildPipeline() {
  const col = $("pipeline-map");
  col.textContent = "";
  PIPELINE_STEPS.forEach((s, i) => {
    const node = document.createElement("div");
    node.className = "map-node locked";
    node.dataset.step = s.id;

    const connector = document.createElement("span");
    connector.className = "connector";

    const ico = document.createElement("span");
    ico.className = "map-ico";
    ico.textContent = "·";

    const label = document.createElement("span");
    label.className = "map-node-label";
    label.textContent = s.label;

    const sub = document.createElement("span");
    sub.className = "map-node-sub";
    sub.textContent = "PENDING";

    node.appendChild(connector);
    node.appendChild(ico);
    node.appendChild(label);
    node.appendChild(sub);
    col.appendChild(node);
  });
}

function setNode(node, status) {
  const rank = { locked: 0, investigating: 1, completed: 2, warning: 3 };
  const current = getNodeRank(node);
  if (rank[status] < current) return;
  node.className = "map-node " + status;
  const ico = node.querySelector(".map-ico");
  const sub = node.querySelector(".map-node-sub");
  if (ico) ico.textContent = { locked: "·", investigating: "◌", completed: "✓", warning: "▲" }[status] || "·";
  if (sub) sub.textContent = { locked: "PENDING", investigating: "WORKING", completed: "DONE", warning: "CHECK" }[status] || "";
}

function getNodeRank(node) {
  const cls = node.className;
  if (cls.includes("completed")) return 2;
  if (cls.includes("investigating")) return 1;
  if (cls.includes("warning")) return 3;
  return 0;
}

/* --------------------------------------------------------------- result */
function onResult(payload) {
  resultData = payload;
  $("view-research").classList.add("hidden");
  $("view-results").classList.remove("hidden");
  renderResults(payload);
}

function renderResults(p) {
  const jobs = p.jobs || [];
  const req = p.request || {};
  $("results-query").textContent =
    (req.role || "—") + " · " + (req.location || "—") + " · " + (req.experience || "—");
  $("jobs-count").textContent = jobs.length;
  $("results-meta").textContent =
    "Role: " + (req.role || "Not specified") + " | Location: " + (req.location || "Not specified") +
    " | Experience: " + (req.experience || "Not specified") +
    (p.ai_available ? " | Model: " + (p.model || "—") : " | AI unavailable — bounded results");

  if (p.report_markdown) {
    const box = $("markdown-report");
    box.classList.remove("hidden");
    box.textContent = p.report_markdown;
  }

  renderJobs(jobs, p.skill_match || {});
  renderInsights(p.analysis || {});
  renderSkills(p.skill_match || {});
}

function renderJobs(jobs, skillMatch) {
  const list = $("jobs-list");
  list.textContent = "";
  $("jobs-hint").textContent = jobs.length ? "listing " + jobs.length + " verified job(s)" : "no listings verified";

  if (!jobs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-note";
    empty.style.color = "var(--text-dim)";
    empty.style.padding = "20px";
    empty.textContent = "No reliable job listings could be verified. Try a broader role, location, or rerun the search.";
    list.appendChild(empty);
    return;
  }

  const sorted = jobs.slice().sort((a, b) => (b.match_score || 0) - (a.match_score || 0));

  sorted.forEach((job) => {
    const card = document.createElement("div");
    card.className = "job-card";

    const head = document.createElement("div");
    head.className = "job-card-head";

    const info = document.createElement("div");
    const title = document.createElement("div");
    title.className = "job-title";
    title.textContent = job.title || "Not specified";
    const company = document.createElement("div");
    company.className = "job-company";
    company.textContent = job.company || "Not specified";
    info.appendChild(title);
    info.appendChild(company);

    const score = job.match_score != null ? Math.round(job.match_score) : null;
    const scoreEl = document.createElement("span");
    scoreEl.className = "job-score" + (score == null ? "" : score >= 60 ? " high" : score < 30 ? " low" : "");
    scoreEl.textContent = score == null ? "—" : score + "% match";
    if (score == null) scoreEl.style.opacity = "0.5";
    head.appendChild(info);
    head.appendChild(scoreEl);
    card.appendChild(head);

    const meta = document.createElement("div");
    meta.className = "job-meta";
    addTag(meta, "Location", job.location);
    addTag(meta, "Experience", job.experience);
    addTag(meta, "Type", job.job_type);
    addTag(meta, "Salary", job.salary);
    addTag(meta, "Posted", job.posted_date);
    card.appendChild(meta);

    const skills = document.createElement("div");
    skills.className = "job-skills";
    const sb = document.createElement("b");
    sb.textContent = "SKILLS: ";
    skills.appendChild(sb);
    skills.appendChild(document.createTextNode(job.skills || "Not specified"));
    card.appendChild(skills);

    const url = (job.url || "").trim();
    if (url && url !== "Not specified") {
      const link = document.createElement("a");
      link.className = "job-url";
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = url.length > 70 ? url.slice(0, 70) + "…" : url;
      card.appendChild(link);
    }

    list.appendChild(card);
  });
}

function addTag(container, label, value) {
  const v = (value || "").trim();
  if (!v || v === "Not specified") return;
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = label + ": " + v;
  container.appendChild(tag);
}

function renderInsights(a) {
  const box = $("insights-box");
  box.textContent = "";
  if (!a) {
    const div = document.createElement("div");
    div.className = "insight-text";
    div.textContent = "No market insights available.";
    box.appendChild(div);
    return;
  }
  addInsightGroup(box, "MOST REQUESTED SKILLS", a.most_requested_skills, "chip");
  addInsightGroup(box, "MOST REQUESTED TECHNOLOGIES", a.most_requested_technologies, "chip");
  addInsightGroup(box, "COMMON EXPERIENCE REQUIREMENTS", a.experience_requirements, "chip");
  addInsightGroup(box, "HIRING COMPANIES", a.companies_hiring, "chip");
  addInsightGroup(box, "COMMON LOCATIONS", a.common_locations, "chip");
  if (a.remote_opportunities && a.remote_opportunities.length) {
    addInsightGroup(box, "REMOTE OPPORTUNITIES", a.remote_opportunities, "chip");
  }
  addInsightGroup(box, "SALARY INFORMATION", [a.salary_insights || "Not enough data available."], "insight-text");
}

function addInsightGroup(box, label, values, cls) {
  if (!values || !values.length) return;
  const group = document.createElement("div");
  group.className = "insight-group";
  const b = document.createElement("b");
  b.textContent = label;
  group.appendChild(b);
  if (cls === "insight-text") {
    const div = document.createElement("div");
    div.className = "insight-text";
    div.textContent = values.join(" ");
    group.appendChild(div);
  } else {
    const chips = document.createElement("div");
    chips.className = "chips";
    values.slice(0, 12).forEach((v) => {
      const chip = document.createElement("span");
      chip.className = cls;
      chip.textContent = v;
      chips.appendChild(chip);
    });
    group.appendChild(chips);
  }
  box.appendChild(group);
}

function renderSkills(m) {
  const box = $("skills-box");
  box.textContent = "";
  if (!m) {
    const div = document.createElement("div");
    div.className = "insight-text";
    div.textContent = "No skill match available.";
    box.appendChild(div);
    return;
  }
  addChipGroup(box, "SKILLS YOU HAVE", m.matching, "chip");
  addChipGroup(box, "COMMONLY REQUESTED — MISSING", m.missing, "chip missing");
  addChipGroup(box, "RECOMMENDED TO LEARN", m.recommended, "chip learn");
}

function addChipGroup(box, label, values, cls) {
  if (!values || !values.length) return;
  const group = document.createElement("div");
  group.className = "insight-group";
  const b = document.createElement("b");
  b.textContent = label;
  group.appendChild(b);
  const chips = document.createElement("div");
  chips.className = "chips";
  values.slice(0, 12).forEach((v) => {
    const chip = document.createElement("span");
    chip.className = cls;
    chip.textContent = v;
    chips.appendChild(chip);
  });
  group.appendChild(chips);
  box.appendChild(group);
}

/* ------------------------------------------------------------ controls */
function resetAll() {
  resultData = null;
  $("view-results").classList.add("hidden");
  $("view-research").classList.add("hidden");
  $("view-landing").classList.remove("hidden");
  $("form-error").classList.add("hidden");
  $("markdown-report").classList.add("hidden");
}

function clearResearch() {
  $("activity-feed").textContent = "";
  $("progress-fill").style.width = "0";
  $("progress-pct").textContent = "0%";
  $("objective").textContent = "Initializing…";
  document.querySelectorAll(".map-node").forEach((node) => setNode(node, "locked"));
}

function bindResults() {
  $("btn-download").addEventListener("click", downloadReport);
  $("btn-error-back").addEventListener("click", () => {
    $("error-modal").classList.add("hidden");
  });
}

function downloadReport() {
  if (!resultData || !resultData.report_markdown) return;
  const req = resultData.request || {};
  const name = (req.role || "job") + "-finder-report";
  const blob = new Blob([resultData.report_markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = sanitizeFilename(name) + ".md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function sanitizeFilename(name) {
  return name.replace(/[^\w\- .]/g, "").replace(/\s+/g, "_").slice(0, 60) || "job";
}

/* ------------------------------------------------------------ error */
function showError(message, detail) {
  $("error-modal").classList.remove("hidden");
  $("error-msg").textContent = message;
  $("error-detail").textContent = detail || "No additional technical detail was provided.";
}