"use strict";
// ALICE's window: talks to the local back end (server.py) over one WebSocket.

const TOKEN = new URLSearchParams(location.search).get("token") || "";
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input");
const COMMANDS = {
  "/read": "PATH [question] - read a PDF, Word or PowerPoint file (or drop it on the window)",
  "/think": "on|off - step-by-step reasoning (slower, better at maths)",
  "/claude": "[question] - ask Claude: re-check the last answer, or a new question",
  "/memory": "what ALICE remembers about you",
  "/remember": "TEXT - add a note to long-term memory",
  "/forget": "ID|all - delete memory notes",
  "/arxiv": "auto on|off - look up named papers on arXiv automatically",
  "/tools": "list the tools",
  "/reset": "start a new conversation",
  "/help": "all commands",
};
const RUNNABLE = { python: "python", py: "python", c: "c", cpp: "cpp", "c++": "cpp", cc: "cpp", java: "java" };
const MATH_RX = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])/g;

let ws = null, state = { ready: false }, welcomed = false, current = null, thinkingEl = null;
let phase = "", phaseStart = performance.now(), renderTimer = null, blockSeq = 0;
let history = [], histPos = null, draft = "";

marked.setOptions({ gfm: true, breaks: false });

// ---------------------------------------------------------------- connection
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws?token=${encodeURIComponent(TOKEN)}`);
  ws.onopen = () => send({ type: "hello" });
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => setStatus("Lost the connection to ALICE. Close the window and start it again.");
}
function send(msg) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg)); }
setInterval(() => { if (!state.ready || state.busy || state.background) send({ type: "status" }); }, 2000);
setInterval(tickStatus, 250);

function handle(m) {
  switch (m.type) {
    case "status": applyStatus(m); break;
    case "info": if (m.clear) { log.innerHTML = ""; current = null; } addNotice("info", m.text); break;
    case "user": addUser(m.text); break;
    case "segment": finishSegment(); break;
    case "delta": appendDelta(m.text, m.target); break;
    case "reasoning": showThinking(m.chars); break;
    case "phase": phase = m.text; break;
    case "tool_start": finishSegment(); addLine("tool", `▸ ${m.name}: ${m.summary}`); break;
    case "tool_done": addLine("tool", `  ◂ ${m.mark}: ${short(m.output, 240)}`); break;
    case "notice": addNotice(m.kind, m.text); break;
    case "confirm": askRun(m); break;
    case "claude_preview": askClaude(m); break;
    case "turn_end": endTurn(m); break;
    case "run_result": showRunResult(m); break;
  }
}

// ---------------------------------------------------------------- status
function applyStatus(m) {
  const wasBusy = state.busy;
  state = { ...state, ...m };
  if (m.busy && !wasBusy) phaseStart = performance.now() - (m.elapsed || 0) * 1000;
  if (!m.ready && !m.error) phaseStart = performance.now() - (m.elapsed || 0) * 1000;
  if (m.phase !== undefined) phase = m.phase;
  const chips = [];
  if (m.model) chips.push(`<span class="chip"><b>${esc(m.model)}</b></span>`);
  if (m.tools) chips.push(`<span class="chip">${m.tools.length} tools</span>`);
  if (m.library !== null && m.library !== undefined) chips.push(`<span class="chip">library <b>${m.library}</b> docs</span>`);
  if (m.memory !== null && m.memory !== undefined) chips.push(`<span class="chip">memory <b>${m.memory}</b> notes</span>`);
  if (m.document) chips.push(`<span class="chip">open: <b>${esc(m.document.slice(0, 40))}</b></span>`);
  if (m.claude) chips.push(`<span class="chip">Claude on call</span>`);
  $("chips").innerHTML = chips.join("");
  $("thinking").checked = !!m.thinking;
  $("claude").classList.toggle("hidden", !m.claude);
  const usable = m.ready && !m.busy;
  input.disabled = !m.ready;
  $("send").disabled = !usable;
  $("send").classList.toggle("hidden", !!m.busy);
  $("stop").classList.toggle("hidden", !m.busy);
  input.placeholder = m.ready ? "Ask ALICE anything...  (/ for commands, Shift+Enter for a new line, drop files here)"
                              : "Loading the model...";
  if (m.error) addNotice("warning", `ALICE couldn't start: ${m.error}`);
  if (m.ready && !welcomed) { welcomed = true; welcome(m); input.focus(); }
  tickStatus();
}

function tickStatus() {
  const secs = Math.round((performance.now() - phaseStart) / 1000);
  if (state.error) return;
  if (!state.ready) {
    setStatus(`Loading the model... ${secs} s` + (secs > 15 ? "  (the first start after a reboot reads the whole model from disk)" : ""));
  } else if (state.busy) {
    setStatus(`${phase || "working"}... ${secs} s  ·  Esc or Stop to interrupt`);
  } else if (state.background) {
    setStatus(`${state.background}...`);
  } else {
    setStatus(state.lastStats || "Ready.");
  }
}
function setStatus(t) { $("status").textContent = t; }

function welcome(m) {
  const have = [`tools: ${m.tools.join(", ")}`];
  if (m.library) have.push(`your library: ${m.library} documents, searched automatically`);
  if (m.memory) have.push(`memory: ${m.memory} notes about you`);
  if (m.claude) have.push("Claude on call for hard questions");
  const el = div("welcome", `<h1>ALICE is ready.</h1><p>${esc(have.join(" · "))}</p>
    <p class="tips">Ask about code, coursework or papers. Drop a PDF, Word or PowerPoint file on the window to read it.
    <kbd>Enter</kbd> send · <kbd>Shift+Enter</kbd> new line · <kbd>↑</kbd>/<kbd>↓</kbd> earlier messages ·
    <kbd>Esc</kbd> stop · type <kbd>/</kbd> for commands. Code blocks have Copy and Run buttons.</p>`);
  log.prepend(el);
}

// ---------------------------------------------------------------- conversation
function div(cls, html) { const d = document.createElement("div"); d.className = cls; if (html !== undefined) d.innerHTML = html; return d; }
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function short(s, n) { s = String(s).replace(/\s+/g, " ").trim(); return s.length <= n ? s : s.slice(0, n - 3) + "..."; }
function atBottom() { return log.scrollHeight - log.scrollTop - log.clientHeight < 60; }
function add(el, force) { const follow = force || atBottom(); log.appendChild(el); if (follow) log.scrollTop = log.scrollHeight; return el; }

function addUser(text) {
  const el = div("msg user"); el.textContent = text; el.prepend(div("who", "you"));
  add(el, true);
}
function addLine(cls, text) { const el = div(cls); el.textContent = text; return add(el); }
function addNotice(kind, text) {
  const el = div(`notice ${kind}`); el.textContent = text;
  if (kind === "escalate") {
    const b = document.createElement("button"); b.textContent = "Ask Claude";
    b.onclick = () => send({ type: "send", text: "/claude" }); el.appendChild(b);
  }
  return add(el);
}
function showThinking(chars) {
  if (!thinkingEl) thinkingEl = addLine("thinking", "thinking");
  thinkingEl.textContent = "thinking" + ".".repeat(Math.min(40, 1 + Math.floor(chars / 300)));
}

function appendDelta(text, target) {
  if (!current) {
    const who = target === "claude" ? "Claude for ALICE" : "ALICE";
    const el = div(`msg ${target === "claude" ? "claude" : "alice"}`);
    const body = div("body");
    el.append(div("who", who), body);
    add(el);
    current = { el, body, raw: "" };
  }
  current.raw += text;
  if (!renderTimer) renderTimer = setTimeout(() => { renderTimer = null; if (current) render(current, false); }, 70);
}
function finishSegment() {
  if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
  if (current) render(current, true);
  current = null;
}
function render(seg, final) {
  const follow = atBottom();
  seg.body.innerHTML = renderMarkdown(seg.raw);
  enhanceCode(seg.body, final);
  if (follow) log.scrollTop = log.scrollHeight;
}

function renderMarkdown(src) {
  const maths = [];
  const parts = src.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(MATH_RX, (m, a, b, c, d) => {
      maths.push({ tex: a ?? b ?? c ?? d, display: a !== undefined || b !== undefined });
      return `@@MATH${maths.length - 1}@@`;
    });
  }
  let html = DOMPurify.sanitize(marked.parse(parts.join("")));
  return html.replace(/@@MATH(\d+)@@/g, (m, i) => {
    const { tex, display } = maths[+i];
    try { return katex.renderToString(tex, { displayMode: display, throwOnError: false }); }
    catch { return esc(tex); }
  });
}

function enhanceCode(root, final) {
  root.querySelectorAll("pre > code").forEach((code) => {
    const pre = code.parentElement;
    const langClass = [...code.classList].find((c) => c.startsWith("language-"));
    const lang = langClass ? langClass.slice(9).toLowerCase() : "";
    try { hljs.highlightElement(code); } catch { /* unknown language: leave plain */ }
    const box = div("codeblock"), bar = div("bar");
    bar.appendChild(Object.assign(document.createElement("span"), { className: "lang", textContent: lang || "code" }));
    if (final) {
      const copy = Object.assign(document.createElement("button"), { textContent: "Copy" });
      copy.onclick = () => navigator.clipboard.writeText(code.textContent).then(() => { copy.textContent = "Copied"; setTimeout(() => copy.textContent = "Copy", 1200); });
      bar.appendChild(copy);
      if (RUNNABLE[lang]) {
        const run = Object.assign(document.createElement("button"), { textContent: "Run" });
        run.onclick = () => toggleRunPanel(box, RUNNABLE[lang], code.textContent);
        bar.appendChild(run);
      }
    }
    pre.replaceWith(box); box.append(bar, pre);
  });
}

function toggleRunPanel(box, language, code) {
  let panel = box.querySelector(".runpanel");
  if (panel) { panel.remove(); return; }
  const id = `b${++blockSeq}`;
  panel = div("runpanel", `<textarea placeholder="Input for the program, if it reads any (one line per prompt)"></textarea>
    <div class="controls"><button class="primary">Run ${esc(language)}</button><span>runs on this computer, 20 s limit</span></div>`);
  panel.dataset.block = id;
  panel.querySelector("button").onclick = () => {
    const out = panel.querySelector("pre.out") || panel.appendChild(Object.assign(document.createElement("pre"), { className: "out" }));
    out.textContent = "running...";
    send({ type: "run", block: id, language, code, stdin: panel.querySelector("textarea").value });
  };
  box.appendChild(panel);
}
function showRunResult(m) {
  const out = document.querySelector(`.runpanel[data-block="${m.block}"] pre.out`);
  if (out) out.textContent = m.output;
}

function endTurn(m) {
  finishSegment();
  if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
  if (m.error) addNotice("warning", `error: ${m.error}`);
  for (const n of m.notices || []) addNotice(n.kind, n.text);
  if (m.stats) { state.lastStats = "last reply " + m.stats.replace(/^\[|\]$/g, ""); addLine("stats", m.stats.replace(/^\[|\]$/g, "")); }
}

// ---------------------------------------------------------------- dialogs
function dialog(title, bodyHtml, yes, no, onAnswer) {
  $("modal-title").textContent = title;
  $("modal-body").innerHTML = bodyHtml;
  $("modal-yes").textContent = yes; $("modal-no").textContent = no;
  $("modal").classList.remove("hidden");
  const done = (ok) => { $("modal").classList.add("hidden"); window.onkeydown = null; onAnswer(ok); input.focus(); };
  $("modal-yes").onclick = () => done(true);
  $("modal-no").onclick = () => done(false);
  window.onkeydown = (e) => { if (e.key === "Escape") { e.preventDefault(); done(false); } };
  $("modal-yes").focus();
}
function askRun(m) {
  const highlighted = (() => { try { return hljs.highlight(m.code, { language: m.language === "cpp" ? "cpp" : m.language }).value; } catch { return esc(m.code); } })();
  const stdin = m.stdin ? `<p>Input it will get:</p><pre>${esc(m.stdin)}</pre>` : "";
  dialog(`ALICE wants to run ${m.language} code`,
    `<p>It runs as you, on this computer, in a temporary folder (20 s limit).</p><pre><code>${highlighted}</code></pre>${stdin}`,
    "Run it", "Don't run", (ok) => send({ type: "confirm_reply", id: m.id, ok }));
}
function askClaude(m) {
  dialog("Ask Claude? This leaves your computer and goes to Anthropic's API.",
    `<p>${esc(m.summary)}</p><p>${esc(m.month)}</p><p>Exactly what is sent:</p><pre>${esc(m.preview)}</pre>`,
    "Send", "Cancel", (ok) => send({ type: "claude_reply", id: m.id, ok }));
}

// ---------------------------------------------------------------- composer
function submit() {
  const text = input.value.trim();
  if (!text || !state.ready) return;
  if (state.busy) { addNotice("info", "ALICE is still answering. Press Stop first."); return; }
  if (history[history.length - 1] !== text) history.push(text);
  histPos = null; input.value = ""; autosize(); showHints();
  send({ type: "send", text });
}
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 220) + "px"; }
function showHints() {
  const t = input.value;
  const word = t.startsWith("/") && !t.includes("\n") ? t.split(" ")[0] : "";
  const lines = word ? Object.entries(COMMANDS).filter(([c]) => c.startsWith(word)).map(([c, d]) => `${c}  ${d}`) : [];
  $("hints").textContent = lines.join("\n");
  $("hints").classList.toggle("hidden", !lines.length);
}
input.addEventListener("input", () => { autosize(); showHints(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); return; }
  if (e.key === "Escape" && state.busy) { send({ type: "stop" }); return; }
  const before = input.value.slice(0, input.selectionStart), after = input.value.slice(input.selectionEnd);
  if (e.key === "ArrowUp" && !before.includes("\n") && history.length) {
    e.preventDefault();
    if (histPos === null) { draft = input.value; histPos = history.length; }
    histPos = Math.max(0, histPos - 1); input.value = history[histPos]; autosize();
  } else if (e.key === "ArrowDown" && histPos !== null && !after.includes("\n")) {
    e.preventDefault();
    histPos += 1;
    if (histPos >= history.length) { histPos = null; input.value = draft; } else input.value = history[histPos];
    autosize();
  }
});
$("send").onclick = submit;
$("stop").onclick = () => send({ type: "stop" });
$("newchat").onclick = () => { if (state.busy) { addNotice("info", "Stop the current reply first."); return; } send({ type: "reset" }); };
$("thinking").onchange = (e) => send({ type: "thinking", on: e.target.checked });
$("claude").onclick = () => send({ type: "send", text: "/claude" });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.busy && $("modal").classList.contains("hidden")) send({ type: "stop" }); });

// ---------------------------------------------------------------- files
$("attach").onclick = () => $("file").click();
$("file").onchange = (e) => { for (const f of e.target.files) upload(f); e.target.value = ""; };
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { if ([...e.dataTransfer.types].includes("Files")) { dragDepth++; $("drop").classList.remove("hidden"); } });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").classList.add("hidden"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault(); dragDepth = 0; $("drop").classList.add("hidden");
  for (const f of e.dataTransfer.files) upload(f);
});
async function upload(file) {
  if (!state.ready) { addNotice("info", "ALICE is still loading; drop the file again in a moment."); return; }
  if (state.busy) { addNotice("info", "ALICE is still answering. Press Stop first, or drop the file when she's done."); return; }
  const form = new FormData(); form.append("file", file);
  addNotice("info", `uploading ${file.name}...`);
  try {
    const r = await fetch(`/upload?token=${encodeURIComponent(TOKEN)}`, { method: "POST", body: form });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    const question = input.value.trim(); input.value = ""; autosize();
    send({ type: "send", text: `/read "${j.path}"${question ? " " + question : ""}` });
  } catch (err) { addNotice("warning", `couldn't upload ${file.name}: ${err.message}`); }
}

connect();
