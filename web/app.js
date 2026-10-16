// VisiOS cockpit: renders the agent's live event stream.

const $ = (selector) => document.querySelector(selector);

const els = {
  app: $("#app"),
  thread: $("#thread"),
  welcome: $("#welcome"),
  composer: $("#composer"),
  prompt: $("#prompt"),
  send: $("#send-btn"),
  stop: $("#stop-btn"),
  hint: $("#composer-hint"),
  screen: $("#screen"),
  vnc: $("#vnc"),
  frame: $("#frame"),
  ghost: $("#ghost"),
  ghostTag: $("#ghost-tag"),
  callout: $("#callout"),
  overlay: document.querySelector(".overlay"),
  screenEmpty: $("#screen-empty"),
  screenEmptyTitle: $("#screen-empty-title"),
  screenEmptyText: $("#screen-empty-text"),
  narration: $("#narration-text"),
  phase: $("#phase-label"),
  progress: $("#progress"),
  progressFill: $("#progress-fill"),
  stepcount: $("#stepcount"),
  sandboxPill: $("#sandbox-pill"),
  sandboxState: $("#sandbox-state"),
  modelPill: $("#model-pill"),
  modelName: $("#model-name"),
  linkPill: $("#link-pill"),
  demoFlag: $("#demo-flag"),
  settings: $("#settings"),
  maxSteps: $("#max-steps"),
  maxStepsOut: $("#max-steps-out"),
  safetyHelp: $("#safety-help"),
  toast: $("#toast"),
};

const PHASES = {
  idle: "Prêt",
  booting: "Démarrage du bureau",
  observing: "Regarde l'écran",
  thinking: "Réfléchit",
  acting: "Agit",
  settling: "Vérifie le résultat",
  waiting_user: "Attend ta confirmation",
};

const RUN_STATUS = {
  running: "En cours",
  done: "Terminé",
  failed: "Échec",
  error: "Erreur",
  blocked: "Bloqué",
  loop_guard: "Bloqué dans une boucle",
  max_steps: "Limite d'étapes atteinte",
  stopped: "Arrêté",
};

const SAFETY_HELP = {
  balanced: "VisiOS te demande avant toute action sensible ou commande destructrice.",
  strict: "Les demandes sensibles sont bloquées et chaque commande shell attend ton accord.",
  fast: "Aucune confirmation. À réserver aux tâches sans risque.",
};

const THINKING_HELP = {
  full: "Le modèle réfléchit aussi longtemps qu'il veut. Seules les boucles sont coupées.",
  budget: "La réflexion est coupée si elle s'éternise ou tourne en rond, puis le modèle répond directement.",
  off: "Le modèle répond sans réfléchir : bien plus rapide, souvent suffisant pour cliquer et taper.",
};

const CUT_REASONS = {
  budget: "trop longue",
  loop: "elle tournait en rond",
};

const APP_LABELS = {
  browser: "le navigateur",
  terminal: "le terminal",
  files: "le gestionnaire de fichiers",
  text_editor: "l'éditeur de texte",
  settings: "les paramètres",
  calculator: "la calculatrice",
  writer: "le traitement de texte",
  calc: "le tableur",
};

const state = {
  ws: null,
  retry: 0,
  config: null,
  settings: { max_steps: 25, safety_mode: "balanced", display_mode: "vnc", thinking: "budget" },
  running: false,
  replaying: false,
  sandbox: "unknown",
  runs: new Map(),
  currentRun: null,
  calloutTimer: null,
  ghostTimer: null,
  statusTimer: null,
};

// ------------------------------------------------------------------ utils
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function clip(value, limit) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function nearBottom() {
  const t = els.thread;
  return t.scrollHeight - t.scrollTop - t.clientHeight < 120;
}

function scrollThread(force = false) {
  if (force || nearBottom()) {
    requestAnimationFrame(() => { els.thread.scrollTop = els.thread.scrollHeight; });
  }
}

function toast(message) {
  els.toast.textContent = message;
  els.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { els.toast.hidden = true; }, 5000);
}

function formatDuration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")}`;
}

function keycaps(keys) {
  return String(keys || "")
    .split(/\s+/)
    .filter(Boolean)
    .map((combo) => combo.split("+").map((k) => `<span class="k">${escapeHtml(k)}</span>`).join("+"))
    .join(" ");
}

// Pull the "thought" string out of a JSON answer that is still streaming.
function partialField(buffer, field) {
  const match = buffer.match(new RegExp(`"${field}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)`));
  if (!match) return "";
  try {
    return JSON.parse(`"${match[1].replace(/\\$/, "")}"`);
  } catch {
    return match[1];
  }
}

// ----------------------------------------------------------------- labels
function actionTitle(a) {
  const text = a.text || "";
  switch (a.type) {
    case "click": return "Clic";
    case "double_click": return "Double-clic";
    case "right_click": return "Clic droit";
    case "move": return "Survol";
    case "type": return `Saisit « ${escapeHtml(clip(text.replace(/<ENTER>/gi, ""), 60))} »${/<ENTER>/i.test(text) || (a.summary || "").includes("+ Enter") ? " puis Entrée" : ""}`;
    case "hotkey": return `Appuie sur <code>${escapeHtml(text)}</code>`;
    case "scroll": return /up/.test(a.summary || "") ? "Défile vers le haut" : "Défile vers le bas";
    case "drag": return "Glisser-déposer";
    case "open_app": return `Ouvre ${escapeHtml(APP_LABELS[(a.app || text).toLowerCase()] || a.app || text)}`;
    case "open_url": return `Ouvre <code>${escapeHtml(clip(a.url || text, 60))}</code>`;
    case "run_command": return `Exécute <code>${escapeHtml(clip(a.command || text, 70))}</code>`;
    case "wait": return "Patiente pendant le chargement";
    case "done": return "Objectif atteint";
    case "fail": return "Abandonne";
    default: return "Réponse illisible";
  }
}

function calloutHtml(a) {
  const text = a.text || "";
  switch (a.type) {
    case "type": return `${escapeHtml(clip(text.replace(/<ENTER>/gi, " ⏎"), 90))}<span class="caret"></span>`;
    case "hotkey": return keycaps(text);
    case "open_url": return `<code>${escapeHtml(clip(a.url || text, 80))}</code>`;
    case "run_command": return `<code>$ ${escapeHtml(clip(a.command || text, 80))}</code>`;
    case "scroll": return /up/.test(a.summary || "") ? "Défile ↑" : "Défile ↓";
    case "open_app": return `Ouvre ${escapeHtml(APP_LABELS[(a.app || text).toLowerCase()] || a.app || text)}`;
    case "wait": return "Patiente…";
    default: return "";
  }
}

// ----------------------------------------------------------------- screen
function setRatio(size) {
  if (!size || !size[0] || !size[1]) return;
  els.screen.style.setProperty("--ratio", (size[0] / size[1]).toFixed(4));
  document.documentElement.style.setProperty("--ratio", (size[0] / size[1]).toFixed(4));
}

function applyDisplayMode() {
  const vncMode = state.settings.display_mode !== "screenshots";
  els.vnc.hidden = !vncMode;
  els.frame.hidden = vncMode || !els.frame.src;
  if (vncMode && state.config && state.sandbox === "running" && !els.vnc.src) {
    els.vnc.src = state.config.vnc_url;
  }
  if (!vncMode && state.sandbox === "running" && !els.frame.src) refreshFrame();
  updateScreenEmpty();
}

async function refreshFrame() {
  try {
    const res = await fetch("/api/frame");
    const data = await res.json();
    if (data.image) showFrame(data.image);
  } catch { /* desktop not ready */ }
}

function showFrame(src) {
  els.frame.src = src;
  if (state.settings.display_mode === "screenshots") els.frame.hidden = false;
}

function updateScreenEmpty() {
  const off = state.sandbox !== "running";
  const show = off && !state.running;
  els.screenEmpty.hidden = !show;
  if (state.sandbox === "unavailable") {
    els.screenEmptyTitle.textContent = "Docker ne répond pas";
    els.screenEmptyText.textContent = "Lance Docker Desktop, puis démarre le bureau.";
  } else {
    els.screenEmptyTitle.textContent = "Le bureau est éteint";
    els.screenEmptyText.textContent = "Démarre-le pour que VisiOS puisse agir. Il démarre aussi tout seul à ta première demande.";
  }
}

function moveGhost(point, label) {
  if (!point) return;
  const [fx, fy] = point;
  els.ghost.style.setProperty("--x", `${(fx * 100).toFixed(2)}%`);
  els.ghost.style.setProperty("--y", `${(fy * 100).toFixed(2)}%`);
  els.ghost.classList.add("visible");
  els.ghostTag.textContent = label || "VisiOS";
  clearTimeout(state.ghostTimer);
}

function hideGhostSoon(delay = 2500) {
  clearTimeout(state.ghostTimer);
  state.ghostTimer = setTimeout(() => els.ghost.classList.remove("visible"), delay);
}

function burst(className, point, lifetime = 800) {
  if (!point || state.replaying) return;
  const node = el("span", className);
  node.style.left = `${point[0] * 100}%`;
  node.style.top = `${point[1] * 100}%`;
  els.overlay.appendChild(node);
  setTimeout(() => node.remove(), lifetime);
}

function showCallout(html, persist = false) {
  clearTimeout(state.calloutTimer);
  if (!html) { els.callout.classList.remove("visible"); return; }
  els.callout.innerHTML = html;
  els.callout.classList.add("visible");
  if (!persist) state.calloutTimer = setTimeout(() => els.callout.classList.remove("visible"), 2600);
}

// -------------------------------------------------------------- narration
function narrate(text) {
  if (state.replaying || !text) return;
  els.narration.textContent = text;
}

function setPhase(phase, step) {
  els.app.dataset.phase = phase;
  els.phase.textContent = PHASES[phase] || phase;
  const run = state.currentRun;
  if (run && step) {
    els.stepcount.textContent = `Étape ${step} sur ${run.maxSteps} max`;
    els.progress.hidden = false;
    els.progressFill.style.width = `${Math.min(100, (step / run.maxSteps) * 100)}%`;
  }
}

function setRunning(running) {
  state.running = running;
  els.app.dataset.running = String(running);
  els.stop.hidden = !running;
  els.send.textContent = running ? "Préciser" : "Envoyer";
  els.prompt.placeholder = running
    ? "Ajoute une précision, VisiOS la lira à l'étape suivante"
    : "Que veux-tu que VisiOS fasse ?";
  els.hint.textContent = running
    ? "Ta précision sera prise en compte sans arrêter la tâche"
    : "Entrée pour envoyer, Maj+Entrée pour aller à la ligne";
  if (!running) {
    els.phase.textContent = PHASES.idle;
    els.progress.hidden = true;
    els.stepcount.textContent = "";
    hideGhostSoon();
  }
  updateScreenEmpty();
}

// ----------------------------------------------------------------- thread
function hideWelcome() {
  els.welcome.hidden = true;
}

function addUserMessage(text) {
  hideWelcome();
  els.thread.appendChild(el("div", "msg-user", text));
  scrollThread(true);
}

function addNote(text) {
  const node = el("div", "msg-note");
  node.innerHTML = `<strong>Précision :</strong> ${escapeHtml(text)}`;
  els.thread.appendChild(node);
  scrollThread(true);
}

function createRun(ev) {
  hideWelcome();
  const root = el("section", "run");
  root.dataset.status = "running";
  const head = el("div", "run-head");
  const stateLabel = el("span", "run-state", RUN_STATUS.running);
  const time = el("span", "run-time", "0 s");
  head.append(stateLabel, time);
  const steps = el("ol", "steps");
  root.append(head, steps);
  els.thread.appendChild(root);

  const run = {
    id: ev.run_id,
    root,
    stateLabel,
    time,
    stepsEl: steps,
    steps: new Map(),
    maxSteps: ev.max_steps || state.settings.max_steps,
    startedAt: (ev.ts || Date.now() / 1000) * 1000,
    timer: null,
  };
  if (!state.replaying) {
    run.timer = setInterval(() => { run.time.textContent = formatDuration(Date.now() - run.startedAt); }, 1000);
  }
  state.runs.set(ev.run_id, run);
  state.currentRun = run;
  scrollThread(true);
  return run;
}

function getRun(ev) {
  return state.runs.get(ev.run_id) || state.currentRun || createRun(ev);
}

function createStep(run, number) {
  const li = el("li", "step");
  li.dataset.state = "active";
  const num = el("span", "step-num", String(number));
  const main = el("div", "step-main");
  const title = el("div", "step-title", "Observe l'écran");
  const live = el("p", "step-live");
  main.append(title, live);
  li.append(num, main);
  run.stepsEl.appendChild(li);
  const step = {
    li, main, title, live,
    buffer: "", // answer text (JSON), without any inline <think> block
    raw: "", // everything streamed as "text"
    thinking: "", // reasoning_content / message.thinking deltas
    think: null, // thinking block UI, created on first reasoning token
    thumbnail: null,
    action: null,
  };
  run.steps.set(number, step);
  scrollThread();
  return step;
}

function getStep(ev) {
  const run = getRun(ev);
  return run.steps.get(ev.step) || createStep(run, ev.step);
}

function fillStep(step, action) {
  step.action = action;
  step.title.innerHTML = actionTitle(action);
  step.live.remove();
  const thought = action.thought || action.observation;
  if (thought) step.main.appendChild(el("p", "step-thought", thought));

  const details = el("details");
  details.appendChild(el("summary", null, "Détails"));
  const body = el("div", "step-detail");
  if (step.thumbnail) {
    const img = el("img");
    img.src = step.thumbnail;
    img.alt = "Écran vu par VisiOS à cette étape";
    img.loading = "lazy";
    body.appendChild(img);
  }
  const raw = [action.observation && `Observation : ${action.observation}`, action.raw && `Réponse brute :\n${action.raw}`]
    .filter(Boolean)
    .join("\n\n");
  if (raw) body.appendChild(el("pre", null, raw));
  details.appendChild(body);
  step.details = details;
  step.main.appendChild(details);
}

function finishStep(step, ev) {
  const status = ev.status;
  let tone = "ok";
  if (status === "invalid" || ev.screen_changed === false) tone = "warn";
  if (status === "failed" || status === "blocked") tone = "bad";
  if (status === "noop") tone = "warn";
  step.li.dataset.state = tone;

  let flag = "";
  if (status === "invalid") flag = "Le modèle n'a pas donné d'action exploitable, nouvel essai.";
  else if (status === "failed" && step.action?.type !== "fail") flag = ev.detail || "L'action a échoué.";
  else if (status === "blocked") flag = ev.detail || "Action bloquée.";
  else if (status === "noop") flag = ev.detail;
  else if (ev.screen_changed === false && !["wait", "move", "run_command"].includes(step.action?.type)) flag = "Aucun effet visible, VisiOS va essayer autrement.";
  if (flag) {
    const node = el("p", `step-flag${tone === "bad" ? " bad" : ""}`, flag);
    step.main.insertBefore(node, step.details || null);
  }
}

function addConfirm(ev) {
  const box = el("div", "confirm");
  const what = el("p");
  what.innerHTML = `VisiOS veut : <strong>${actionTitle(ev.action)}</strong>`;
  const why = el("p", "why", ev.reason || "Action sensible.");
  const row = el("div", "row");
  const yes = el("button", "btn ok", "Autoriser");
  const no = el("button", "btn", "Refuser");
  yes.type = no.type = "button";
  const answer = (approved) => {
    send({ type: "confirm", approved });
    row.replaceWith(el("p", "why", approved ? "Autorisé." : "Refusé."));
  };
  yes.addEventListener("click", () => answer(true));
  no.addEventListener("click", () => answer(false));
  row.append(yes, no);
  box.append(what, why, row);
  els.thread.appendChild(box);
  if (state.replaying) row.replaceWith(el("p", "why", "Demande expirée."));
  else yes.focus({ preventScroll: true });
  scrollThread(true);
  return box;
}

function finishRun(ev) {
  const run = getRun(ev);
  clearInterval(run.timer);
  run.root.dataset.status = ev.status;
  run.stateLabel.textContent = RUN_STATUS[ev.status] || ev.status;
  run.time.textContent = formatDuration((ev.ts || Date.now() / 1000) * 1000 - run.startedAt);
  const muted = ev.status !== "done";
  els.thread.appendChild(el("div", `msg-agent${muted ? " muted" : ""}`, ev.reason || RUN_STATUS[ev.status]));
  if (!state.replaying) {
    narrate(ev.status === "done" ? ev.reason : `${RUN_STATUS[ev.status] || ev.status}. ${ev.reason || ""}`);
    showCallout("");
  }
  state.currentRun = null;
  scrollThread(true);
}

function clearThread() {
  for (const run of state.runs.values()) clearInterval(run.timer);
  state.runs.clear();
  state.currentRun = null;
  [...els.thread.children].forEach((child) => { if (child !== els.welcome) child.remove(); });
  els.welcome.hidden = false;
}

// ----------------------------------------------------------------- events
function handle(ev) {
  switch (ev.type) {
    case "hello": return onHello(ev);
    case "status": return onStatus(ev);
    case "settings": state.settings = { ...state.settings, ...ev.settings }; return syncSettingsForm();
    case "user_message": return addUserMessage(ev.text);
    case "user_note": return addNote(ev.text);
    case "run_started":
      createRun(ev);
      if (!state.replaying) { setRunning(true); narrate("Je regarde l'écran…"); }
      return;
    case "phase":
      if (!state.replaying) setPhase(ev.phase, ev.step);
      return;
    case "step_started":
      createStep(getRun(ev), ev.step);
      return;
    case "observation": {
      const step = getStep(ev);
      step.thumbnail = ev.thumbnail;
      setRatio(ev.screen_size);
      // The agent just captured the screen, so the desktop is up even if the
      // last status poll predates its (auto) start.
      if (!state.replaying && state.sandbox !== "running") {
        markSandboxRunning();
      }
      return;
    }
    case "model_stream": return onStream(ev);
    case "action_planned": return onPlanned(ev);
    case "confirmation_required":
      addConfirm(ev);
      if (!state.replaying) narrate("J'attends ton accord avant de continuer.");
      return;
    case "step_finished": return onStepFinished(ev);
    case "run_finished":
      finishRun(ev);
      if (!state.replaying) setRunning(false);
      return;
    case "frame": return showFrame(ev.image);
    case "reset": return clearThread();
    case "error": return toast(ev.message);
    case "log": return console.debug("[visios]", ev.message);
    default: return undefined;
  }
}

function onHello(ev) {
  state.config = ev.config;
  state.settings = { ...state.settings, ...ev.config.settings };
  els.demoFlag.hidden = !ev.config.demo;
  els.modelName.textContent = ev.config.model;
  $("#info-model").textContent = ev.config.model;
  $("#info-api").textContent = ev.config.api_mode;
  $("#info-coords").textContent = ev.config.coord_mode === "pixel" ? "pixels de l'image" : "grille 0–1000";
  setRatio(ev.config.screen);
  syncSettingsForm();

  clearThread();
  state.replaying = true;
  try {
    for (const past of ev.history || []) handle(past);
  } finally {
    state.replaying = false;
  }
  // A run that was live before the reconnect keeps going.
  setRunning(Boolean(ev.running));
  if (!ev.running && state.currentRun) state.currentRun = null;
  applyDisplayMode();
  scrollThread(true);
}

function onStatus(ev) {
  state.sandbox = ev.sandbox;
  const labels = { running: "allumé", exited: "éteint", missing: "éteint", created: "en création", unavailable: "Docker absent" };
  els.sandboxState.textContent = labels[ev.sandbox] || ev.sandbox;
  els.sandboxPill.dataset.tone = ev.sandbox === "running" ? "ok" : ev.sandbox === "unavailable" ? "bad" : "idle";
  els.sandboxPill.title = ev.sandbox_error || "";
  els.modelPill.dataset.tone = ev.model_ok ? "ok" : "bad";
  els.modelPill.title = ev.model_ok ? "Le serveur du modèle répond" : `Serveur du modèle injoignable : ${ev.model_detail || ""}`;
  if (!ev.running && state.running) setRunning(false);
  applyDisplayMode();
}

function markSandboxRunning() {
  state.sandbox = "running";
  els.sandboxState.textContent = "allumé";
  els.sandboxPill.dataset.tone = "ok";
  applyDisplayMode();
}

// ------------------------------------------------------------- thinking
// Reasoning of thinking models, streamed token by token into a collapsible block.
function wordCount(text) {
  return (text.match(/\S+/g) || []).length;
}

function formatSeconds(ms) {
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1).replace(".", ",")} s` : formatDuration(ms);
}

function createThink(step, open = true) {
  const box = el("details", "think");
  box.open = open;
  const summary = el("summary", "think-head");
  const label = el("span", "think-label", "Réfléchit");
  const meta = el("span", "think-meta", "");
  summary.append(label, meta);
  const body = el("div", "think-body");
  box.append(summary, body);
  step.main.insertBefore(box, step.live.isConnected ? step.live : step.main.children[1] || null);
  const think = {
    box, label, meta, body,
    shown: 0, // characters already appended to body
    pending: "",
    frame: 0,
    startedAt: performance.now(),
    timer: null,
    done: false,
    stick: true, // follow the end of the text unless the user scrolled up
  };
  body.addEventListener("scroll", () => {
    think.stick = body.scrollHeight - body.scrollTop - body.clientHeight < 24;
  });
  if (open && !state.replaying) {
    think.timer = setInterval(() => {
      think.meta.textContent = formatSeconds(performance.now() - think.startedAt);
    }, 200);
  }
  step.think = think;
  return think;
}

function thinkingText(step) {
  // Some servers stream the reasoning inline as <think>…</think> in the content.
  const open = step.raw.indexOf("<think>");
  let inline = "";
  if (open !== -1) {
    const close = step.raw.indexOf("</think>", open);
    inline = close === -1 ? step.raw.slice(open + 7) : step.raw.slice(open + 7, close);
    step.buffer = close === -1 ? "" : step.raw.slice(close + 8);
  } else {
    step.buffer = step.raw;
  }
  return (step.thinking + inline).replace(/^\s+/, "");
}

function pushThinking(step) {
  const text = thinkingText(step);
  if (!text) return;
  const think = step.think || createThink(step);
  if (think.done) return;
  if (text.length > think.shown) {
    think.pending += text.slice(think.shown);
    think.shown = text.length;
  }
  if (!think.frame) think.frame = requestAnimationFrame(() => flushThinking(think));
  const sentence = text.split(/(?<=[.!?:])\s+/).filter((part) => part.trim().length > 12).pop();
  if (sentence && !partialField(step.buffer, "thought")) narrate(clip(sentence, 170));
}

function flushThinking(think) {
  think.frame = 0;
  if (!think.pending) return;
  const span = el("span", "tok", think.pending);
  think.pending = "";
  think.body.appendChild(span);
  // Keep the DOM small on very long reasoning: fold old fragments into plain text.
  if (think.body.childNodes.length > 300) {
    const text = think.body.textContent;
    think.body.textContent = text;
  }
  if (think.stick) think.body.scrollTop = think.body.scrollHeight;
  scrollThread();
}

function finishThinking(step, fullText, thinkMs) {
  let think = step.think;
  const text = (fullText || "").trim();
  if (!think && !text) return;
  if (!think) {
    think = createThink(step, false);
    think.body.textContent = text;
  } else {
    if (think.frame) cancelAnimationFrame(think.frame);
    if (think.pending) think.body.appendChild(el("span", "tok", think.pending));
    think.pending = "";
    // Deltas can be missed (late join, reconnection): trust the final text.
    if (text && !think.cut && Math.abs(think.body.textContent.trim().length - text.length) > 40) think.body.textContent = text;
    think.box.open = false;
  }
  clearInterval(think.timer);
  think.done = true;
  think.box.classList.add("is-done");
  const words = wordCount(think.body.textContent);
  const elapsed = thinkMs || performance.now() - think.startedAt;
  think.label.textContent = think.cut ? "Réflexion coupée" : "A réfléchi";
  think.meta.textContent = `${formatSeconds(elapsed)} · ${words} mot${words > 1 ? "s" : ""}`;
}

function onStream(ev) {
  const step = getStep(ev);
  if (ev.kind === "thinking") {
    step.thinking += ev.delta;
    pushThinking(step);
  } else if (ev.kind === "text") {
    step.raw += ev.delta;
    pushThinking(step);
    const thought = partialField(step.buffer, "thought") || partialField(step.buffer, "observation");
    if (thought) {
      step.live.textContent = thought;
      narrate(thought);
    }
  } else if (ev.kind === "status" && ev.delta.startsWith("thinking_cut")) {
    const reason = CUT_REASONS[ev.delta.split(":")[1]] || "trop longue";
    const think = step.think || createThink(step);
    think.cut = true;
    think.box.classList.add("is-cut");
    think.label.textContent = "Réflexion coupée";
    if (think.frame) { cancelAnimationFrame(think.frame); think.frame = 0; }
    if (think.pending) { think.body.appendChild(el("span", "tok", think.pending)); think.pending = ""; }
    think.body.appendChild(el("span", "think-cut", `Réflexion coupée : ${reason}. VisiOS demande une réponse directe.`));
    // Whatever the follow-up streams goes after the marker.
    think.shown = thinkingText(step).length;
    narrate("La réflexion s'éternisait, je demande une réponse directe.");
  } else if (ev.kind === "status" && ev.delta === "retry") {
    step.raw = "";
    step.buffer = "";
    step.live.textContent = "Réponse illisible, nouvelle tentative…";
    if (step.think && !step.think.done) {
      step.thinking += "\n\n— Nouvel essai —\n\n";
      pushThinking(step);
    }
  }
  scrollThread();
}

function onPlanned(ev) {
  const step = getStep(ev);
  const action = ev.action;
  finishThinking(step, action.thinking, ev.think_ms);
  fillStep(step, action);
  if (state.replaying) return;
  if (action.thought) narrate(action.thought);
  if (action.point) {
    moveGhost(action.point, action.type === "type" ? "saisie" : "VisiOS");
    burst("target", action.point, 900);
  }
  const html = calloutHtml(action);
  if (html) showCallout(html, true);
  scrollThread();
}

function onStepFinished(ev) {
  const run = getRun(ev);
  const step = run.steps.get(ev.step);
  if (!step) return;
  finishStep(step, ev);
  if (state.replaying) return;
  const action = step.action || {};
  if (ev.executed && ["click", "double_click", "right_click", "type"].includes(action.type) && action.point) {
    burst("ripple", action.point);
    if (action.type === "double_click") setTimeout(() => burst("ripple", action.point), 140);
  }
  if (action.type === "drag" && action.end_point) moveGhost(action.end_point);
  showCallout(els.callout.classList.contains("visible") ? els.callout.innerHTML.replace('<span class="caret"></span>', "") : "", false);
}

// -------------------------------------------------------------- transport
function send(message) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(message));
    return true;
  }
  toast("Connexion au serveur perdue, nouvelle tentative en cours.");
  return false;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.addEventListener("open", () => {
    state.retry = 0;
    els.linkPill.hidden = true;
  });
  ws.addEventListener("message", (msg) => {
    try { handle(JSON.parse(msg.data)); } catch (err) { console.error(err); }
  });
  ws.addEventListener("close", () => {
    els.linkPill.hidden = false;
    els.linkPill.dataset.tone = "warn";
    const delay = Math.min(8000, 500 * 2 ** state.retry++);
    setTimeout(connect, delay);
  });
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    onStatus(await res.json());
  } catch { /* server restarting */ }
}

async function pollStatus() {
  clearTimeout(state.statusTimer);
  if (!state.running) await refreshStatus();
  // Poll faster while the desktop is not up yet, so it appears as soon as it boots.
  state.statusTimer = setTimeout(pollStatus, state.sandbox === "running" ? 15000 : 3000);
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") pollStatus();
});

async function sandboxAction(action) {
  try {
    if (action === "start") { els.sandboxState.textContent = "démarrage…"; }
    const res = await fetch(`/api/sandbox/${action}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) toast(data.error || "Action impossible.");
    else onStatus(data);
    if (action === "stop") { els.vnc.removeAttribute("src"); els.frame.removeAttribute("src"); applyDisplayMode(); }
  } catch {
    toast("Le serveur VisiOS ne répond pas.");
  }
}

// --------------------------------------------------------------- composer
function submitPrompt() {
  const text = els.prompt.value.trim();
  if (!text) return;
  const ok = state.running
    ? send({ type: "note", text })
    : send({ type: "run", prompt: text, settings: state.settings });
  if (ok) {
    els.prompt.value = "";
    autosize();
  }
}

function autosize() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 180)}px`;
}

els.composer.addEventListener("submit", (event) => { event.preventDefault(); submitPrompt(); });
els.prompt.addEventListener("input", autosize);
els.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submitPrompt();
  }
});
els.stop.addEventListener("click", () => send({ type: "stop" }));
$("#badge-stop").addEventListener("click", () => send({ type: "stop" }));
$("#screen-start").addEventListener("click", () => sandboxAction("start"));
$("#suggestions").addEventListener("click", (event) => {
  if (event.target instanceof HTMLButtonElement) {
    els.prompt.value = event.target.textContent;
    autosize();
    els.prompt.focus();
  }
});

// --------------------------------------------------------------- settings
function syncSettingsForm() {
  els.maxSteps.value = state.settings.max_steps;
  els.maxStepsOut.textContent = state.settings.max_steps;
  const safety = document.querySelector(`input[name="safety"][value="${state.settings.safety_mode}"]`);
  if (safety) safety.checked = true;
  els.safetyHelp.textContent = SAFETY_HELP[state.settings.safety_mode] || "";
  const thinking = document.querySelector(`input[name="thinking"][value="${state.settings.thinking}"]`);
  if (thinking) thinking.checked = true;
  $("#thinking-help").textContent = THINKING_HELP[state.settings.thinking] || "";
  const display = document.querySelector(`input[name="display"][value="${state.settings.display_mode}"]`);
  if (display) display.checked = true;
}

function pushSettings() {
  send({ type: "settings", settings: state.settings });
}

$("#open-settings").addEventListener("click", () => els.settings.showModal());
els.maxSteps.addEventListener("input", () => {
  state.settings.max_steps = Number(els.maxSteps.value);
  els.maxStepsOut.textContent = els.maxSteps.value;
});
els.maxSteps.addEventListener("change", pushSettings);
document.querySelectorAll('input[name="safety"]').forEach((input) => input.addEventListener("change", () => {
  state.settings.safety_mode = input.value;
  els.safetyHelp.textContent = SAFETY_HELP[input.value] || "";
  pushSettings();
}));
document.querySelectorAll('input[name="thinking"]').forEach((input) => input.addEventListener("change", () => {
  state.settings.thinking = input.value;
  $("#thinking-help").textContent = THINKING_HELP[input.value] || "";
  pushSettings();
}));
document.querySelectorAll('input[name="display"]').forEach((input) => input.addEventListener("change", () => {
  state.settings.display_mode = input.value;
  pushSettings();
  applyDisplayMode();
}));
$("#sandbox-start").addEventListener("click", () => sandboxAction("start"));
$("#sandbox-stop").addEventListener("click", () => sandboxAction("stop"));
$("#clear-thread").addEventListener("click", () => {
  if (state.running) { toast("Arrête la tâche en cours avant d'effacer la conversation."); return; }
  send({ type: "reset" });
  els.settings.close();
});

connect();
pollStatus();
els.prompt.focus();
