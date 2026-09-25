/* Prompt-to-Drive web UI client.
 * Talks to webui/server.py over one WebSocket. Every message is either
 *   {type:"event", stage, payload}  - a pipeline stage report
 *   {type:"frame", iteration, jpeg} - a base64 JPEG from the Engine camera
 *   {type:"hello"|"error", ...}
 */
(() => {
  "use strict";

  const STAGES = ["generator", "engine", "verifier", "reflector"];
  const PRED_LABELS = {
    weather_correct: "Weather matches the prompt",
    pedestrian_present: "Pedestrian present",
    vehicles_present: "Vehicles present",
    event_observable: "Event is visible",
  };
  const STATUS_TEXT = {
    passed: ["Scenario verified", "The simulation matches the prompt."],
    iteration_limit_reached: ["Not verified", "The loop ran out of iterations before every check passed."],
    generator_only_complete: ["Generated without the loop", "Verification ran once; no reflection."],
  };
  const MAX_FRAMES_PER_ITER = 30;

  const $ = (id) => document.getElementById(id);
  const token = new URLSearchParams(location.search).get("token");
  const tokenQs = token ? `?token=${encodeURIComponent(token)}` : "";

  const el = {
    form: $("prompt-form"), prompt: $("prompt"), count: $("char-count"),
    gen: $("generate-btn"), clear: $("clear-btn"), examples: $("examples"),
    tabs: $("iter-tabs"), lane: $("lane"), frame: $("frame"), empty: $("screen-empty"),
    liveTag: $("live-tag"), film: $("filmstrip"), verdict: $("verdict"), log: $("log"),
    toast: $("toast"), mode: $("mode-badge"), carla: $("carla-badge"), backend: $("backend-badge"),
  };

  let config = null;
  let ws = null;
  let retry = 0;
  let state = freshState();

  function freshState() {
    return { running: false, iters: {}, selected: 1, follow: true, final: null, frameSel: null };
  }
  function iter(n) {
    if (!state.iters[n]) {
      state.iters[n] = { frames: [], st: Object.fromEntries(STAGES.map((s) => [s, { state: "pending", data: {} }])) };
    }
    return state.iters[n];
  }

  // ---------- tiny DOM helper (always textContent, never innerHTML) ----------
  function h(tag, attrs = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  const secs = (s) => (s == null ? null : `${Number(s).toFixed(1)} s`);
  const pct = (x) => (x == null ? "n/a" : `${Math.round(x * 100)}%`);

  // ---------- config + badges ----------
  async function loadConfig() {
    try {
      const r = await fetch(`/api/config${tokenQs}`);
      if (r.status === 401) { toast("This UI needs an access token. Open the link that includes ?token=…"); return; }
      config = await r.json();
    } catch { return; }
    if (config.mode === "replay") {
      el.mode.textContent = "Replay of recorded runs";
      el.mode.className = "badge replay";
      el.prompt.placeholder = "Pick a recorded prompt below, then Generate to replay it.";
    } else {
      el.mode.textContent = "Live CARLA";
      el.mode.className = "badge";
      el.carla.hidden = false;
      el.carla.textContent = config.carla.reachable ? `CARLA ${config.carla.host}:${config.carla.port}` : "CARLA not reachable";
      el.carla.className = `badge ${config.carla.reachable ? "ok" : "bad"}`;
      el.backend.hidden = false;
      el.backend.textContent = config.verifier_backend && config.verifier_backend !== config.backend
        ? `Generator ${config.backend}, Verifier ${config.verifier_backend}` : `Backend ${config.backend}`;
    }
    renderExamples();
  }

  function renderExamples() {
    el.examples.replaceChildren(...(config?.examples || []).map((ex) =>
      h("button", { type: "button", title: ex.prompt, onclick: () => { el.prompt.value = ex.prompt; updateCount(); el.prompt.focus(); } },
        ex.prompt)));
  }

  // ---------- socket ----------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws${tokenQs}`);
    ws.onopen = () => { retry = 0; };
    ws.onmessage = (ev) => handle(JSON.parse(ev.data));
    ws.onclose = (ev) => {
      if (ev.code === 4401) { toast("Access token missing or wrong."); return; }
      el.mode.textContent = "Reconnecting…";
      const wait = Math.min(5000, 1000 + retry++ * 500); // backoff, as in the old socket UI
      setTimeout(() => { connect(); loadConfig(); }, wait);
    };
  }

  function handle(msg) {
    if (msg.type === "hello") {
      state = freshState();
      for (const e of msg.history || []) onEvent(e.stage, e.payload, e.t, true);
      if (msg.last_frame) onFrame(msg.last_frame);
      state.running = !!msg.busy;
      render();
    } else if (msg.type === "event") {
      onEvent(msg.stage, msg.payload, msg.t);
      render();
    } else if (msg.type === "frame") {
      onFrame(msg);
    } else if (msg.type === "error") {
      toast(msg.message);
    }
  }

  // ---------- pipeline events ----------
  function onEvent(stage, p, t, quiet) {
    log(stage, p, t);
    const n = p.iteration;
    const setStage = (s, st, data) => { const it = iter(n); it.st[s].state = st; Object.assign(it.st[s].data, data || {}); };
    switch (stage) {
      case "run_start":
        state = freshState(); state.running = true; state.prompt = p.prompt;
        el.log.replaceChildren(); log(stage, p, t);
        break;
      case "generator": setStage("generator", "active", p); if (state.follow) state.selected = n; break;
      case "generator_done": setStage("generator", "done", p); break;
      case "engine": setStage("engine", "active", p); break;
      case "engine_done": setStage("engine", p.success ? "done" : "fail", p); break;
      case "verifier": setStage("verifier", "active", p); break;
      case "verifier_done": setStage("verifier", p.passed ? "done" : "fail", p); break;
      case "reflector": setStage("reflector", "active", p); break;
      case "reflector_done": setStage("reflector", "done", p); break;
      case "done": {
        state.running = false; state.final = p;
        const last = state.iters[p.iterations];
        if (last && last.st.reflector.state === "pending") last.st.reflector.state = "skipped";
        break;
      }
      case "error": {
        state.running = false;
        for (const it of Object.values(state.iters)) for (const s of STAGES) if (it.st[s].state === "active") it.st[s].state = "fail";
        if (!quiet) toast(p.message);
        break;
      }
    }
  }

  function onFrame(msg) {
    const it = iter(msg.iteration || 1);
    const src = `data:image/jpeg;base64,${msg.jpeg}`;
    it.frames.push(src);
    if (it.frames.length > MAX_FRAMES_PER_ITER) it.frames.shift();
    if (state.selected === (msg.iteration || 1)) { state.frameSel = null; renderCamera(); }
  }

  // ---------- rendering ----------
  function render() {
    el.gen.disabled = state.running;
    el.gen.textContent = state.running ? "Running…" : "Generate";
    el.lane.classList.toggle("looping", state.running);
    renderTabs(); renderLane(); renderCamera(); renderVerdict();
  }

  function renderTabs() {
    const ns = Object.keys(state.iters).map(Number).sort((a, b) => a - b);
    el.tabs.replaceChildren(...ns.map((n) => h("button", {
      type: "button", role: "tab", "aria-selected": String(n === state.selected),
      onclick: () => { state.selected = n; state.follow = n === Math.max(...ns); state.frameSel = null; render(); },
    }, `Iteration ${n}`)));
  }

  function renderLane() {
    const it = state.iters[state.selected];
    for (const li of el.lane.querySelectorAll(".station")) {
      const s = li.dataset.stage;
      const st = it ? it.st[s] : { state: "pending", data: {} };
      li.dataset.state = st.state;
      li.querySelector(".out").replaceChildren(...(st.state === "pending" ? [] : OUT[s](st.data, st.state)));
    }
  }

  const OUT = {
    generator(d) {
      const out = [];
      if (d.refinement_context) out.push(h("p", { class: "meta" }, "Using the Reflector's correction from the previous iteration."));
      if (d.spec) {
        const sp = d.spec;
        out.push(h("div", { class: "chips" },
          chip("weather", sp.weather), chip("map", sp.town), chip("vehicles", sp.num_vehicles),
          chip("pedestrian", sp.pedestrian ? "yes" : "no"), sp.cyclist ? chip("cyclist", "yes") : null,
          chip("event", String(sp.event_type).replaceAll("_", " ")), chip("duration", `${sp.duration_s} s`)));
        if (sp.notes) out.push(h("p", { class: "meta" }, sp.notes));
        if (d.latency_s != null) out.push(h("p", { class: "meta" }, `Took ${secs(d.latency_s)}`));
      }
      return out;
    },
    engine(d, st) {
      if (st === "active") return [h("p", { class: "meta" }, `Loading ${d.town || "the map"} and running the scenario…`)];
      return [
        h("p", {}, d.success ? `Ran to completion and captured ${d.n_frames} frame${d.n_frames === 1 ? "" : "s"}.`
                             : `The simulation did not complete.${d.error ? " " + d.error : ""}`),
        d.latency_s != null ? h("p", { class: "meta" }, `Took ${secs(d.latency_s)}`) : null,
      ];
    },
    verifier(d, st) {
      if (st === "active") return [h("p", { class: "meta" }, "Asking the vision model about the captured frames…")];
      const out = [];
      const failed = new Set(d.missing_elements || []);
      const names = d.applicable_predicates || Object.keys(d.predicates || {});
      if (names.length) {
        out.push(h("table", { class: "preds" }, h("tbody", {}, names.map((k) => {
          const ok = d.predicates ? d.predicates[k] : !failed.has(k);
          return h("tr", {}, h("td", {}, PRED_LABELS[k] || k), h("td", { class: ok ? "yes" : "no" }, ok ? "✓ holds" : "✗ fails"));
        }))));
      } else if (failed.size) {
        out.push(h("table", { class: "preds" }, h("tbody", {}, [...failed].map((k) =>
          h("tr", {}, h("td", {}, PRED_LABELS[k] || k), h("td", { class: "no" }, "✗ fails"))))));
      }
      if (d.predicate_agreement != null) {
        out.push(h("div", { class: "score-row", style: "margin-top:10px" },
          h("span", {}, `Predicate agreement over ${d.n_samples} VLM sample${d.n_samples === 1 ? "" : "s"}`),
          h("strong", {}, `${pct(d.predicate_agreement)} ± ${Number(d.predicate_agreement_std || 0).toFixed(2)}`)));
      }
      if (d.score != null) {
        const bar = h("div", { class: `bar ${d.passed ? "pass" : ""}` }, h("i", { style: `width:${Math.min(100, d.score * 100)}%` }),
          d.threshold != null ? h("b", { style: `left:${d.threshold * 100}%`, title: `Pass threshold ${pct(d.threshold)}` }) : null);
        out.push(h("div", { class: "score" },
          h("div", { class: "score-row" }, h("span", {}, "Hybrid score (VLM, state log, execution)"),
            h("strong", { class: d.passed ? "yes" : "no" }, `${pct(d.score)} ${d.passed ? "pass" : "fail"}`)), bar));
      }
      const meta = [];
      if (d.log_checks_total) meta.push(`state-log checks ${d.log_checks_passed}/${d.log_checks_total}`);
      if (d.backend) meta.push(`backend ${d.backend}`);
      if (d.latency_s != null) meta.push(`took ${secs(d.latency_s)}`);
      if (meta.length) out.push(h("p", { class: "meta" }, meta.join(", ")));
      if (d.predicate_agreement == null && config?.mode === "replay") {
        out.push(h("p", { class: "meta" }, "Recorded runs store the hybrid score and failed checks only."));
      }
      return out;
    },
    reflector(d, st) {
      if (st === "skipped") return [h("p", { class: "meta" }, "Not needed: every check passed.")];
      if (st === "active") return [h("p", { class: "meta" }, `Diagnosing: ${(d.missing_elements || []).map((k) => PRED_LABELS[k] || k).join(", ") || "low score"}`)];
      return [h("pre", { class: "correction" }, d.refinement_context || "(no correction text)"),
              d.latency_s != null ? h("p", { class: "meta" }, `Took ${secs(d.latency_s)}`) : null];
    },
  };
  function chip(k, v) { return h("span", { class: "chip" }, `${k} `, h("b", {}, v)); }

  function renderCamera() {
    const it = state.iters[state.selected];
    const frames = it ? it.frames : [];
    const idx = state.frameSel ?? frames.length - 1;
    if (frames.length) {
      el.frame.src = frames[idx]; el.frame.hidden = false; el.empty.hidden = true;
      el.liveTag.hidden = false;
      el.liveTag.textContent = `Iteration ${state.selected}, frame ${idx + 1} of ${frames.length}`;
    } else {
      el.frame.hidden = true; el.empty.hidden = false; el.liveTag.hidden = true;
    }
    el.film.replaceChildren(...frames.map((src, i) => h("button", {
      type: "button", "aria-label": `Show frame ${i + 1}`, "aria-current": String(i === idx),
      onclick: () => { state.frameSel = i; renderCamera(); },
    }, h("img", { src, alt: "" }))));
  }

  function renderVerdict() {
    const f = state.final;
    if (!f) { el.verdict.hidden = true; return; }
    const [title, body] = STATUS_TEXT[f.final_status] || [f.final_status, ""];
    el.verdict.hidden = false;
    el.verdict.className = `verdict ${f.final_status === "passed" ? "pass" : "fail"}`;
    el.verdict.replaceChildren(h("h2", {}, title), h("p", {},
      `${body} ${f.iterations} iteration${f.iterations === 1 ? "" : "s"}${f.total_wall_time_s ? `, ${secs(f.total_wall_time_s)} total` : ""}.`));
  }

  function log(stage, p, t) {
    const time = new Date((t || Date.now() / 1000) * 1000).toLocaleTimeString([], { hour12: false });
    const it = p && p.iteration ? ` (iteration ${p.iteration})` : "";
    const extra = stage === "verifier_done" ? `: score ${pct(p.score)}, ${p.passed ? "pass" : "fail"}`
      : stage === "error" ? `: ${p.message}` : stage === "done" ? `: ${p.final_status}` : "";
    el.log.prepend(h("li", {}, h("time", {}, time), h("span", {}, `${stage}${it}${extra}`)));
  }

  let toastTimer = null;
  function toast(text) {
    el.toast.textContent = text; el.toast.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.toast.hidden = true; }, 7000);
  }

  // ---------- input ----------
  function updateCount() { el.count.textContent = `${el.prompt.value.length} / ${config?.max_prompt_chars || 500}`; }
  function submit() {
    const prompt = el.prompt.value.trim();
    if (!prompt || state.running) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) { toast("Not connected to the server yet."); return; }
    ws.send(JSON.stringify({ type: "generate", prompt }));
  }
  el.form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  el.prompt.addEventListener("input", updateCount);
  el.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  });
  el.clear.addEventListener("click", () => { el.prompt.value = ""; updateCount(); el.prompt.focus(); });

  loadConfig().then(connect);
  updateCount();
  render();
})();
