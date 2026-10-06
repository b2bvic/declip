"use strict";
const token = document.querySelector('meta[name="declip-token"]').content;
const player = document.querySelector("#player");
const notice = document.querySelector("#notice");
const resultMode = document.querySelector("#play-result");
let state, currentId = null, queue = Promise.resolve(), finished = false, aroundEnd = null;
const seconds = value => `${value.toFixed(3)} s`;
function message(text) { notice.textContent = text; }
function play() {
  player.play().catch(error => {
    if (error.name !== "AbortError") message(error.message);
  });
}
async function request(path, method = "GET", body) {
  const response = await fetch(path, {
    method, headers: {"X-Declip-Token": token, "Content-Type": "application/json"},
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  const data = await response.json();
  if (response.status === 409) {
    state = await request("/api/edit"); draw();
    message("The edit list changed. Reloaded the latest decisions. Please repeat your change.");
    return null;
  }
  if (response.status === 422) {
    message(`${data.unresolved_count} cuts still need a decision.`); return null;
  }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function enqueue(action) {
  queue = queue.then(async () => { if (!finished) await action(); }).catch(error => message(error.message));
  return queue;
}
function save(change) {
  return enqueue(async () => {
    const update = typeof change === "function" ? change() : change;
    const data = await request("/api/edit", "PUT", {revision: state.revision, ...update});
    if (data) { state = data; draw(); message("Saved."); }
  });
}
function cuts() { return state.edit_list.cuts; }
function orderedCuts() { return [...cuts()].sort((a, b) => a.start - b.start || a.end - b.end); }
function currentCut() { return cuts().find(cut => cut.id === currentId); }
function choose(id) {
  currentId = id;
  document.querySelectorAll(".cut-row").forEach(row => {
    row.classList.toggle("current", row.dataset.id === id);
    if (row.dataset.id === id) row.scrollIntoView({block: "nearest"});
  });
}
function decide(status) {
  const cut = currentCut();
  if (cut && cut.origin === "auto") save({decisions: [{id: cut.id, status}]});
  else if (cut) message("Manual cuts are accepted. Use Remove manual cut to undo one.");
}
function toggle(id) {
  choose(id);
  save(() => {
    const cut = cuts().find(c => c.id === id);
    if (!cut) throw new Error("This cut was removed.");
    return {decisions: [{id, status: cut.status === "accepted" ? "rejected" : "accepted"}]};
  });
}
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function draw() {
  const edit = state.edit_list, summary = state.summary;
  document.querySelector("#source").textContent = edit.source.name;
  document.querySelector("#summary").textContent =
    `${summary.counts.proposed} proposed · ${summary.counts.accepted} accepted · ${summary.counts.rejected} rejected · ` +
    `${seconds(summary.removed_seconds)} removed · Output: ${seconds(summary.output_duration)}`;
  const transcript = document.querySelector("#transcript"); transcript.replaceChildren();
  const words = edit.transcript ? edit.transcript.words : [];
  if (!words.length) transcript.append(element("p", "No transcript. Review the timed cuts below."));
  words.forEach(word => {
    const span = element("span", word.text, "word"); span.dataset.word = word.i;
    const covering = cuts().filter(cut => word.start < cut.end && word.end > cut.start && cut.kind !== "gap");
    covering.forEach(cut => span.classList.add(cut.kind, cut.status));
    span.tabIndex = 0; span.title = `Seek to ${seconds(word.start)}`;
    span.addEventListener("click", () => { player.currentTime = word.start; });
    span.addEventListener("keydown", event => { if (event.key === "Enter") player.currentTime = word.start; });
    transcript.append(span, document.createTextNode(" "));
  });
  const list = document.querySelector("#cuts"); list.replaceChildren();
  for (const cut of orderedCuts()) {
    const row = element("div", undefined, `cut-row ${cut.kind} ${cut.status}`); row.dataset.id = cut.id;
    row.classList.toggle("current", cut.id === currentId);
    const button = element("button",
      `${cut.kind} · ${seconds(cut.start)}–${seconds(cut.end)} · ${cut.label} · ${cut.status}`);
    button.type = "button"; button.dataset.cut = cut.id;
    button.addEventListener("click", () => {
      if (cut.origin === "auto") toggle(cut.id);
      else { choose(cut.id); player.currentTime = cut.start; }
    });
    row.append(button);
    if (cut.kind === "gap") row.append(element("span", `Gap ${seconds(cut.end - cut.start)}`, "badge"));
    if (cut.low_confidence) row.append(element("span", "Low confidence", "badge warning"));
    if (cut.origin === "manual") {
      const remove = element("button", "Remove manual cut"); remove.type = "button";
      remove.addEventListener("click", () => save({manual_cuts: {remove: [cut.id]}})); row.append(remove);
    }
    list.append(row);
  }
}
function manualSelection() {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || !selection.rangeCount) {
    message("Select a word range first."); return;
  }
  const range = selection.getRangeAt(0);
  const selected = [...document.querySelectorAll("#transcript .word")]
    .filter(node => range.intersectsNode(node)).map(node => Number(node.dataset.word));
  if (!selected.length) { message("Select transcript words first."); return; }
  const words = state.edit_list.transcript.words;
  const first = words[Math.min(...selected)], last = words[Math.max(...selected)];
  const label = words.slice(first.i, last.i + 1).map(word => word.text).join(" ").slice(0, 200);
  selection.removeAllRanges();
  save({manual_cuts: {add: [{start: first.start, end: last.end, label}]}});
}
function playAround() {
  const cut = currentCut(); if (!cut) { message("Choose a cut first."); return; }
  resultMode.checked = true; player.currentTime = Math.max(0, cut.start - 1);
  aroundEnd = Math.min(state.edit_list.media.duration, cut.end + 1);
  play();
}
player.addEventListener("timeupdate", () => {
  if (state && resultMode.checked) {
    const interval = state.summary.removed_intervals.find(([start, end]) => player.currentTime >= start && player.currentTime < end);
    if (interval) {
      if (interval[1] >= player.duration) player.pause();
      player.currentTime = Math.min(interval[1], player.duration || interval[1]);
    }
  }
  if (aroundEnd !== null && player.currentTime >= aroundEnd) { player.pause(); aroundEnd = null; }
});
document.querySelector("#around").addEventListener("click", playAround);
document.querySelector("#bulk").addEventListener("click", () => save(() => ({decisions:
  cuts().filter(cut => cut.origin === "auto" && cut.kind === document.querySelector("#bulk-kind").value)
    .map(cut => ({id: cut.id, status: "accepted"}))
})));
document.querySelector("#finish").addEventListener("click", () => enqueue(async () => {
  const data = await request("/api/finish", "POST", {revision: state.revision});
  if (data) {
    finished = true; player.pause();
    document.querySelectorAll("button, select, input").forEach(node => { node.disabled = true; });
    message("Review passed. Decisions saved. You can close this page and render or export.");
  }
}));
document.addEventListener("keydown", event => {
  if (!state || finished || event.ctrlKey || event.metaKey || event.altKey ||
      /INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (!["j", "k", "a", "x", " ", "p", "c"].includes(key)) return;
  event.preventDefault();
  if (key === "j" || key === "k") {
    const ordered = orderedCuts(), index = ordered.findIndex(cut => cut.id === currentId);
    const next = index < 0 ? (key === "j" ? 0 : ordered.length - 1) :
      Math.max(0, Math.min(ordered.length - 1, index + (key === "j" ? 1 : -1)));
    if (ordered[next]) choose(ordered[next].id);
  } else if (key === "a") decide("accepted");
  else if (key === "x") decide("rejected");
  else if (key === "c") manualSelection();
  else if (key === "p") playAround();
  else if (key === " ") {
    aroundEnd = null;
    if (player.paused) play(); else player.pause();
  }
});
request("/api/edit").then(data => { state = data; draw(); }).catch(error => message(error.message));
