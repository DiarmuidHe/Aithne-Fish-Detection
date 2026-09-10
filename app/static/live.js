/* Independent of the upload/batch dashboard state. No build step. */
(() => {
  const el = (id) => document.getElementById(`live-${id}`);
  const open = new Set(["queued", "starting", "running", "reconnecting", "stopping"]);
  const STORAGE_KEY = "live-source-key";
  const SPECIES_STORAGE_KEY = "live-species-fish-target";
  let fishial = {enabled: false};
  let speciesDraft = null;
  let speciesLocked = false;
  let session = null;
  let enabled = false;
  let available = false;
  let busy = false;
  let imageVersion = null;
  let gallerySession = null;
  let timer = null;
  let sources = [];
  let sourceKey = null;
  let optionSignature = null;
  const cards = new Map();

  function message(error = "") {
    el("error").textContent = error;
    el("error").hidden = !error;
  }

  function stored(value) {
    // A browser can refuse storage entirely; the default camera is still usable.
    try {
      return value === undefined ? window.localStorage.getItem(STORAGE_KEY) : window.localStorage.setItem(STORAGE_KEY, value);
    } catch (error) {
      return null;
    }
  }

  function selected() {
    return sources.find((source) => source.key === sourceKey) || null;
  }

  function sourceLabel() {
    return selected()?.label || "the camera";
  }

  function resetView() {
    gallerySession = null;
    cards.clear();
    el("gallery").replaceChildren();
    el("gallery-empty").hidden = false;
    el("species-breakdown").hidden = true;
    imageVersion = null;
    el("player").hidden = true;
    el("player").removeAttribute("src");
    el("placeholder").hidden = false;
  }

  function chooseSource(key, remember) {
    if (key === sourceKey) return;
    sourceKey = key;
    if (remember) stored(key);
    session = null;
    resetView();
  }

  function renderSources() {
    // Rebuilding on every poll would fight the operator's own pointer.
    const signature = sources.map((source) => `${source.key}:${source.active_session_id || ""}`).join("|");
    if (signature !== optionSignature && document.activeElement !== el("source")) {
      optionSignature = signature;
      el("source").replaceChildren(...sources.map((source) => {
        const option = document.createElement("option");
        option.value = source.key;
        const place = source.location ? ` · ${source.location}` : "";
        option.textContent = `${source.label}${place}${source.active_session_id ? " · monitoring" : ""}`;
        return option;
      }));
    }
    if (sourceKey) el("source").value = sourceKey;
    el("heading").textContent = selected() ? `Live ${selected().label}` : "Live camera";
  }

  function applySources(registry) {
    enabled = registry.enabled;
    available = registry.available;
    sources = registry.sources;
    fishial = registry.fishial || {enabled: false};
    if (speciesDraft === null) {
      let count = 0;
      try { count = Number(window.localStorage.getItem(SPECIES_STORAGE_KEY)) || 0; } catch (error) { /* Storage is optional. */ }
      speciesDraft = {count: Math.max(0, Math.min(fishial.max_fish_per_session || 0, Math.floor(count))),
                      frames: fishial.default_frames_per_fish || 5};
      el("species-count").value = speciesDraft.count;
      el("species-frames").value = speciesDraft.frames;
    }
    el("species-count").max = fishial.max_fish_per_session || 0;
    el("species-controls").hidden = !fishial.enabled;
    const keys = new Set(sources.map((source) => source.key));
    if (!keys.has(sourceKey)) {
      const remembered = stored();
      sourceKey = keys.has(remembered) ? remembered : registry.default_key;
    }
    // Only one session runs at a time, so a reload or a second tab must show it
    // rather than polling a camera that is not the one being analyzed.
    const running = sources.find((source) => source.active_session_id);
    if (running && running.key !== sourceKey) chooseSource(running.key, false);
    renderSources();
  }

  function controls() {
    const running = Boolean(session && open.has(session.status));
    el("start").disabled = busy || !enabled || !available || running;
    el("stop").disabled = busy || !session || !running || session.status === "stopping";
    el("source").disabled = busy || running || !sources.length;
    el("source-hint").hidden = !running;
    el("species-count").disabled = busy || running || !fishial.enabled;
    el("species-frames").disabled = busy || running || !fishial.enabled;
    el("species-hint").hidden = !running || !fishial.enabled;
    if (running && session.species_id) {
      el("species-count").value = session.species_id.fish_target;
      el("species-frames").value = session.species_id.frames_per_fish;
      speciesLocked = true;
    } else if (speciesLocked && speciesDraft) {
      el("species-count").value = speciesDraft.count;
      el("species-frames").value = speciesDraft.frames;
      speciesLocked = false;
    }
  }

  function showStatus() {
    const species = session?.species_id;
    el("species-progress").hidden = !species?.enabled;
    if (species?.enabled) el("species-progress").textContent = `${species.fish_enrolled}/${species.fish_target} enrolled · ${species.fish_identified} identified · ${species.fish_review_required} need review · ${species.api_calls} calls (${species.calls_saved} saved)`;
    if (!enabled) {
      el("status").textContent = "Live monitoring is disabled. Enable LIVE_MONITOR_ENABLED on the API and live worker.";
    } else if (!available) {
      el("status").textContent = "Live monitoring needs the VIAME worker in GPU mode. The current deployment uses mock detections.";
    } else if (!session) {
      el("status").textContent = `Ready to monitor ${sourceLabel()}.`;
    } else {
      const lag = session.lag_seconds == null ? "Waiting for frames" : `${session.lag_seconds.toFixed(1)}s since the latest analyzed frame`;
      const queued = session.status === "queued" ? " · Waiting for the live worker" : "";
      el("status").textContent = `${session.status} · ${lag}${queued} · ${session.reconnect_count} reconnects · ${session.dropped_segments} skipped segments`;
      el("gallery-note").textContent = `Fish histories finish after about ${session.lost_track_seconds} seconds without a detection, or when monitoring stops. Showing the latest 50 histories.`;
      message(session.worker_stale ? "The live worker heartbeat is overdue. Check the live worker service." : session.error_message || "");
      // Fetch a fresh immutable response every poll, avoiding stuck MJPEG sockets after reconnects.
      if (session.annotated_stream_url && imageVersion !== session.last_frame_at) {
        imageVersion = session.last_frame_at;
        el("player").src = `${session.annotated_stream_url}?snapshot=true&t=${encodeURIComponent(imageVersion)}`;
      }
    }
    controls();
  }

  el("player").addEventListener("load", () => {
    el("player").hidden = false;
    el("placeholder").hidden = true;
  });
  el("player").addEventListener("error", () => {
    imageVersion = null;
    el("player").hidden = true;
    el("placeholder").hidden = false;
    el("placeholder").textContent = "Waiting for an annotated frame…";
  });

  function showActivity(activity) {
    const metrics = [["Active tracks", activity.active_tracks], ["Tracks in window", activity.window_tracks],
      ["Detections in window", activity.window_detections], ["Finalized", activity.finalized_tracks]];
    el("metrics").replaceChildren(...metrics.map(([label, value]) => {
      const card = document.createElement("div");
      const number = document.createElement("strong");
      number.textContent = value;
      const caption = document.createElement("span");
      caption.textContent = label;
      card.append(number, caption);
      return card;
    }));
    const max = Math.max(1, ...activity.series.map((point) => point.tracks));
    el("chart").replaceChildren(...activity.series.map((point) => {
      const bar = document.createElement("span");
      bar.style.height = `${Math.max(2, point.tracks / max * 100)}%`;
      bar.title = `${new Date(point.at).toLocaleTimeString()}: ${point.tracks} tracks`;
      return bar;
    }));
    el("chart").setAttribute("aria-label", `Tracks per interval, oldest to newest: ${activity.series.map((p) => p.tracks).join(", ")}`);
    el("window").textContent = `Last ${activity.window_seconds} seconds of analyzed footage · distinct tracks per interval`;
  }

  function showGallery(tracks) {
    // Retain existing video elements so polling never interrupts playback.
    const ids = new Set(tracks.map((track) => track.id));
    for (const [id, card] of cards) {
      if (!ids.has(id)) { card.remove(); cards.delete(id); }
    }
    for (const track of tracks) {
      if (cards.has(track.id)) {
        updateSpecies(cards.get(track.id), track);
        continue;
      }
      const card = document.createElement("article");
      card.className = "clip-card";
      const title = document.createElement("h4");
      title.textContent = `${track.species || "Fish"} · ${track.id.slice(0, 8)}`;
      card.append(title);
      const speciesBadge = document.createElement("p");
      speciesBadge.className = "live-species-badge chart-note";
      card.append(speciesBadge);
      updateSpecies(card, track);
      if (track.clip_url) {
        const video = document.createElement("video");
        video.controls = true; video.muted = true; video.loop = true; video.playsInline = true;
        video.preload = "none"; video.src = track.clip_url;
        if (track.crop_url) video.poster = track.crop_url;
        card.append(video);
      } else if (track.crop_url) {
        const img = document.createElement("img");
        img.src = track.crop_url; img.alt = `Annotated crop of ${track.species || "fish"}`; img.loading = "lazy";
        card.append(img);
      }
      const detail = document.createElement("p");
      detail.className = "chart-note";
      detail.textContent = `${track.detection_count} detections · ${(track.max_confidence * 100).toFixed(0)}% confidence · ${track.finalization_reason} · ${new Date(track.last_seen_at).toLocaleTimeString()}`;
      card.append(detail);
      if (track.crop_url) {
        const link = document.createElement("a");
        link.href = track.crop_url; link.textContent = "Open annotated crop"; link.className = "text-link";
        card.append(link);
      }
      if (track.media_error) {
        const error = document.createElement("p"); error.textContent = track.media_error; card.append(error);
      }
      cards.set(track.id, card);
    }
    // Only move cards whose order changed.
    tracks.forEach((track, index) => {
      const card = cards.get(track.id);
      if (el("gallery").children[index] !== card) el("gallery").insertBefore(card, el("gallery").children[index] || null);
    });
    el("gallery-empty").hidden = tracks.length > 0;
    el("clips-count").textContent = tracks.length;
  }

  function updateSpecies(card, track) {
    const title = card.querySelector("h4");
    const badge = card.querySelector(".live-species-badge");
    const state = track.fishial_state;
    title.textContent = `${state === "identified" ? track.fishial_species : track.species || "Fish"} · ${track.id.slice(0, 8)}`;
    badge.classList.toggle("count-pill", state === "review_required" || state === "error");
    badge.title = "";
    badge.hidden = !state || state === "disabled";
    if (state === "identified") {
      badge.textContent = `(${(track.fishial_species_confidence * 100).toFixed(0)}% · Fishial AI)`;
    } else if (state === "review_required" || state === "error") {
      const audit = track.fishial_votes || {};
      const diagnostics = track.fishial_diagnostics || {};
      const reason = diagnostics.reason || audit.reason || "Consensus not reached";
      const scope = diagnostics.budget_scope ? ` (${diagnostics.budget_scope})` : "";
      const candidates = diagnostics.candidates || [];
      const names = candidates.map((candidate) => {
        const name = candidate.common_name
          ? `${candidate.common_name} (${candidate.species})` : candidate.species;
        const rejected = candidate.rejected_for_region ? "; rejected for region" : "";
        return `${name} (${(candidate.max_score * 100).toFixed(0)}% best Fishial score${rejected})`;
      });
      badge.textContent = names.length
        ? `Review required · Tentative species: ${names.join("; ")} · ${reason}${scope}`
        : `Review required: ${reason}${scope}`;
      badge.title = `${diagnostics.stop_reason || reason} · ${JSON.stringify(diagnostics.frame_reasons || {})} · votes ${JSON.stringify(audit.tally || {})}`;
      badge.tabIndex = 0;
      badge.setAttribute("aria-label", `${badge.textContent} · ${badge.title}`);
    } else if (state === "candidate") {
      badge.textContent = "Staged for identification";
    } else {
      badge.textContent = "Identifying species…";
    }
  }

  function showSpecies(data) {
    el("species-breakdown").hidden = !session?.species_id?.enabled;
    el("species-list").replaceChildren(...data.species.map((entry) => {
      const item = document.createElement("li");
      item.textContent = `${entry.species} · ${entry.count} fish · ${(entry.mean_confidence * 100).toFixed(0)}% mean confidence`;
      return item;
    }));
    const declined = data.declined ? `${data.declined} not named by the model · ` : "";
    el("species-summary").textContent = `${data.review_required} need review · ${declined}`
      + `${data.api_calls} Fishial image calls (${data.calls_saved} saved)`
      + Object.entries(data.review_reasons || {}).map(([reason, count]) => ` · ${count} ${reason}`).join("");
  }

  async function refresh() {
    if (busy) return;
    busy = true;
    try {
      applySources(await apiRequest("/live/sources"));
      const latest = await apiRequest(`/live/latest?source=${encodeURIComponent(sourceKey)}`);
      enabled = latest.enabled; available = latest.available; session = latest.session;
      if (session && gallerySession !== session.id) {
        resetView();
        gallerySession = session.id;
      }
      showStatus();
      if (session) {
        const [activity, tracks, species] = await Promise.all([
          apiRequest(`/live/${session.id}/activity`), apiRequest(`/live/${session.id}/clips`),
          apiRequest(`/live/${session.id}/species`),
        ]);
        showActivity(activity); showGallery(tracks); showSpecies(species);
      }
    } catch (error) { message(error.message); }
    finally { busy = false; controls(); }
  }

  async function act(action) {
    if (busy) return;
    if (action === "start" && fishial.enabled &&
        (!el("species-count").reportValidity() || !el("species-frames").reportValidity())) return;
    busy = true; controls(); message();
    try {
      const request = action === "start"
        ? ["/live/start", {method: "POST", headers: {"Content-Type": "application/json"},
                           body: JSON.stringify({source: sourceKey,
                             species_id_fish_target: fishial.enabled ? Number(el("species-count").value) : 0,
                             species_id_frames_per_fish: Number(el("species-frames").value)})}]
        : [`/live/${session.id}/stop`, {method: "POST"}];
      session = await apiRequest(...request);
      showStatus();
    } catch (error) { message(error.message); }
    finally {
      busy = false; controls();
      window.clearTimeout(timer);
      timer = window.setTimeout(poll, 0);
    }
  }
  el("start").addEventListener("click", () => act("start"));
  el("stop").addEventListener("click", () => act("stop"));
  for (const name of ["count", "frames"]) {
    el(`species-${name}`).addEventListener("change", () => {
      speciesDraft[name] = Number(el(`species-${name}`).value);
      if (name === "count") {
        try { window.localStorage.setItem(SPECIES_STORAGE_KEY, String(speciesDraft.count)); } catch (error) { /* Optional. */ }
      }
    });
  }
  el("source").addEventListener("change", (event) => {
    chooseSource(event.target.value, true);
    showStatus();
    window.clearTimeout(timer);
    timer = window.setTimeout(poll, 0);
  });
  async function poll() {
    await refresh();
    timer = window.setTimeout(poll, session && open.has(session.status) ? 2000 : 10000);
  }
  poll();
})();
