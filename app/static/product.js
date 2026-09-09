/* FastAPI-served product layer; no external chart library or build step. */
const selectedVideos = new Set();
const productElement = id => document.getElementById(id);
let batchBusy = false;
let reviewBusy = false;
let inspectedTrackId = null;

function renderBars(id, rows) {
  const root = productElement(id);
  const maximum = Math.max(1, ...rows.map(row => row.value));
  root.innerHTML = rows.length ? rows.map(row => `
    <div class="bar-row"><span class="bar-label" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</span>
      <span class="bar-rail"><span class="bar-fill" style="width:${Math.max(0, row.value / maximum * 100)}%"></span></span>
      <strong>${escapeHtml(row.value)}</strong></div>`).join("") : '<p class="chart-note">No videos yet.</p>';
}

function syncSelection() {
  for (const id of selectedVideos) if (!state.videos.some(v => v.id === id)) selectedVideos.delete(id);
  document.querySelectorAll(".batch-select").forEach(input => {
    input.checked = selectedVideos.has(input.value);
    input.disabled = batchBusy;
    input.onchange = () => {
      input.checked ? selectedVideos.add(input.value) : selectedVideos.delete(input.value);
      syncSelection();
    };
  });
  const videos = state.videos.filter(v => selectedVideos.has(v.id));
  productElement("selection-count").textContent = `${videos.length} selected`;
  const all = productElement("select-all");
  all.checked = videos.length > 0 && videos.length === state.videos.length;
  all.indeterminate = videos.length > 0 && videos.length < state.videos.length;
  all.disabled = batchBusy;
  productElement("batch-process").disabled = batchBusy || !videos.length;
  productElement("batch-annotate").disabled = batchBusy || !videos.some(v => v.processing_status === "completed");
  productElement("batch-export").disabled = !videos.length;
  if (!batchBusy) {
    const count = status => videos.filter(v => v.processing_status === status).length;
    productElement("batch-progress").max = Math.max(1, videos.length);
    productElement("batch-progress").value = count("completed") + count("failed");
    productElement("batch-status").textContent = videos.length
      ? `${count("completed")} completed · ${count("failed")} failed · ${count("processing")} processing · ${count("queued")} queued · ${count("uploaded")} uploaded`
      : "Select videos to begin.";
  }
}

async function renderOverview() {
  const rows = await apiRequest("/analytics/videos");
  renderBars("fish-chart", rows.map(row => ({label: row.filename, value: row.accepted_fish_count})));
  renderBars("detections-chart", rows.map(row => ({label: row.filename, value: row.accepted_detections})));
}

function clipCard(clip) {
  const species = clip.species ? ` · ${clip.species}` : "";
  const duration = `${clip.duration_seconds.toFixed(1)} s`;
  return `<figure class="clip-card">
    <video src="/tracks/${encodeURIComponent(clip.track_id)}/clip?v=${encodeURIComponent(clip.generated_at)}" controls loop muted playsinline preload="metadata"></video>
    <figcaption>
      <strong>Fish #${escapeHtml(clip.viame_track_id)}</strong>${escapeHtml(species)}
      <span class="clip-meta">${escapeHtml(formatSeconds(clip.start_seconds))}–${escapeHtml(formatSeconds(clip.end_seconds))} · ${escapeHtml(duration)} · ${escapeHtml(clip.detection_count)} observations · ${escapeHtml((clip.max_confidence * 100).toFixed(1))}%</span>
      <a class="text-link" href="/tracks/${encodeURIComponent(clip.track_id)}/clip?download=true" download>Download clip</a>
    </figcaption>
  </figure>`;
}

function renderClips(clips) {
  const grid = productElement("clips-grid");
  const signature = clips.map(clip => `${clip.track_id}:${clip.generated_at}`).join("|");
  productElement("clips-count").textContent = clips.length;
  productElement("clips-empty").hidden = clips.length > 0;
  // Rebuilding identical cards would restart a clip the operator is watching.
  if (grid.dataset.signature === signature) return;
  grid.dataset.signature = signature;
  grid.innerHTML = clips.map(clipCard).join("");
}

async function loadClips(video) {
  const clips = await apiRequest(`/videos/${video.id}/fish-clips`);
  if (state.selectedVideoId !== video.id) return;
  renderClips(clips);
}

productElement("clips-button").addEventListener("click", async () => {
  const video = state.videos.find(v => v.id === state.selectedVideoId);
  if (!video) return;
  const button = productElement("clips-button");
  button.disabled = true;
  button.textContent = "Cutting clips…";
  try {
    const clips = await apiRequest(`/videos/${video.id}/fish-clips`, {method: "POST"});
    if (state.selectedVideoId === video.id) renderClips(clips);
    showToast(clips.length ? `${clips.length} fish clips ready` : "No accepted fish tracks to clip");
  } catch (error) { showToast(error.message, true); }
  finally {
    button.disabled = false;
    button.textContent = "Generate fish clips";
  }
});

async function renderProductDetail(video) {
  const data = await apiRequest(`/analytics/videos/${video.id}`);
  if (state.selectedVideoId !== video.id) return;
  const timePrecision = data.time_bins[0].end - data.time_bins[0].start < 0.1 ? 3 : 1;
  renderBars("timeline-chart", data.time_bins.map(row => ({label: `${row.start.toFixed(timePrecision)}–${row.end.toFixed(timePrecision)} s`, value: row.accepted_detections})));
  renderBars("confidence-chart", data.confidence_distribution.map(row => ({label: `${Math.round(row.start * 100)}–${Math.round(row.end * 100)}%`, value: row.count})));
  renderBars("duration-chart", data.track_duration_distribution.map(row => ({label: `${row.start.toFixed(2)}–${row.end.toFixed(2)} s`, value: row.count})));
  productElement("analytics-missing").textContent = `${data.unknown_timestamp_detections} observations without timestamps · ${data.unknown_duration_tracks} tracks without duration. Final bin includes its upper boundary.`;
  await loadClips(video);
  productElement("video-exports").innerHTML = [["summary", "Summary CSV"], ["accepted-tracks", "Accepted tracks CSV"], ["all-tracks", "All tracks CSV"], ["detections", "Observations CSV"]].map(([kind, label]) => `<a class="text-link" href="/videos/${video.id}/exports/${kind}.csv">${label}</a>`).join("");
}

productElement("select-all").addEventListener("change", event => {
  selectedVideos.clear();
  if (event.target.checked) state.videos.forEach(video => selectedVideos.add(video.id));
  syncSelection();
});

async function runBatch(action) {
  if (batchBusy) return;
  const videos = state.videos.filter(v => selectedVideos.has(v.id));
  batchBusy = true;
  syncSelection();
  const results = productElement("batch-results");
  results.innerHTML = "";
  results.hidden = false;
  productElement("batch-progress").max = videos.length;
  productElement("batch-progress").value = 0;
  for (const [index, video] of videos.entries()) {
    productElement("batch-status").textContent = `${action === "process" ? "Queueing" : "Annotating"} ${index + 1} of ${videos.length}: ${video.original_filename}`;
    let message;
    try {
      if (action === "annotate" && video.processing_status !== "completed") {
        message = "Skipped: processing is not completed";
      } else {
        const data = await apiRequest(`/batch/${action}`, {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({video_ids: [video.id]}),
        });
        const result = data.results[0];
        message = result.ok ? (action === "annotate" ? "Annotation ready" : result.status) : result.error;
      }
    } catch (error) { message = error.message; }
    const li = document.createElement("li");
    li.textContent = `${video.original_filename}: ${message}`;
    results.append(li);
    productElement("batch-progress").value = index + 1;
  }
  batchBusy = false;
  showToast("Batch actions finished. See per-video results.");
  await refreshAll();
  syncSelection();
}
productElement("batch-process").addEventListener("click", () => runBatch("process"));
productElement("batch-annotate").addEventListener("click", () => runBatch("annotate"));
productElement("batch-export").addEventListener("click", () => {
  const params = new URLSearchParams({kind: productElement("batch-export-kind").value});
  selectedVideos.forEach(id => params.append("video_ids", id));
  window.location.assign(`/exports/batch.csv?${params}`);
});

productElement("tracks-body").addEventListener("change", async event => {
  const input = event.target.closest(".review-select");
  if (!input || reviewBusy) return;
  reviewBusy = true;
  input.disabled = true;
  try {
    await apiRequest(`/tracks/${input.dataset.trackId}/review`, {
      method: "PATCH", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({review_state: input.value}),
    });
    showToast("Review saved. Regenerate annotations to include this decision.");
  } catch (error) { showToast(error.message, true); }
  finally {
    reviewBusy = false;
    input.blur();
    await refreshAll();
    input.disabled = false;
  }
});

productElement("tracks-body").addEventListener("click", async event => {
  const button = event.target.closest(".inspect-track");
  if (!button) return;
  try {
    const track = await apiRequest(`/tracks/${button.dataset.trackId}`);
    const video = state.videos.find(v => v.id === track.video_id);
    productElement("inspector-title").textContent = `Track #${track.viame_track_id}`;
    productElement("inspector-meta").textContent = `${track.review_state} · ${track.detection_count} observations · VIAME max confidence ${(track.max_confidence * 100).toFixed(1)}%`;
    productElement("source-player").src = `/videos/${track.video_id}/source-video`;
    inspectedTrackId = track.id;
    resetTrackClipPlayer();
    productElement("observations-body").innerHTML = track.detections.map(d => {
      const time = video?.fps ? d.frame_number / video.fps : d.timestamp_seconds;
      return `<tr><td><button class="seek-observation text-link" ${time == null ? "disabled" : `data-time="${time}"`}>${d.frame_number}</button></td><td>${escapeHtml(formatSeconds(d.timestamp_seconds))}</td><td>${(d.confidence * 100).toFixed(1)}%</td><td>${[d.x1, d.y1, d.x2, d.y2].map(n => Number(n).toFixed(1)).join(", ")}</td></tr>`;
    }).join("");
    productElement("track-inspector").showModal();
  } catch (error) { showToast(error.message, true); }
});
productElement("observations-body").addEventListener("click", event => {
  const button = event.target.closest("[data-time]");
  if (button) { productElement("source-player").pause(); productElement("source-player").currentTime = Number(button.dataset.time); }
});
productElement("track-clip-button").addEventListener("click", async () => {
  const trackId = inspectedTrackId;
  if (!trackId) return;
  const button = productElement("track-clip-button");
  const player = productElement("track-clip-player");
  button.disabled = true;
  button.textContent = "Cutting clip…";
  try {
    const clip = await apiRequest(`/tracks/${trackId}/clip`, {method: "POST"});
    if (inspectedTrackId !== trackId) return;
    player.src = `/tracks/${trackId}/clip?v=${encodeURIComponent(clip.generated_at)}`;
    player.hidden = false;
    button.hidden = true;
  } catch (error) { showToast(error.message, true); }
  finally {
    button.disabled = false;
    button.textContent = "Show cropped clip of this fish";
  }
});

function resetTrackClipPlayer() {
  const player = productElement("track-clip-player");
  player.pause();
  player.removeAttribute("src");
  player.load();
  player.hidden = true;
  productElement("track-clip-button").hidden = false;
}

productElement("close-inspector").addEventListener("click", () => productElement("track-inspector").close());
productElement("track-inspector").addEventListener("close", () => {
  productElement("source-player").pause();
  productElement("source-player").removeAttribute("src");
  productElement("source-player").load();
  inspectedTrackId = null;
  resetTrackClipPlayer();
});
window.productUI = {syncSelection, renderOverview, renderDetail: renderProductDetail};
