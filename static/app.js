"use strict";

const $ = (sel) => document.querySelector(sel);
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const fileTag = $("#fileTag");
const analyzeBtn = $("#analyzeBtn");
const errorBox = $("#errorBox");
const loader = $("#loader");
const loadMsg = $("#loadMsg");
const dashboard = $("#dashboard");

let selectedFile = null;
let lastResult = null;

const LOAD_STEPS = [
  "Decoding image locally...",
  "Computing spectral indices (ExG, HSV, texture)...",
  "Classifying pixels: trees / buildings / roads / water / open...",
  "Removing noise regions and consolidating masks...",
  "Detecting plantable areas with exclusion buffers...",
  "Placing trees via deterministic Poisson-disk sampling...",
  "Calculating heat metrics and projections...",
  "Assembling dashboard and PDF report...",
];
let loadTimer = null;

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function setFile(f) {
  if (!f) return;
  const okExt = /\.(jpe?g|png)$/i.test(f.name);
  if (!okExt) { showError("Only JPG and PNG images are supported."); return; }
  if (f.size > 48 * 1024 * 1024) { showError("File exceeds the 48 MB limit."); return; }
  selectedFile = f;
  errorBox.hidden = true;
  fileTag.hidden = false;
  fileTag.textContent = `${f.name} - ${(f.size / 1048576).toFixed(2)} MB`;
  analyzeBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

function startLoader() {
  let i = 0;
  loadMsg.textContent = LOAD_STEPS[0];
  loader.hidden = false;
  loadTimer = setInterval(() => {
    i = Math.min(i + 1, LOAD_STEPS.length - 1);
    loadMsg.textContent = LOAD_STEPS[i];
  }, 700);
}

function stopLoader() {
  clearInterval(loadTimer);
  loader.hidden = true;
}

async function analyze() {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  startLoader();
  try {
    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("base_temp_c", $("#baseTemp").value);
    fd.append("gsd_m", $("#gsd").value);
    fd.append("spacing_m", $("#spacing").value);
    fd.append("crown_radius_m", $("#crownR").value);

    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed.");
    lastResult = data;
    renderDashboard(data);
  } catch (err) {
    showError(err.message);
  } finally {
    stopLoader();
    analyzeBtn.disabled = false;
  }
}
analyzeBtn.addEventListener("click", analyze);

$("#pdfBtn").addEventListener("click", () => {
  if (lastResult) window.open(lastResult.report_url, "_blank");
});

$("#resetBtn").addEventListener("click", () => {
  dashboard.hidden = true;
  $("#uploadPanel").hidden = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
});

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tabpage").forEach((p) => (p.hidden = true));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).hidden = false;
  });
});

const lightbox = $("#lightbox");
const lightboxImg = $("#lightboxImg");
const lightboxCaption = $("#lightboxCaption");

function openLightbox(src, title) {
  lightboxImg.src = src;
  lightboxImg.alt = title;
  lightboxCaption.textContent = title;
  lightbox.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  lightbox.hidden = true;
  lightboxImg.src = "";
  document.body.style.overflow = "";
}

$("#mapGrid").addEventListener("click", (e) => {
  const img = e.target.closest("img[data-full]");
  if (img) openLightbox(img.dataset.full, img.dataset.title);
});
$("#lightboxClose").addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) closeLightbox();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
});

const fmt = (v, digits = 1) => Number(v).toFixed(digits);

function heatBadgeColor(score) {
  if (score < 30) return "var(--accent)";
  if (score < 50) return "var(--amber)";
  if (score < 70) return "#d97a3c";
  return "var(--red)";
}

function kpiCard(label, value, sub, color, deltaHtml = "") {
  return `<div class="kpi" style="--kpi-color:${color}">
    <div class="k-label">${label}</div>
    <div class="k-value">${value}</div>
    ${sub ? `<div class="k-sub">${sub}</div>` : ""}
    ${deltaHtml ? `<div class="k-delta">${deltaHtml}</div>` : ""}
  </div>`;
}

function renderDashboard(r) {
  const m = r.metrics;
  const gsd = m.gsd_m;

  const kpis = [
    kpiCard("Existing Canopy", fmt(m.canopy_pct, 2) + "%",
      `${Math.round(m.canopy_area_m2).toLocaleString()} m&sup2; of trees`, "var(--text)"),
    kpiCard("Plantable Area", fmt(m.plantable_pct, 2) + "%",
      `${Math.round(m.plantable_area_m2).toLocaleString()} m&sup2; available`, "var(--text)"),
    kpiCard("Current Heat Score", fmt(m.heat_score_current),
      "0-100 scale, lower is better", heatBadgeColor(m.heat_score_current)),
    kpiCard("Est. Current Temperature", fmt(m.temp_current_c, 1) + "&deg;C",
      `baseline ${fmt(m.base_temp_c)}&deg;C at ${gsd} m/px`, "var(--text)"),
    kpiCard("Recommended Planting", "+" + fmt(m.recommended_planting_pct, 2) + "%",
      `${m.recommended_trees} new trees &middot; ${m.tree_spacing_m} m spacing`, "var(--accent)"),
    kpiCard("Projected Canopy", fmt(m.projected_canopy_pct, 2) + "%",
      `+${fmt(m.recommended_planting_pct, 2)} pct points vs today`, "var(--accent)"),
    kpiCard("Projected Heat Score", fmt(m.projected_heat_score),
      `from ${fmt(m.heat_score_current)} today`, heatBadgeColor(m.projected_heat_score)),
    kpiCard("Projected Temperature", fmt(m.projected_temp_c, 1) + "&deg;C",
      `from ${fmt(m.temp_current_c, 1)}&deg;C today`, "var(--text)"),
    kpiCard("Cooling Reduction", "-" + fmt(m.cooling_reduction_c, 2) + "&deg;C",
      "estimated local air temperature drop", "var(--accent)"),
    kpiCard("Shade Improvement", "+" + fmt(m.shade_improvement_pct_points, 2) + " pts",
      `${Math.round(m.shade_improvement_m2).toLocaleString()} m&sup2; new shade`, "var(--accent)"),
  ];
  $("#kpiGrid").innerHTML = kpis.join("");

  const comp = [
    ["Trees / canopy", m.canopy_pct, "#1B7A3D"],
    ["Grass / vegetation", m.grass_pct, "#9CCC65"],
    ["Buildings", m.buildings_pct, "#E65100"],
    ["Roads / paved", m.roads_pct, "#616161"],
    ["Water", m.water_pct, "#1E88E5"],
    ["Bare soil", m.bare_soil_pct, "#A48674"],
  ];
  $("#compositionBars").innerHTML = comp.map(([name, pct, color]) => `
    <div class="comp-row">
      <div class="cr-top"><span>${name}</span><span>${fmt(pct, 2)}%</span></div>
      <div class="cr-track"><div class="cr-fill" style="width:${Math.min(pct * 3, 100)}%;background:${color}"></div></div>
    </div>`).join("");

  $("#legendBox").innerHTML = r.legend
    .filter((l) => l.name !== "Plantable / Open")
    .map((l) => `<span class="lg-item"><span class="lg-swatch" style="background:${l.hex}"></span>${l.name}</span>`)
    .join("");

  const impact = [
    ["CO&#8322; sequestration added", `~${Math.round(m.co2_offset_kg_yr).toLocaleString()} kg CO&#8322;/yr`],
    ["New shaded ground", `${Math.round(m.shade_improvement_m2).toLocaleString()} m&sup2;`],
    ["Total study area", `${Math.round(m.total_area_m2).toLocaleString()} m&sup2;`],
    ["Impervious surface today", `${fmt(m.impervious_pct)}%`],
    ["Processing time", `${m.processing_seconds}s (local CPU)`],
  ];
  $("#impactList").innerHTML = impact.map(([k, v]) => `<li><span>${k}</span><span class="iv">${v}</span></li>`).join("");

  const maps = [
    [r.images.original, "Original Satellite Image", "tag-input"],
    [r.images.segmentation, "AI Land-Cover Segmentation", "tag-seg"],
    [r.images.plantable, "Detected Plantable Areas (blue)", "tag-plant"],
    [r.images.recommendation, "Recommended Planting Plan", "tag-plan"],
  ];
  $("#mapGrid").innerHTML = maps.map(([src, title, tag]) => `
    <div class="map-card">
      <h4>${title}<span class="tag ${tag}">${r.metrics.image_size[0]}&times;${r.metrics.image_size[1]}</span></h4>
      <div class="map-thumb">
        <img src="${src}" alt="${title}" loading="lazy" data-full="${src}" data-title="${title}">
        <div class="map-expand" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/>
          </svg>
        </div>
      </div>
    </div>`).join("");

  renderBA("baHeat", "Heat score", m.heat_score_current, m.projected_heat_score, "", 100);
  renderBA("baTemp", "Temperature", m.temp_current_c, m.projected_temp_c, "&deg;C", null);
  renderBA("baCanopy", "Canopy %", m.canopy_pct, m.projected_canopy_pct, "%", 100);

  const deltas = [
    [`&minus;${fmt(m.cooling_reduction_c, 2)}&deg;C`, "Estimated cooling from recommended planting"],
    [`+${fmt(m.recommended_planting_pct, 2)}%`, `Canopy added (${m.recommended_trees} trees)`],
    [`&minus;${fmt(m.heat_score_current - m.projected_heat_score)} pts`, "Heat score improvement"],
    [`+${Math.round(m.shade_improvement_m2).toLocaleString()} m&sup2;`, "New shade coverage"],
  ];
  $("#deltaGrid").innerHTML = deltas.map(([v, l]) =>
    `<div class="delta-card"><div class="d-value">${v}</div><div class="d-label">${l}</div></div>`).join("");

  $("#uploadPanel").hidden = true;
  dashboard.hidden = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderBA(id, label, before, after, unit, maxOverride) {
  const maxV = maxOverride !== null && maxOverride !== undefined
    ? maxOverride
    : Math.max(before, after) * 1.15 || 1;
  const el = $("#" + id);
  el.innerHTML = `
    <div class="ba-row ba-before">
      <div class="ba-label">Before</div>
      <div class="ba-track"><div class="ba-fill" style="width:${(before / maxV) * 100}%">${fmt(before)}${unit}</div></div>
    </div>
    <div class="ba-row ba-after">
      <div class="ba-label">After</div>
      <div class="ba-track"><div class="ba-fill" style="width:0%">${fmt(after)}${unit}</div></div>
    </div>`;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      el.querySelector(".ba-after .ba-fill").style.width = `${(after / maxV) * 100}%`;
    });
  });
}
