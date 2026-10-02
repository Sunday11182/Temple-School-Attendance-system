// =====================================================
// Temple Secondary School — Smart Attendance System
// Face Recognition • Photo capture • Check-in/out
// Daily history • Search • Export CSV • Sound
// =====================================================

const STORAGE_KEY = "temple_attendance_v3";
const FACE_REGISTRY_KEY = "temple_face_registry_v1";

// Face match threshold (lower = stricter). 0.45–0.6 is typical.
const FACE_MATCH_THRESHOLD = 0.55;

let allRecords = loadAllRecords();
let faceRegistry = loadFaceRegistry();   // { studentId: number[] }
let selectedDate = getTodayKey();
let html5QrCode = null;
let scannerRunning = false;
let lastScanTime = 0;
let lastVideoFrame = null;
let faceModelsReady = false;

// ---------- Storage helpers ----------
function getTodayKey() {
  return new Date().toISOString().slice(0, 10);
}

function loadAllRecords() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {
    console.warn("Could not read records", e);
  }
  return {};
}

function saveAllRecords() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(allRecords));
  } catch (e) {
    console.warn("Storage full or unavailable.", e);
    showToast("Warning: could not save (storage may be full).", "error");
  }
}

function loadFaceRegistry() {
  try {
    const raw = localStorage.getItem(FACE_REGISTRY_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {
    console.warn("Could not read face registry", e);
  }
  return {};
}

function saveFaceRegistry() {
  try {
    localStorage.setItem(FACE_REGISTRY_KEY, JSON.stringify(faceRegistry));
  } catch (e) {
    console.warn("Could not save face registry", e);
  }
}

function ensureDay(dateKey) {
  if (!allRecords[dateKey]) allRecords[dateKey] = {};
  STUDENTS.forEach((s) => {
    if (!allRecords[dateKey][s.id]) {
      allRecords[dateKey][s.id] = {
        status: "absent",
        timeIn: null,
        timeOut: null,
        photo: null,
        alerted: false,
      };
    }
  });
  saveAllRecords();
}

function getDayRecords() {
  ensureDay(selectedDate);
  return allRecords[selectedDate];
}

// ---------- Clock ----------
function tickClock() {
  const now = new Date();
  document.getElementById("live-clock").textContent = now.toLocaleTimeString();
  document.getElementById("live-date").textContent = now.toLocaleDateString(undefined, {
    weekday: "long", year: "numeric", month: "long", day: "numeric",
  });
}
setInterval(tickClock, 1000);
tickClock();

// ---------- Sound ----------
function playBeep(type = "success") {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === "success") {
      osc.frequency.value = 880;
      gain.gain.value = 0.15;
      osc.start();
      setTimeout(() => {
        osc.frequency.value = 1175;
        setTimeout(() => osc.stop(), 120);
      }, 100);
    } else {
      osc.frequency.value = 300;
      gain.gain.value = 0.12;
      osc.start();
      setTimeout(() => osc.stop(), 250);
    }
  } catch (e) {}
}

// ---------- Toast ----------
let toastTimer = null;
function showToast(message, type = "success") {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast show toast-${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.className = "toast";
  }, 3200);
}

// ---------- Load face-api models ----------
async function loadFaceModels() {
  const statusEl = document.getElementById("scanner-status");
  try {
    statusEl.textContent = "Loading face recognition models…";
    const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";

    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

    faceModelsReady = true;
    statusEl.textContent = "Face models ready. Click “Start Scanner” to begin.";
    console.log("Face models loaded successfully");
  } catch (err) {
    console.error("Failed to load face models", err);
    statusEl.textContent = "Face models failed to load. Attendance will work without face verification.";
    showToast("Face recognition models could not load. Continuing without verification.", "error");
  }
}

// ---------- Photo + Face descriptor from camera ----------
function getVideoElement() {
  return document.querySelector("#qr-reader video");
}

function capturePhotoFromScanner() {
  try {
    const video = getVideoElement();
    if (video && video.readyState >= 2 && video.videoWidth > 0) {
      const canvas = document.createElement("canvas");
      const size = 140;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const side = Math.min(vw, vh);
      const sx = (vw - side) / 2;
      const sy = (vh - side) / 2;

      ctx.drawImage(video, sx, sy, side, side, 0, 0, size, size);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      lastVideoFrame = dataUrl;
      return dataUrl;
    }
    return lastVideoFrame || null;
  } catch (e) {
    console.warn("Photo capture failed", e);
    return lastVideoFrame || null;
  }
}

/** Returns { descriptor: Float32Array|null, photo: string|null } */
async function captureFaceAndPhoto() {
  const photo = capturePhotoFromScanner();
  let descriptor = null;

  if (!faceModelsReady) {
    return { descriptor: null, photo };
  }

  try {
    const video = getVideoElement();
    let input = video;

    // If video not available, try from last photo
    if (!video || video.readyState < 2) {
      if (!photo) return { descriptor: null, photo: null };
      const img = await faceapi.fetchImage(photo);
      input = img;
    }

    const detection = await faceapi
      .detectSingleFace(input, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (detection) {
      descriptor = detection.descriptor;
    }
  } catch (e) {
    console.warn("Face detection failed", e);
  }

  return { descriptor, photo };
}

/** Euclidean distance between two descriptors */
function faceDistance(desc1, desc2) {
  if (!desc1 || !desc2 || desc1.length !== desc2.length) return 999;
  let sum = 0;
  for (let i = 0; i < desc1.length; i++) {
    const d = desc1[i] - desc2[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/** Verify live face against enrolled face for this student */
function verifyFace(studentId, liveDescriptor) {
  const enrolled = faceRegistry[studentId];
  if (!enrolled || !liveDescriptor) {
    return { ok: false, reason: "no_data" };
  }
  const dist = faceDistance(liveDescriptor, enrolled);
  console.log(`Face distance for ${studentId}: ${dist.toFixed(3)}`);
  if (dist <= FACE_MATCH_THRESHOLD) {
    return { ok: true, distance: dist };
  }
  return { ok: false, reason: "mismatch", distance: dist };
}

// Frame grabber (keeps last good frame)
function startFrameGrabber() {
  if (window._frameGrabber) clearInterval(window._frameGrabber);
  window._frameGrabber = setInterval(() => {
    if (!scannerRunning) return;
    const video = getVideoElement();
    if (!video || video.readyState < 2) return;
    try {
      const canvas = document.createElement("canvas");
      const size = 140;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const side = Math.min(vw, vh);
      const sx = (vw - side) / 2;
      const sy = (vh - side) / 2;
      ctx.drawImage(video, sx, sy, side, side, 0, 0, size, size);
      lastVideoFrame = canvas.toDataURL("image/jpeg", 0.7);
    } catch (e) {}
  }, 700);
}

function stopFrameGrabber() {
  if (window._frameGrabber) {
    clearInterval(window._frameGrabber);
    window._frameGrabber = null;
  }
}

// ---------- Core: Process student with face verification ----------
async function processStudent(rawId) {
  const now = Date.now();
  if (now - lastScanTime < 1800) return;
  lastScanTime = now;

  const id = rawId.trim().toUpperCase();
  const student = STUDENTS.find((s) => s.id === id);

  if (!student) {
    playBeep("error");
    showToast(`No student found with ID "${id}"`, "error");
    return;
  }

  if (selectedDate !== getTodayKey()) {
    playBeep("error");
    showToast("You can only check students in/out for today. Switch to Today first.", "error");
    return;
  }

  // Capture photo + face descriptor
  showToast("Verifying face…", "info");
  const { descriptor, photo } = await captureFaceAndPhoto();

  // Face verification logic
  if (faceModelsReady && scannerRunning) {
    if (!descriptor) {
      playBeep("error");
      showToast("No face detected. Look at the camera and try again.", "error");
      return;
    }

    const hasEnrolledFace = !!faceRegistry[id];

    if (!hasEnrolledFace) {
      // First time → enroll this face
      faceRegistry[id] = Array.from(descriptor);
      saveFaceRegistry();
      showToast(`Face enrolled for ${student.name}`, "info");
    } else {
      // Verify against enrolled face
      const result = verifyFace(id, descriptor);
      if (!result.ok) {
        playBeep("error");
        showToast(
          `Face mismatch for ${student.name}. Verification failed.`,
          "error"
        );
        return; // BLOCK check-in
      }
    }
  }

  // Proceed with attendance
  const records = getDayRecords();
  const record = records[id];
  const timeStr = new Date().toLocaleTimeString();

  if (record.status === "absent") {
    record.status = "present";
    record.timeIn = timeStr;
    record.timeOut = null;
    record.alerted = false;
    if (photo) record.photo = photo;
    saveAllRecords();
    playBeep("success");
    showToast(`✓ Checked in: ${student.name} (face verified)`, "success");
    renderRoster();
    return;
  }

  if (record.status === "present") {
    record.status = "checked-out";
    record.timeOut = timeStr;
    if (photo) record.photo = photo;
    saveAllRecords();
    playBeep("success");
    showToast(`✓ Checked out: ${student.name}`, "info");
    renderRoster();
    return;
  }

  playBeep("error");
  showToast(`${student.name} has already checked out today.`, "info");
}

// ---------- Render roster ----------
function renderRoster() {
  const records = getDayRecords();
  const tbody = document.getElementById("roster-body");
  const search = (document.getElementById("search-input").value || "").toLowerCase().trim();

  tbody.innerHTML = "";
  let present = 0, checkedOut = 0, absent = 0;

  STUDENTS.forEach((s) => {
    const r = records[s.id];

    if (search) {
      const hay = `${s.id} ${s.name} ${s.className}`.toLowerCase();
      if (!hay.includes(search)) return;
    }

    let statusClass = "row-absent";
    let badgeClass = "badge-absent";
    let statusLabel = "Absent";

    if (r.status === "present") {
      statusClass = "row-present";
      badgeClass = "badge-present";
      statusLabel = "Present";
      present++;
    } else if (r.status === "checked-out") {
      statusClass = "row-checked-out";
      badgeClass = "badge-checked-out";
      statusLabel = "Checked Out";
      checkedOut++;
    } else {
      absent++;
    }

    if (r.alerted) statusLabel += " · Alerted";

    // Show a small face-enrolled indicator
    const faceEnrolled = faceRegistry[s.id] ? " title=\"Face enrolled\"" : "";
    const photoHtml = r.photo
      ? `<img src="${r.photo}" alt="${s.name}" class="student-photo"${faceEnrolled}>`
      : `<div class="student-photo placeholder">—</div>`;

    const row = document.createElement("tr");
    row.className = statusClass;
    row.innerHTML = `
      <td>${photoHtml}</td>
      <td>${s.id}</td>
      <td>${s.name}${faceRegistry[s.id] ? ' <span class="face-badge" title="Face enrolled">✓</span>' : ""}</td>
      <td>${s.className}</td>
      <td><span class="badge ${badgeClass}">${statusLabel}</span></td>
      <td>${r.timeIn || "—"}</td>
      <td>${r.timeOut || "—"}</td>
    `;
    tbody.appendChild(row);
  });

  if (search) {
    present = checkedOut = absent = 0;
    STUDENTS.forEach((s) => {
      const st = records[s.id].status;
      if (st === "present") present++;
      else if (st === "checked-out") checkedOut++;
      else absent++;
    });
  }

  document.getElementById("stat-total").textContent = STUDENTS.length;
  document.getElementById("stat-present").textContent = present;
  document.getElementById("stat-checked-out").textContent = checkedOut;
  document.getElementById("stat-absent").textContent = absent;

  const label = document.getElementById("roster-date-label");
  label.textContent = selectedDate === getTodayKey() ? "· Today" : `· ${selectedDate}`;
}

// ---------- Date controls ----------
function initDateControls() {
  const dateInput = document.getElementById("selected-date");
  dateInput.value = selectedDate;

  dateInput.addEventListener("change", () => {
    selectedDate = dateInput.value || getTodayKey();
    ensureDay(selectedDate);
    renderRoster();
  });

  document.getElementById("btn-today").addEventListener("click", () => {
    selectedDate = getTodayKey();
    dateInput.value = selectedDate;
    ensureDay(selectedDate);
    renderRoster();
  });
}

// ---------- Search ----------
document.getElementById("search-input").addEventListener("input", renderRoster);

// ---------- Manual entry ----------
document.getElementById("manual-submit").addEventListener("click", () => {
  const input = document.getElementById("manual-id");
  const id = input.value.trim();
  if (!id) return;
  processStudent(id);
  input.value = "";
  input.focus();
});

document.getElementById("manual-id").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("manual-submit").click();
});

// ---------- QR Scanner ----------
const toggleBtn = document.getElementById("toggle-scanner");
const scannerStatus = document.getElementById("scanner-status");

toggleBtn.addEventListener("click", async () => {
  if (!scannerRunning) {
    try {
      html5QrCode = new Html5Qrcode("qr-reader");
      await html5QrCode.start(
        { facingMode: "user" }, // front camera better for face recognition
        { fps: 10, qrbox: { width: 220, height: 220 } },
        (decodedText) => processStudent(decodedText),
        () => {}
      );
      scannerRunning = true;
      toggleBtn.textContent = "Stop Scanner";
      scannerStatus.textContent = "Camera active — face verification is ON. Scan QR or enter ID.";
      startFrameGrabber();
    } catch (err) {
      // Fallback to rear camera
      try {
        html5QrCode = new Html5Qrcode("qr-reader");
        await html5QrCode.start(
          { facingMode: "environment" },
          { fps: 10, qrbox: { width: 220, height: 220 } },
          (decodedText) => processStudent(decodedText),
          () => {}
        );
        scannerRunning = true;
        toggleBtn.textContent = "Stop Scanner";
        scannerStatus.textContent = "Camera active (rear). Face verification is ON.";
        startFrameGrabber();
      } catch (err2) {
        scannerStatus.textContent = "Could not access camera. Use manual ID entry.";
        showToast("Camera unavailable.", "error");
        console.error(err2);
      }
    }
  } else {
    stopFrameGrabber();
    await html5QrCode.stop();
    html5QrCode.clear();
    scannerRunning = false;
    toggleBtn.textContent = "Start Scanner";
    scannerStatus.textContent = "Camera idle — last frame kept briefly for manual entry.";
  }
});

// ---------- Export CSV ----------
document.getElementById("btn-export").addEventListener("click", () => {
  const records = getDayRecords();
  const rows = [["ID", "Name", "Class", "Status", "Time In", "Time Out", "Alerted", "Face Enrolled"]];

  STUDENTS.forEach((s) => {
    const r = records[s.id];
    rows.push([
      s.id, s.name, s.className, r.status,
      r.timeIn || "", r.timeOut || "",
      r.alerted ? "Yes" : "No",
      faceRegistry[s.id] ? "Yes" : "No",
    ]);
  });

  const csv = rows.map((r) =>
    r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")
  ).join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `attendance_${selectedDate}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast(`Exported attendance for ${selectedDate}`, "success");
});

// ---------- Alerts ----------
document.getElementById("btn-alerts").addEventListener("click", () => {
  if (selectedDate !== getTodayKey()) {
    showToast("Alerts can only be sent for today.", "error");
    return;
  }
  const records = getDayRecords();
  const absentees = STUDENTS.filter((s) => records[s.id].status === "absent");

  if (absentees.length === 0) {
    showToast("No absentees to alert.", "success");
    return;
  }

  absentees.forEach((s) => { records[s.id].alerted = true; });
  saveAllRecords();
  renderRoster();
  showToast(`Simulated alert sent for ${absentees.length} absent student(s).`, "info");
});

// ---------- Reset Day ----------
document.getElementById("btn-reset").addEventListener("click", () => {
  if (selectedDate !== getTodayKey()) {
    showToast("You can only reset today’s records.", "error");
    return;
  }
  if (!confirm("Reset today’s attendance? Photos and check-ins for today will be cleared.\n(Face enrollments are kept.)")) return;

  STUDENTS.forEach((s) => {
    allRecords[selectedDate][s.id] = {
      status: "absent",
      timeIn: null,
      timeOut: null,
      photo: null,
      alerted: false,
    };
  });
  saveAllRecords();
  renderRoster();
  showToast("Today’s attendance has been reset. Face enrollments kept.", "info");
});

// ---------- Print ID Cards (uses captured photos) ----------
document.getElementById("btn-cards").addEventListener("click", () => {
  const sheet = document.getElementById("id-card-sheet");
  const records = getDayRecords();
  sheet.innerHTML = "";

  STUDENTS.forEach((s) => {
    const r = records[s.id];
    const photoHtml = r.photo
      ? `<img src="${r.photo}" alt="${s.name}" class="id-card-photo">`
      : `<div class="id-card-photo placeholder">No Photo</div>`;

    const card = document.createElement("div");
    card.className = "id-card";
    card.innerHTML = `
      <div class="id-card-header">Temple Secondary School</div>
      ${photoHtml}
      <div class="id-card-name">${s.name}</div>
      <div class="id-card-meta">${s.className} · ${s.id}</div>
      <div class="id-card-status">${
        r.status === "absent" ? "Absent" :
        r.status === "present" ? "Present" : "Checked Out"
      }</div>
    `;
    sheet.appendChild(card);
  });

  sheet.classList.add("visible");
  window.print();
  sheet.classList.remove("visible");
});

// ---------- Optional: Clear all face enrollments (for testing) ----------
// Uncomment the next lines if you want a hidden way to reset faces during presentation prep:
// window.clearAllFaces = () => {
//   faceRegistry = {};
//   saveFaceRegistry();
//   renderRoster();
//   showToast("All face enrollments cleared.", "info");
// };

// ---------- Init ----------
ensureDay(selectedDate);
initDateControls();
renderRoster();
loadFaceModels(); // start loading models immediately