import { pipeline, env, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";
import { FilesetResolver, FaceDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* ---------- Config ---------- */
const MODEL_ID = "onnx-community/face-emotion-detection-ONNX";
const MODEL_FILE = "model_q4.onnx";           // dtype "q4" -> onnx/model_q4.onnx
const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MP_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const INFER_INTERVAL_MS = 150;
const MISSES_BEFORE_RESET = 6;                  // frames without a face before scores are cleared
const CROP_PADDING = 0.15;                      // extra context around the detector box

// Model card order: ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'].
// config.json only has LABEL_0..LABEL_6, so this order comes from the model card.
const MODEL_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"];
const DISPLAY = { Happy: "Happy", Sad: "Sad", Angry: "Angry", Fear: "Fearful", Surprise: "Surprised", Neutral: "Neutral", Disgust: "Disgusted" };
const DISPLAY_ORDER = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"];

/* ---------- DOM ---------- */
const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), stage = $("stage");
const startBtn = $("startBtn"), stopBtn = $("stopBtn");
const statusEl = $("status"), mainLabel = $("mainLabel"), mainConf = $("mainConf");
const loader = $("loader"), loadBar = $("loadBar"), errorEl = $("error");
const octx = overlay.getContext("2d");

const rows = {};
for (const key of DISPLAY_ORDER) {
  const li = document.createElement("li");
  li.className = "row";
  li.innerHTML = `<span>${DISPLAY[key]}</span><span class="pct">0.0%</span>`;
  $("scores").appendChild(li);
  rows[key] = { li, pct: li.querySelector(".pct") };
}

const setStatus = (t) => (statusEl.textContent = t);
function showError(msg) { errorEl.textContent = msg; errorEl.hidden = false; setStatus(msg); }
function setStep(name, state) {
  const li = document.querySelector(`#steps [data-step="${name}"]`);
  li.classList.remove("active", "done");
  if (state) li.classList.add(state);
}

/* ---------- Module: face detection (MediaPipe) ---------- */
const FaceModule = {
  detector: null,
  async init() {
    const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
    this.detector = await FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MP_MODEL },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });
  },
  largestFace(timestamp) {
    const { detections } = this.detector.detectForVideo(video, timestamp);
    let best = null, bestArea = 0;
    for (const d of detections) {
      const b = d.boundingBox;
      if (b && b.width * b.height > bestArea) { best = b; bestArea = b.width * b.height; }
    }
    return best;
  },
};

/* ---------- Module: expression classification (Transformers.js) ---------- */
const ExpressionModule = {
  classifier: null,
  cropCanvas: Object.assign(document.createElement("canvas"), { width: 224, height: 224 }),
  async init(onProgress) {
    env.allowLocalModels = false;
    env.useBrowserCache = true;                 // Cache API: file is downloaded once
    let sawModel = false;
    this.classifier = await pipeline("image-classification", MODEL_ID, {
      dtype: "q4",
      device: "wasm",
      progress_callback: (p) => {
        if (p.file && p.file.endsWith(MODEL_FILE)) sawModel = true;
        else if (p.file && p.file.endsWith(".onnx")) throw new Error(`Unexpected model file requested: ${p.file}`);
        onProgress(p);
      },
    });
    // Cached loads can skip progress events, so confirm via the browser's resource log too.
    const fetched = performance.getEntriesByType("resource").map((r) => r.name).filter((n) => n.includes(".onnx"));
    if (!sawModel && fetched.length && !fetched.some((n) => n.includes(MODEL_FILE))) {
      throw new Error("Classifier did not resolve to model_q4.onnx.");
    }
  },
  async warmup() {
    const c = this.cropCanvas.getContext("2d");
    c.fillStyle = "#808080"; c.fillRect(0, 0, 224, 224);
    await this.classifier(RawImage.fromCanvas(this.cropCanvas), { top_k: 7 });
  },
  /** Crop a padded square around the face into one reused 224x224 canvas, then classify. */
  async classify(box) {
    const side = Math.max(box.width, box.height) * (1 + CROP_PADDING * 2);
    const cx = box.originX + box.width / 2, cy = box.originY + box.height / 2;
    const sx = Math.max(0, cx - side / 2), sy = Math.max(0, cy - side / 2);
    const sw = Math.min(video.videoWidth - sx, side), sh = Math.min(video.videoHeight - sy, side);
    this.cropCanvas.getContext("2d").drawImage(video, sx, sy, sw, sh, 0, 0, 224, 224);
    const out = await this.classifier(RawImage.fromCanvas(this.cropCanvas), { top_k: 7 });
    const scores = {};
    for (const { label, score } of out) {
      const idx = Number(label.replace("LABEL_", ""));
      const name = MODEL_LABELS[idx];
      if (!name) throw new Error(`Unexpected classifier label: ${label}`);
      scores[name] = score;
    }
    if (Object.keys(scores).length !== MODEL_LABELS.length) throw new Error("Invalid classifier output.");
    return scores;
  },
};

/* ---------- UI helpers ---------- */
function renderScores(scores) {
  let top = null;
  if (scores) top = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  for (const key of DISPLAY_ORDER) {
    const v = scores ? scores[key] : 0;
    rows[key].pct.textContent = `${(v * 100).toFixed(1)}%`;
    rows[key].li.classList.toggle("top", !!top && top[0] === key);
  }
  mainLabel.textContent = top ? DISPLAY[top[0]].toUpperCase() : "—";
  mainConf.textContent = top ? `${(top[1] * 100).toFixed(1)}%` : "";
}
function drawBox(box) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!box) return;
  octx.lineWidth = Math.max(2, overlay.width / 240);
  octx.strokeStyle = "#4fe3c1";
  octx.strokeRect(box.originX, box.originY, box.width, box.height);
}

/* ---------- Detection loop ---------- */
let stream = null, rafId = 0, running = false, busy = false, lastInfer = 0, lastTs = -1, misses = 0;

function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.videoWidth === 0) return;
  const now = performance.now();
  if (now <= lastTs) return;                    // MediaPipe needs strictly increasing timestamps
  lastTs = now;

  let box = null;
  try { box = FaceModule.largestFace(now); }
  catch (e) { console.error(e); showError("Face detection failed. Try restarting the camera."); stopCamera(true); return; }
  drawBox(box);

  if (!box) {
    if (++misses >= MISSES_BEFORE_RESET) { renderScores(null); if (!busy) setStatus("No face detected."); }
    return;
  }
  misses = 0;
  if (busy || now - lastInfer < INFER_INTERVAL_MS) return;
  busy = true; lastInfer = now;
  setStatus("Classifying expression...");
  ExpressionModule.classify(box)
    .then((scores) => { if (running && misses === 0) { renderScores(scores); setStatus("Face detected."); } })
    .catch((e) => { console.error(e); setStatus("Classification error: " + e.message); })
    .finally(() => { busy = false; });
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return showError("This browser does not support camera access.");
  startBtn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
  } catch (e) {
    startBtn.disabled = false;
    return showError(e.name === "NotAllowedError" ? "Camera permission denied. Allow camera access and try again."
      : e.name === "NotFoundError" ? "No webcam found." : "Could not start camera: " + e.message);
  }
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  stage.classList.add("live");
  stopBtn.disabled = false;
  errorEl.hidden = true;
  running = true; misses = 0; lastInfer = 0; lastTs = -1;
  setStatus("No face detected.");
  rafId = requestAnimationFrame(loop);
}

function stopCamera(keepMessage = false) {
  running = false;
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null; video.srcObject = null;
  stage.classList.remove("live");
  octx.clearRect(0, 0, overlay.width, overlay.height);
  renderScores(null);
  startBtn.disabled = false; stopBtn.disabled = true;
  if (!keepMessage) setStatus("Camera stopped.");
}

/* ---------- Startup: both models load concurrently, once ---------- */
async function init() {
  if (!("WebAssembly" in window)) return showError("This browser does not support WebAssembly.");
  navigator.storage?.persist?.().catch(() => {});   // ask for persistent cache storage
  renderScores(null);
  loadBar.className = "bar indeterminate";
  setStep("mediapipe", "active"); setStep("classifier", "active");

  const onProgress = (p) => {
    // Only real byte progress for the model file is shown as a percentage.
    if (p.status === "progress" && p.file?.endsWith(MODEL_FILE) && typeof p.progress === "number") {
      loadBar.className = "bar determinate";
      loadBar.firstElementChild.style.width = `${p.progress.toFixed(1)}%`;
    }
  };

  const mp = FaceModule.init().then(() => setStep("mediapipe", "done"))
    .catch((e) => { throw new Error("MediaPipe failed to initialize: " + e.message); });
  const clf = ExpressionModule.init(onProgress).then(() => setStep("classifier", "done"))
    .catch((e) => { throw new Error("Q4 classifier failed to load: " + e.message); });

  try {
    await Promise.all([mp, clf]);
    setStep("engine", "active");
    loadBar.className = "bar indeterminate";
    await ExpressionModule.warmup();             // first run compiles the ONNX session
    setStep("engine", "done"); setStep("ready", "done");
    loader.classList.add("ready");
    startBtn.disabled = false;
    setStatus("System ready. Click Start Camera.");
  } catch (e) {
    console.error(e);
    loadBar.style.display = "none";
    showError(e.message + " Check your connection and reload.");
  }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", () => stopCamera());
window.addEventListener("pagehide", () => stopCamera(true));
init();