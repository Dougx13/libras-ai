import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs';

const MODEL_URL = '/models/libras-static-v1.json';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm';
const CAMERA_POLL_MS = 90;
const CONFIDENCE_MIN = 0.62;
const CONFIDENCE_MARGIN_MIN = 0.08;
const STABLE_FRAMES = 7;
const STABLE_VOTES_MIN = 5;
const STATIC_SMOOTHING_FRAMES = 3;
const STATIC_SMOOTHING_RESET_DELTA = 0.18;
const RELEASE_MS = 550;
const WORD_BREAK_MS = 1500;
const DYNAMIC_LETTERS = ['Ç', 'H', 'J', 'K', 'X', 'Y', 'Z'];
const DYNAMIC_FRAME_COUNT = 24;
const DYNAMIC_SAMPLE_MAX_MS = 3600;
const DYNAMIC_MIN_SAMPLES_PER_CLASS = 5;
const DYNAMIC_MOTION_THRESHOLD = 0.02;
const DYNAMIC_START_MOTION_FRAMES = 2;
const DYNAMIC_END_PAUSE_MS = 700;
const DYNAMIC_MAX_GESTURE_MS = 3200;
const DYNAMIC_DISTANCE_LIMIT = 0.95;
const DYNAMIC_DB_NAME = 'libras-ai-dynamic-v1';
const DYNAMIC_DB_STORE = 'samples';
const DYNAMIC_MOVEMENT_POINTS = [0, 4, 8, 12, 16, 20];
const DYNAMIC_POSE_FEATURE_COUNT = 63;
const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const ctx = overlay.getContext('2d');

let classifier;
let landmarker;
let stream;
let running = false;
let lastFrameAt = -Infinity;
let recentPredictions = new Map();
let lastConfirmedLetters = new Map();
let staticLandmarkHistory = new Map();
let handMissingSince = 0;
let lastHandSeenAt = 0;
let missingHandSince = new Map();
let nextCameraPoll = 0;
let toastTimer;
let dynamicSamples = [];
let recordingSample = null;
let dynamicTracks = new Map();
let dynamicFinishedThisFrame = false;
let dynamicLastPrediction = null;
let dynamicModeEnabled = false;

function notify(message) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2800);
}

async function loadClassifier() {
  if (classifier) return classifier;
  $('modelStatus').textContent = 'CARREGANDO MODELO';
  const response = await fetch(MODEL_URL, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Modelo não encontrado (${response.status}).`);
  const model = await response.json();
  if (model.format !== 'libras-random-forest-v1' || !Array.isArray(model.trees) || !Array.isArray(model.classes)) {
    throw new Error('O arquivo do classificador tem um formato inválido.');
  }
  classifier = model;
  $('modelStatus').textContent = '21 LETRAS · 2 MÃOS';
  $('modelStatus').classList.add('model-ready');
  return classifier;
}

async function loadHandDetector() {
  if (landmarker) return landmarker;
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' },
    runningMode: 'VIDEO', numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.45,
  };
  try {
    landmarker = await HandLandmarker.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } });
  } catch {
    landmarker = await HandLandmarker.createFromOptions(vision, options);
  }
  return landmarker;
}

function normalizeLandmarks(points) {
  const wrist = points[0];
  const middleMcp = points[9];
  const scale = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y, middleMcp.z - wrist.z) || 1e-6;
  return points.flatMap((point) => [(point.x - wrist.x) / scale, (point.y - wrist.y) / scale, (point.z - wrist.z) / scale]);
}

function palmScale(points) {
  const wrist = points[0];
  const middleMcp = points[9];
  return Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y, middleMcp.z - wrist.z) || 1e-6;
}

function cloneLandmarks(points) {
  return points.map(({ x, y, z }) => ({ x, y, z }));
}

function makeDynamicFrame(points, origin) {
  const pose = normalizeLandmarks(points);
  const path = DYNAMIC_MOVEMENT_POINTS.flatMap((index) => {
    const point = points[index];
    return [
      (point.x - origin.x) / origin.scale,
      (point.y - origin.y) / origin.scale,
      (point.z - origin.z) / origin.scale,
    ];
  });
  return [...pose, ...path];
}

function resampleSequence(sequence, targetCount = DYNAMIC_FRAME_COUNT) {
  if (sequence.length === targetCount) return sequence;
  if (sequence.length < 2) return [];
  const sampled = [];
  for (let index = 0; index < targetCount; index++) {
    const position = index * (sequence.length - 1) / (targetCount - 1);
    const lower = Math.floor(position);
    const upper = Math.min(sequence.length - 1, lower + 1);
    const ratio = position - lower;
    sampled.push(sequence[lower].map((value, feature) => value + (sequence[upper][feature] - value) * ratio));
  }
  return sampled;
}

function dynamicFrameDistance(first, second) {
  let poseSum = 0;
  let pathSum = 0;
  for (let index = 0; index < DYNAMIC_POSE_FEATURE_COUNT; index++) {
    const difference = first[index] - second[index];
    poseSum += difference * difference;
  }
  for (let index = DYNAMIC_POSE_FEATURE_COUNT; index < first.length; index++) {
    const difference = first[index] - second[index];
    pathSum += difference * difference;
  }
  const poseDistance = Math.sqrt(poseSum / DYNAMIC_POSE_FEATURE_COUNT);
  const pathDistance = Math.sqrt(pathSum / Math.max(1, first.length - DYNAMIC_POSE_FEATURE_COUNT));
  // Motion contributes more than one frame's hand shape so signs with similar poses
  // but different trajectories (for example J and Z) can be told apart.
  return poseDistance * 0.42 + pathDistance * 0.58;
}

function smoothDynamicSequence(sequence) {
  return sequence.map((frame, frameIndex) => {
    const start = Math.max(0, frameIndex - 1);
    const end = Math.min(sequence.length - 1, frameIndex + 1);
    const frameCount = end - start + 1;
    return frame.map((_, featureIndex) => {
      let total = 0;
      for (let index = start; index <= end; index++) total += sequence[index][featureIndex];
      return total / frameCount;
    });
  });
}

function dynamicTimeWarping(first, second) {
  const rows = first.length + 1;
  const columns = second.length + 1;
  const costs = Array.from({ length: rows }, () => new Float64Array(columns).fill(Infinity));
  costs[0][0] = 0;
  const band = Math.max(2, Math.floor(Math.max(first.length, second.length) * 0.18));
  for (let row = 1; row < rows; row++) {
    const firstColumn = Math.max(1, row - band);
    const lastColumn = Math.min(columns - 1, row + band);
    for (let column = firstColumn; column <= lastColumn; column++) {
      const previous = Math.min(costs[row - 1][column], costs[row][column - 1], costs[row - 1][column - 1]);
      costs[row][column] = dynamicFrameDistance(first[row - 1], second[column - 1]) + previous;
    }
  }
  return costs[rows - 1][columns - 1] / (first.length + second.length);
}

function dynamicClassCounts(side) {
  return Object.fromEntries(DYNAMIC_LETTERS.map((letter) => [
    letter,
    dynamicSamples.filter((sample) => sample.label === letter && sample.side === side).length,
  ]));
}

function dynamicReady(side) {
  const counts = dynamicClassCounts(side);
  return DYNAMIC_LETTERS.some((letter) => counts[letter] >= DYNAMIC_MIN_SAMPLES_PER_CLASS);
}

function updateDynamicStatus(message) {
  const side = $('dynamicSide')?.value || 'Right';
  const counts = dynamicClassCounts(side);
  const readyLetters = DYNAMIC_LETTERS.filter((letter) => counts[letter] >= DYNAMIC_MIN_SAMPLES_PER_CLASS);
  const missingLetters = DYNAMIC_LETTERS.filter((letter) => counts[letter] < DYNAMIC_MIN_SAMPLES_PER_CLASS);
  $('dynamicCount').textContent = `${side === 'Right' ? 'DIREITA' : 'ESQUERDA'} · ${DYNAMIC_LETTERS.map((letter) => `${letter} ${counts[letter]}`).join(' · ')}`;
  if (message) $('dynamicStatus').textContent = message;
  else if (!dynamicModeEnabled) $('dynamicStatus').textContent = 'Modo desligado. Ative “Letras com movimento” para gravar ou reconhecer essas letras.';
  else if (readyLetters.length) $('dynamicStatus').textContent = `Reconhecimento ativo para ${readyLetters.join(', ')}. Grave ${DYNAMIC_MIN_SAMPLES_PER_CLASS} exemplos de ${missingLetters.join(', ') || 'mais variações'} para habilitar essas letras.`;
  else $('dynamicStatus').textContent = `Grave pelo menos ${DYNAMIC_MIN_SAMPLES_PER_CLASS} exemplos de cada letra com a mão ${side === 'Right' ? 'direita' : 'esquerda'}.`;
}

function toggleDynamicMode() {
  dynamicModeEnabled = !dynamicModeEnabled;
  const button = $('dynamicModeToggle');
  button.setAttribute('aria-pressed', String(dynamicModeEnabled));
  button.classList.toggle('active', dynamicModeEnabled);
  button.textContent = dynamicModeEnabled ? 'Desativar letras com movimento' : 'Ativar letras com movimento';
  $('dynamicModeState').textContent = dynamicModeEnabled
    ? 'ATIVADO · letras estáticas pausadas'
    : 'DESATIVADO · reconhecimento estático';

  if (!dynamicModeEnabled) {
    recordingSample = null;
    dynamicTracks.clear();
    $('recordDynamicButton').textContent = '● Gravar exemplo';
  }
  $('recordDynamicButton').disabled = !(dynamicModeEnabled && running && landmarker && classifier);
  resetPoseLock();
  updateDynamicStatus();
  if (running) {
    $('cameraHint').textContent = dynamicModeEnabled
      ? 'Modo movimento ativo · faça um gesto ou grave exemplos'
      : 'Modo estático ativo · mantenha a pose parada';
    $('translation').textContent = '—';
    $('confidence').textContent = dynamicModeEnabled ? 'Aguardando letras com movimento' : 'Reconhecimento estático ativo';
    $('confidenceBar').style.width = '0%';
  }
}

function openDynamicDatabase() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) return reject(new Error('Armazenamento local não disponível neste navegador.'));
    const request = indexedDB.open(DYNAMIC_DB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(DYNAMIC_DB_STORE)) {
        request.result.createObjectStore(DYNAMIC_DB_STORE, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Não foi possível abrir o armazenamento local.'));
  });
}

async function saveDynamicSample(sample) {
  const database = await openDynamicDatabase();
  await new Promise((resolve, reject) => {
    const transaction = database.transaction(DYNAMIC_DB_STORE, 'readwrite');
    transaction.objectStore(DYNAMIC_DB_STORE).put(sample);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error || new Error('Falha ao salvar exemplo.'));
  });
  database.close();
}

async function loadDynamicSamples() {
  let fileSamples = [];
  try {
    const response = await fetch('/models/libras-dynamic-samples.json', { cache: 'no-store' });
    if (response.ok) {
      const dataset = await response.json();
      if (dataset.format === 'libras-dynamic-dtw-v1' && Array.isArray(dataset.samples)) fileSamples = dataset.samples.filter(isValidDynamicSample);
    }
  } catch (error) {
    console.warn('Não foi possível ler exemplos incluídos no projeto:', error);
  }
  try {
    const database = await openDynamicDatabase();
    const samples = await new Promise((resolve, reject) => {
      const request = database.transaction(DYNAMIC_DB_STORE, 'readonly').objectStore(DYNAMIC_DB_STORE).getAll();
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error || new Error('Falha ao ler exemplos salvos.'));
    });
    database.close();
    const merged = new Map([...fileSamples, ...samples.filter(isValidDynamicSample)].map((sample) => [sample.id, sample]));
    dynamicSamples = [...merged.values()];
    updateDynamicStatus();
  } catch (error) {
    console.warn('Exemplos dinâmicos ficarão apenas nesta sessão:', error);
    dynamicSamples = fileSamples;
    updateDynamicStatus('O navegador não permitiu salvar exemplos entre sessões. Você ainda pode gravar e baixar um JSON.');
  }
}

function isValidDynamicSample(sample) {
  return Boolean(sample && DYNAMIC_LETTERS.includes(sample.label)
    && ['Left', 'Right'].includes(sample.side)
    && Array.isArray(sample.frames) && sample.frames.length === DYNAMIC_FRAME_COUNT
    && sample.frames.every((frame) => Array.isArray(frame) && frame.length === 81 && frame.every(Number.isFinite)));
}

function classifyDynamicSequence(sequence, side) {
  const templates = dynamicSamples.filter((sample) => sample.side === side);
  if (!templates.length) return null;
  const smoothedSequence = smoothDynamicSequence(sequence);
  const distances = new Map(DYNAMIC_LETTERS.map((letter) => [letter, []]));
  for (const sample of templates) distances.get(sample.label).push(dynamicTimeWarping(smoothedSequence, smoothDynamicSequence(sample.frames)));
  const ranked = [...distances.entries()]
    .filter(([, values]) => values.length >= DYNAMIC_MIN_SAMPLES_PER_CLASS)
    .map(([label, values]) => {
      const nearest = values.sort((a, b) => a - b).slice(0, Math.min(5, values.length));
      return { label, distance: nearest.reduce((sum, value) => sum + value, 0) / nearest.length };
    })
    .sort((a, b) => a.distance - b.distance);
  if (!ranked.length) return null;
  if (ranked[0].distance > DYNAMIC_DISTANCE_LIMIT) return { label: ranked[0].label, distance: ranked[0].distance, rejected: true };
  if (ranked[1] && ranked[1].distance - ranked[0].distance < Math.max(0.035, ranked[0].distance * 0.08)) return { label: ranked[0].label, distance: ranked[0].distance, rejected: true, ambiguous: true };
  return { label: ranked[0].label, distance: ranked[0].distance, rejected: false };
}

function classify(vector) {
  const votes = new Float64Array(classifier.classes.length);
  for (const tree of classifier.trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      node = vector[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    const distribution = tree.value[node];
    let total = 0;
    for (const score of distribution) total += score;
    if (total > 0) for (let index = 0; index < votes.length; index++) votes[index] += distribution[index] / total;
  }
  let bestIndex = 0;
  for (let index = 1; index < votes.length; index++) if (votes[index] > votes[bestIndex]) bestIndex = index;
  let secondBest = 0;
  for (let index = 0; index < votes.length; index++) if (index !== bestIndex && votes[index] > secondBest) secondBest = votes[index];
  const confidence = votes[bestIndex] / classifier.trees.length;
  const margin = (votes[bestIndex] - secondBest) / classifier.trees.length;
  return { label: classifier.classes[bestIndex], confidence, margin };
}

function landmarkDistance(points, firstIndex, secondIndex) {
  const first = points[firstIndex];
  const second = points[secondIndex];
  return Math.hypot(first.x - second.x, first.y - second.y, first.z - second.z);
}

function fingerExtension(points, mcp, pip, tip) {
  const palm = palmScale(points);
  const proximal = landmarkDistance(points, mcp, pip);
  const fullFinger = landmarkDistance(points, mcp, tip);
  return fullFinger / Math.max(proximal, palm * 0.08);
}

function refineWithFingerArticulation(points, prediction) {
  const extensions = [
    fingerExtension(points, 5, 6, 8),
    fingerExtension(points, 9, 10, 12),
    fingerExtension(points, 13, 14, 16),
    fingerExtension(points, 17, 18, 20),
  ];
  const [indexExtension, middleExtension, ringExtension, pinkyExtension] = extensions;
  const palm = palmScale(points);
  const extended = extensions.map((value) => value > 1.48);
  const folded = extensions.map((value) => value < 1.48);
  const indexMiddleGap = landmarkDistance(points, 8, 12) / palmScale(points);
  const thumbIndexGap = landmarkDistance(points, 4, 8) / palmScale(points);
  const thumbReach = landmarkDistance(points, 2, 4) / palm;
  const thumbToFingerPips = [6, 10, 14, 18].map((index) => landmarkDistance(points, 4, index) / palm);
  const thumbAcrossPalm = (() => {
    const acrossX = points[17].x - points[5].x;
    const acrossY = points[17].y - points[5].y;
    const acrossZ = points[17].z - points[5].z;
    const thumbX = points[4].x - points[5].x;
    const thumbY = points[4].y - points[5].y;
    const thumbZ = points[4].z - points[5].z;
    const denominator = acrossX * acrossX + acrossY * acrossY + acrossZ * acrossZ;
    return (thumbX * acrossX + thumbY * acrossY + thumbZ * acrossZ) / (denominator || 1);
  })();
  const adjacentTipGaps = [[8, 12], [12, 16], [16, 20]]
    .map(([first, second]) => landmarkDistance(points, first, second) / palm);
  const tipsTogether = adjacentTipGaps.every((gap) => gap < 0.62);
  const indexMiddleCrossed = (points[6].x - points[10].x) * (points[8].x - points[12].x) < 0;
  const lowConfidence = prediction.confidence < CONFIDENCE_MIN || prediction.margin < CONFIDENCE_MARGIN_MIN;
  let candidate = null;
  let confusableLabels = [];

  // Finger-count and spacing shapes are reliable tie-breakers for these poses.
  if (extended[0] && extended[1] && folded[2] && folded[3]) {
    if (indexMiddleCrossed) {
      candidate = 'R';
      confusableLabels = ['U', 'V'];
    } else if (indexMiddleGap < 0.58) {
      candidate = 'U';
      confusableLabels = ['R', 'V', 'W'];
    } else {
      const thumbBetweenFingers = Math.min(
        landmarkDistance(points, 4, 6), landmarkDistance(points, 4, 10),
      ) / palm < 0.56;
      if (thumbBetweenFingers) {
        candidate = 'K';
        confusableLabels = ['U', 'V', 'R'];
      } else if (indexMiddleGap > 0.64) {
        candidate = 'V';
        confusableLabels = ['R', 'U', 'W'];
      }
    }
  } else if (extended[0] && extended[1] && extended[2] && folded[3]) {
    candidate = 'W';
    confusableLabels = ['B', 'U', 'V'];
  } else if (extended.every(Boolean) && tipsTogether) {
    candidate = 'B';
    confusableLabels = ['W', 'V'];
  } else if (folded[0] && folded[1] && folded[2] && extended[3]) {
    candidate = 'I';
    confusableLabels = ['A', 'S'];
  } else if (extended[0] && folded[1] && folded[2] && folded[3]) {
    const thumbTouchesOtherFingers = Math.min(...[12, 16, 20].map((index) => landmarkDistance(points, 4, index) / palm)) < 0.72;
    const indexVector = [points[8].x - points[5].x, points[8].y - points[5].y, points[8].z - points[5].z];
    const thumbVector = [points[4].x - points[2].x, points[4].y - points[2].y, points[4].z - points[2].z];
    const vectorCosine = (indexVector[0] * thumbVector[0] + indexVector[1] * thumbVector[1] + indexVector[2] * thumbVector[2])
      / (Math.hypot(...indexVector) * Math.hypot(...thumbVector) || 1);
    if (thumbTouchesOtherFingers) {
      candidate = 'D';
      confusableLabels = ['G', 'L'];
    } else if (thumbReach > 0.72 && Math.abs(vectorCosine) < 0.62) {
      candidate = 'L';
      confusableLabels = ['D', 'G'];
    } else if (thumbReach > 0.72 && Math.abs(vectorCosine) > 0.78) {
      const indexPointsDown = Math.abs(points[8].y - points[5].y) > Math.abs(points[8].x - points[5].x);
      candidate = indexPointsDown ? 'Q' : 'G';
      confusableLabels = ['D', 'L', 'Q', 'G'];
    }
  } else if (thumbIndexGap < 0.40 && extended[1] && extended[2] && extended[3]) {
    candidate = 'F';
    confusableLabels = ['C', 'O'];
  } else if (folded.every(Boolean) && thumbIndexGap > 0.42
    && Math.min(...thumbToFingerPips) < 0.56 && thumbAcrossPalm > 0.06 && thumbAcrossPalm < 0.92) {
    candidate = 'S';
    confusableLabels = ['A', 'E', 'O'];
  } else if (thumbIndexGap < 0.48 && indexExtension < 1.85
    && middleExtension < 1.95 && ringExtension < 1.75 && pinkyExtension < 1.8) {
    candidate = 'O';
    confusableLabels = ['A', 'C', 'E', 'F', 'S'];
  } else if (thumbIndexGap > 0.62 && thumbIndexGap < 1.25
    && extensions.every((value) => value < 1.75)) {
    candidate = 'C';
    confusableLabels = ['A', 'F', 'O'];
  }

  if (!candidate) return prediction;
  if (candidate === prediction.label) return { ...prediction, articulationMatch: true };
  if (!lowConfidence && !confusableLabels.includes(prediction.label)) return prediction;

  return {
    ...prediction,
    label: candidate,
    confidence: Math.max(prediction.confidence, CONFIDENCE_MIN + 0.02),
    margin: Math.max(prediction.margin, CONFIDENCE_MARGIN_MIN + 0.02),
    articulationMatch: true,
  };
}

function smoothStaticLandmarks(side, points) {
  const history = staticLandmarkHistory.get(side) || [];
  if (history.length) {
    const previous = history[history.length - 1];
    const meanMovement = points.reduce((sum, point, index) =>
      sum + Math.hypot(
        point.x - previous[index].x,
        point.y - previous[index].y,
        point.z - previous[index].z,
      ), 0) / points.length / palmScale(points);
    if (meanMovement > STATIC_SMOOTHING_RESET_DELTA) history.length = 0;
  }
  history.push(cloneLandmarks(points));
  if (history.length > STATIC_SMOOTHING_FRAMES) history.shift();
  staticLandmarkHistory.set(side, history);
  return points.map((_, pointIndex) => {
    const total = history.reduce((sum, frame) => ({
      x: sum.x + frame[pointIndex].x,
      y: sum.y + frame[pointIndex].y,
      z: sum.z + frame[pointIndex].z,
    }), { x: 0, y: 0, z: 0 });
    return {
      x: total.x / history.length,
      y: total.y / history.length,
      z: total.z / history.length,
    };
  });
}

function drawHand(points, side) {
  const width = overlay.width;
  const height = overlay.height;
  const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  ctx.lineWidth = Math.max(2, width / 340);
  ctx.lineCap = 'round';
  ctx.strokeStyle = side === 'Left' ? '#c1f17b' : '#77d8d4';
  for (const [a, b] of connections) {
    ctx.beginPath();
    ctx.moveTo(points[a].x * width, points[a].y * height);
    ctx.lineTo(points[b].x * width, points[b].y * height);
    ctx.stroke();
  }
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, Math.max(2.5, width / 210), 0, 2 * Math.PI);
    ctx.fillStyle = side === 'Left' ? '#f3ffdf' : '#e4ffff';
    ctx.fill();
  }
}

function addTextAtCursor(text) {
  const field = $('phrase');
  const start = field.selectionStart ?? field.value.length;
  const end = field.selectionEnd ?? field.value.length;
  field.setRangeText(text, start, end, 'end');
  field.dispatchEvent(new Event('input', { bubbles: true }));
  $('speakButton').disabled = field.value.trim().length === 0;
}

function confirmLetter(letter) {
  if (!letter) return;
  addTextAtCursor(letter);
  $('cameraHint').textContent = `Letra${letter.length > 1 ? 's' : ''} ${letter.split('').join(' + ')} adicionada${letter.length > 1 ? 's' : ''} ao texto`;
  notify(`${letter.split('').join(' + ')} adicionada${letter.length > 1 ? 's' : ''}`);
}

function resetPoseLock() {
  lastConfirmedLetters.clear();
  recentPredictions.clear();
  staticLandmarkHistory.clear();
}

function stabilizePrediction(side, prediction) {
  const history = recentPredictions.get(side) || [];
  history.push(prediction);
  if (history.length > STABLE_FRAMES) history.shift();
  recentPredictions.set(side, history);
  const counts = new Map();
  for (const item of history) {
    if (item.confidence >= CONFIDENCE_MIN && item.margin >= CONFIDENCE_MARGIN_MIN) {
      counts.set(item.label, (counts.get(item.label) || 0) + item.confidence);
    }
  }
  const best = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  if (!best) return { ...prediction, stable: false };
  const supportingFrames = history.filter((item) => item.label === best[0]
    && item.confidence >= CONFIDENCE_MIN && item.margin >= CONFIDENCE_MARGIN_MIN);
  const stable = supportingFrames.length >= STABLE_VOTES_MIN;
  const confidence = stable
    ? supportingFrames.reduce((sum, item) => sum + item.confidence, 0) / supportingFrames.length
    : prediction.confidence;
  const margin = stable
    ? supportingFrames.reduce((sum, item) => sum + item.margin, 0) / supportingFrames.length
    : prediction.margin;
  return {
    label: best[0], confidence, margin, stable,
    articulationMatch: supportingFrames.some((item) => item.articulationMatch),
  };
}

function handSide(result, index, points) {
  const detectedSide = result.handedness?.[index]?.[0]?.categoryName;
  if (detectedSide === 'Left' || detectedSide === 'Right') return detectedSide;
  return points[0].x < 0.5 ? 'Right' : 'Left';
}

function createDynamicOrigin(points) {
  return { x: points[0].x, y: points[0].y, z: points[0].z, scale: palmScale(points) };
}

function beginDynamicRecording() {
  if (!dynamicModeEnabled) {
    notify('Ative primeiro o modo Letras com movimento.');
    return;
  }
  if (!running) {
    notify('Ative a câmera antes de gravar exemplos.');
    return;
  }
  if (recordingSample) {
    recordingSample = null;
    $('recordDynamicButton').textContent = '● Gravar exemplo';
    updateDynamicStatus('Gravação cancelada.');
    return;
  }
  recordingSample = {
    label: $('dynamicLetter').value,
    side: $('dynamicSide').value,
    startedAt: null,
    origin: null,
    initialPoints: null,
    previousPoints: null,
    preRoll: [],
    motionCount: 0,
    lastMovementAt: 0,
    movementSeen: false,
    frames: [],
  };
  $('recordDynamicButton').textContent = '■ Cancelar gravação';
  updateDynamicStatus(`Preparado para gravar ${recordingSample.label} com a mão ${recordingSample.side === 'Right' ? 'direita' : 'esquerda'}. Faça o sinal completo dentro do quadro e pare a mão ao terminar.`);
}

async function finishDynamicRecording() {
  const capture = recordingSample;
  recordingSample = null;
  $('recordDynamicButton').textContent = '● Gravar exemplo';
  if (!capture || capture.frames.length < 10 || !capture.movementSeen) {
    updateDynamicStatus('Não houve quadros ou movimento suficientes. Tente novamente fazendo o sinal completo.');
    return;
  }
  const sample = {
    id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    label: capture.label,
    side: capture.side,
    frames: resampleSequence(capture.frames),
  };
  dynamicSamples.push(sample);
  try {
    await saveDynamicSample(sample);
    updateDynamicStatus(`Exemplo de ${sample.label} salvo (${capture.frames.length} quadros).`);
  } catch (error) {
    updateDynamicStatus(`Exemplo mantido nesta sessão; baixe o JSON antes de fechar. ${error.message}`);
  }
}

function movementScore(previous, current) {
  if (!previous) return 0;
  const scale = palmScale(previous);
  return Math.max(...DYNAMIC_MOVEMENT_POINTS.map((index) => {
    const a = previous[index], b = current[index];
    return Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z) / scale;
  }));
}

function finishDynamicGesture(side, track, now) {
  track.active = false;
  track.cooldownUntil = now + 850;
  track.preRoll = [];
  track.motionFrames = [];
  if (track.frames.length < 8) {
    track.frames = [];
    return false;
  }
  const sequence = resampleSequence(track.frames);
  track.frames = [];
  const prediction = classifyDynamicSequence(sequence, side);
  if (prediction && !prediction.rejected) {
    dynamicLastPrediction = prediction;
    confirmLetter(prediction.label);
    resetPoseLock();
    updateDynamicStatus(`Movimento reconhecido como ${prediction.label} (distância ${prediction.distance.toFixed(2)}).`);
  } else {
    dynamicLastPrediction = null;
    const reason = prediction?.ambiguous ? 'dois exemplos ficaram parecidos' : prediction ? `a distância ficou alta (${prediction.distance.toFixed(2)})` : 'faltam exemplos para essa mão';
    updateDynamicStatus(`Não confirmei a letra: ${reason}. Repita o sinal ou grave mais exemplos.`);
  }
  dynamicFinishedThisFrame = true;
  return true;
}

function processDynamicHands(hands, now) {
  dynamicFinishedThisFrame = false;
  dynamicLastPrediction = null;
  if (!dynamicModeEnabled) return false;
  if (recordingSample) {
    const sampleHand = hands.find((hand) => hand.side === recordingSample.side);
    if (!sampleHand) {
      $('dynamicStatus').textContent = `Mostre a mão ${recordingSample.side === 'Right' ? 'direita' : 'esquerda'} inteira para gravar ${recordingSample.label}.`;
      return true;
    }
    if (!recordingSample.previousPoints) {
      recordingSample.previousPoints = cloneLandmarks(sampleHand.points);
      recordingSample.preRoll.push(cloneLandmarks(sampleHand.points));
      $('dynamicStatus').textContent = `Pronto para ${recordingSample.label}. Faça o movimento completo sem tirar a mão do quadro.`;
      return true;
    }
    const score = movementScore(recordingSample.previousPoints, sampleHand.points);
    recordingSample.previousPoints = cloneLandmarks(sampleHand.points);
    if (recordingSample.startedAt === null) {
      recordingSample.preRoll.push(cloneLandmarks(sampleHand.points));
      if (recordingSample.preRoll.length > 5) recordingSample.preRoll.shift();
      recordingSample.motionCount = score > DYNAMIC_MOTION_THRESHOLD ? recordingSample.motionCount + 1 : 0;
      if (recordingSample.motionCount >= DYNAMIC_START_MOTION_FRAMES) {
        recordingSample.initialPoints = recordingSample.preRoll[0];
        recordingSample.origin = createDynamicOrigin(recordingSample.initialPoints);
        recordingSample.frames = recordingSample.preRoll.map((frame) => makeDynamicFrame(frame, recordingSample.origin));
        recordingSample.startedAt = now - (recordingSample.preRoll.length - 1) * CAMERA_POLL_MS;
        recordingSample.lastMovementAt = now;
        recordingSample.movementSeen = true;
        $('dynamicStatus').textContent = `Gravando o movimento de ${recordingSample.label}…`;
      }
      return true;
    }
    recordingSample.frames.push(makeDynamicFrame(sampleHand.points, recordingSample.origin));
    if (score > DYNAMIC_MOTION_THRESHOLD) recordingSample.lastMovementAt = now;
    const elapsed = now - recordingSample.startedAt;
    $('dynamicStatus').textContent = `Gravando ${recordingSample.label}… ${recordingSample.frames.length} quadros`;
    if ((elapsed > 500 && now - recordingSample.lastMovementAt > DYNAMIC_END_PAUSE_MS) || elapsed >= DYNAMIC_SAMPLE_MAX_MS) finishDynamicRecording();
    return true;
  }

  const hasReadyHand = hands.some((hand) => dynamicReady(hand.side));
  const hasActiveReadyTrack = [...dynamicTracks].some(([side, track]) => track.active && dynamicReady(side));
  if (!hasReadyHand && !hasActiveReadyTrack) return false;

  const seenSides = new Set();
  let anyMotion = false;
  let completedGesture = false;
  for (const hand of hands) {
    const { side, points } = hand;
    if (!dynamicReady(side)) continue;
    seenSides.add(side);
    let track = dynamicTracks.get(side);
    if (!track) {
      track = { lastPoints: null, lastSeen: now, preRoll: [], motionFrames: [], active: false, motionCount: 0, cooldownUntil: 0, frames: [] };
      dynamicTracks.set(side, track);
    }
    const previous = track.lastPoints;
    const score = movementScore(previous, points);
    track.lastPoints = cloneLandmarks(points);
    track.lastSeen = now;
    if (score > DYNAMIC_MOTION_THRESHOLD) anyMotion = true;

    if (track.active) {
      track.frames.push(makeDynamicFrame(points, track.origin));
      if (score > DYNAMIC_MOTION_THRESHOLD) track.lastMovementAt = now;
      if (track.frames.length % 5 === 0) $('dynamicStatus').textContent = `Capturando trajetória da mão ${side === 'Right' ? 'direita' : 'esquerda'}…`;
      if (now - track.lastMovementAt >= DYNAMIC_END_PAUSE_MS || now - track.startedAt >= DYNAMIC_MAX_GESTURE_MS) {
        completedGesture = finishDynamicGesture(side, track, now) || completedGesture;
      }
      continue;
    }

    track.preRoll.push(cloneLandmarks(points));
    if (track.preRoll.length > 5) track.preRoll.shift();
    if (now < track.cooldownUntil) continue;
    track.motionCount = score > DYNAMIC_MOTION_THRESHOLD ? track.motionCount + 1 : 0;
    if (track.motionCount >= DYNAMIC_START_MOTION_FRAMES) {
      const startingPoints = track.preRoll;
      track.origin = createDynamicOrigin(startingPoints[0]);
      track.frames = startingPoints.map((frame) => makeDynamicFrame(frame, track.origin));
      track.startedAt = now - (startingPoints.length - 1) * CAMERA_POLL_MS;
      track.lastMovementAt = now;
      track.active = true;
      track.motionCount = 0;
      $('dynamicStatus').textContent = `Movimento iniciado com a mão ${side === 'Right' ? 'direita' : 'esquerda'}…`;
    }
  }

  for (const [side, track] of dynamicTracks) {
    if (track.active && !seenSides.has(side) && now - track.lastSeen >= DYNAMIC_END_PAUSE_MS) {
      completedGesture = finishDynamicGesture(side, track, now) || completedGesture;
    }
  }
  return anyMotion || completedGesture || [...dynamicTracks.values()].some((track) => track.active);
}

function handFrameIssue(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.min(...xs), right = Math.max(...xs);
  const top = Math.min(...ys), bottom = Math.max(...ys);
  if (left < 0.018 || right > 0.982 || top < 0.018 || bottom > 0.982) return 'Mantenha a mão inteira dentro do quadro';
  if (right - left < 0.075 || bottom - top < 0.12) return 'Aproxime a mão até os dedos ficarem nítidos';
  return '';
}

function showPredictions(predictions) {
  if (!predictions.length) {
    $('translation').textContent = '—';
    $('confidence').textContent = classifier ? 'Mostre uma ou duas letras do alfabeto' : 'Carregando modelo';
    $('confidenceBar').style.width = '0%';
    return;
  }
  const ordered = [...predictions].sort((a, b) => (a.side === 'Right' ? 0 : 1) - (b.side === 'Right' ? 0 : 1));
  $('translation').textContent = ordered.map((item) => item.confidence >= CONFIDENCE_MIN && item.margin >= CONFIDENCE_MARGIN_MIN ? item.label : '?').join(' + ');
  const details = ordered.map((item) => {
    const side = item.side === 'Left' ? 'esquerda' : 'direita';
    if (item.confidence < CONFIDENCE_MIN || item.margin < CONFIDENCE_MARGIN_MIN) return `Mão ${side}: sinal incerto`;
    const articulation = item.articulationMatch ? ' · dedos compatíveis' : '';
    return `Mão ${side}: ${Math.round(item.confidence * 100)}%${item.stable ? '' : '…'}${articulation}`;
  });
  $('confidence').textContent = details.join(' · ');
  const average = ordered.reduce((sum, item) => sum + item.confidence, 0) / ordered.length;
  $('confidenceBar').style.width = `${Math.round(average * 100)}%`;
}

function processHands(result, now) {
  const hands = (result.landmarks || []).map((points, index) => {
    const side = handSide(result, index, points);
    const stablePoints = smoothStaticLandmarks(side, points);
    return {
      side,
      points,
      prediction: refineWithFingerArticulation(stablePoints, classify(normalizeLandmarks(stablePoints))),
    };
  });
  const activeSides = new Set(hands.map((hand) => hand.side));
  const current = hands.map((hand) => ({
    side: hand.side,
    ...stabilizePrediction(hand.side, hand.prediction),
  }));
  const dynamicHandling = processDynamicHands(hands, now);

  if (dynamicHandling) {
    if (dynamicFinishedThisFrame && dynamicLastPrediction) {
      $('translation').textContent = dynamicLastPrediction.label;
      $('confidence').textContent = `Movimento reconhecido · distância ${dynamicLastPrediction.distance.toFixed(2)}`;
      $('confidenceBar').style.width = `${Math.max(10, Math.round((1 - dynamicLastPrediction.distance / DYNAMIC_DISTANCE_LIMIT) * 100))}%`;
    } else if (!recordingSample) {
      $('translation').textContent = '↝';
      $('confidence').textContent = 'Analisando a trajetória da mão…';
      $('confidenceBar').style.width = '0%';
      $('cameraHint').textContent = 'Movimento detectado · conclua o sinal e pare a mão';
    } else {
      $('translation').textContent = '●';
      $('confidence').textContent = 'Gravando exemplo de movimento';
    }
    return;
  }

  if (dynamicModeEnabled) {
    const readyHands = hands.filter((hand) => dynamicReady(hand.side));
    const selectedHandReady = dynamicReady($('dynamicSide').value);
    $('translation').textContent = '↝';
    $('confidence').textContent = (readyHands.length || selectedHandReady)
      ? 'Modo de movimento ativo · faça o gesto completo e pare a mão'
      : `Grave ${DYNAMIC_MIN_SAMPLES_PER_CLASS} exemplos de uma letra para a mão ${$('dynamicSide').value === 'Right' ? 'direita' : 'esquerda'}`;
    $('confidenceBar').style.width = '0%';
    $('cameraHint').textContent = (readyHands.length || selectedHandReady)
      ? hands.length ? 'Modo movimento ativo · letras estáticas pausadas' : 'Mostre a mão para iniciar o sinal'
      : 'Modo movimento ativo · grave exemplos no painel';
    return;
  }

  for (const side of ['Left', 'Right']) {
    if (activeSides.has(side)) {
      missingHandSince.delete(side);
      continue;
    }
    if (!missingHandSince.has(side)) missingHandSince.set(side, now);
    if (now - missingHandSince.get(side) > RELEASE_MS) {
      recentPredictions.delete(side);
      lastConfirmedLetters.delete(side);
      staticLandmarkHistory.delete(side);
    }
  }

  if (!hands.length) {
    if (!handMissingSince) handMissingSince = now;
    if (now - handMissingSince > WORD_BREAK_MS && $('phrase').value && !$('phrase').value.endsWith(' ')) {
      addTextAtCursor(' ');
      $('cameraHint').textContent = 'Espaço inserido · comece outra palavra';
    }
    if (now - handMissingSince > 350) showPredictions([]);
    if (now - handMissingSince > 1200) $('cameraHint').textContent = 'Mostre uma ou duas mãos dentro das molduras';
    return;
  }

  handMissingSince = 0;
  lastHandSeenAt = now;
  const ordered = [...current].sort((a, b) => (a.side === 'Right' ? 0 : 1) - (b.side === 'Right' ? 0 : 1));
  showPredictions(ordered);

  const frameIssue = hands.map((hand) => handFrameIssue(hand.points)).find(Boolean);
  if (frameIssue) $('cameraHint').textContent = frameIssue;
  else if (ordered.some((item) => item.confidence < CONFIDENCE_MIN || item.margin < CONFIDENCE_MARGIN_MIN)) $('cameraHint').textContent = 'Sinal pouco nítido · ajuste a iluminação e abra bem os dedos';
  else if (ordered.some((item) => !item.stable)) $('cameraHint').textContent = 'Pose detectada · mantenha a mão parada por mais um instante';
  else $('cameraHint').textContent = `${ordered.length === 2 ? 'Duas mãos detectadas' : 'Mão detectada'} · pose estável`;

  // Each detected hand uses the single-hand alphabet model independently.
  // When two poses are stable together, append the two letters in viewer order.
  if (ordered.length === 2) {
    if (ordered.every((item) => item.stable)) {
      const changed = ordered.filter((item) => lastConfirmedLetters.get(item.side) !== item.label);
      if (changed.length) {
        confirmLetter(changed.map((item) => item.label).join(''));
        for (const item of ordered) lastConfirmedLetters.set(item.side, item.label);
      }
    }
  } else {
    const item = ordered[0];
    if (item.stable && lastConfirmedLetters.get(item.side) !== item.label) {
      confirmLetter(item.label);
      lastConfirmedLetters.set(item.side, item.label);
    }
  }
}

async function startCamera() {
  if (running) return stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) {
    notify('Câmera indisponível. Abra o app em localhost ou HTTPS.');
    $('cameraHint').textContent = 'Use localhost ou HTTPS para acessar a câmera';
    return;
  }
  $('cameraButton').disabled = true;
  $('cameraHint').textContent = 'Solicitando acesso à câmera…';
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 20 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    $('cameraPlaceholder').hidden = true;
    $('cameraStatus').innerHTML = '<i></i> CÂMERA ATIVA';
    $('cameraStatus').classList.add('active');
    $('cameraButton').innerHTML = '<span class="button-icon">■</span> Desativar câmera';
    $('cameraHint').textContent = 'Câmera ativa · preparando o reconhecimento…';
    $('cameraGuide').classList.add('visible');
    resetPoseLock();
    running = true;
    requestAnimationFrame(renderFrame);

    // Open the camera first so permission and the live image are not held up by downloads.
    await Promise.all([loadClassifier(), loadHandDetector()]);
    if (!running) return;
    $('recordDynamicButton').disabled = !dynamicModeEnabled;
    $('cameraHint').textContent = 'Mostre uma ou duas letras para soletrar';
    $('confidence').textContent = 'A câmera acompanha até duas mãos';
  } catch (error) {
    console.error(error);
    if (stream && running) {
      $('recordDynamicButton').disabled = true;
      $('modelStatus').textContent = 'ERRO NA IA';
      $('modelStatus').classList.add('model-error');
      $('cameraHint').textContent = `Câmera ativa; falha ao iniciar IA: ${error.message}`;
      $('confidence').textContent = 'O vídeo está ativo, mas o detector não iniciou';
      notify(`A câmera abriu, mas a IA falhou: ${error.message}`);
    } else {
      const messages = {
        NotAllowedError: 'Acesso à câmera negado. Libere a câmera para este endereço nas permissões do navegador.',
        NotFoundError: 'Nenhuma câmera foi encontrada neste dispositivo.',
        NotReadableError: 'A câmera está ocupada por outro aplicativo. Feche-o e tente de novo.',
        OverconstrainedError: 'A câmera não aceitou a resolução solicitada. Tente novamente.',
      };
      $('cameraHint').textContent = messages[error.name] || `Não foi possível acessar a câmera: ${error.message}`;
      notify(messages[error.name] || `Não foi possível iniciar: ${error.message}`);
    }
  } finally {
    $('cameraButton').disabled = false;
  }
}

function stopCamera() {
  running = false;
  recordingSample = null;
  dynamicTracks.clear();
  resetPoseLock();
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $('cameraPlaceholder').hidden = false;
  $('cameraStatus').innerHTML = '<i></i> CÂMERA DESLIGADA';
  $('cameraStatus').classList.remove('active');
  $('cameraButton').innerHTML = '<span class="button-icon">◉</span> Ativar câmera';
  $('recordDynamicButton').disabled = true;
  $('recordDynamicButton').textContent = '● Gravar exemplo';
  $('cameraHint').textContent = 'Aguardando câmera';
  $('cameraGuide').classList.remove('visible');
  $('translation').textContent = '—';
  $('confidence').textContent = 'Ative a câmera para começar';
  $('confidenceBar').style.width = '0%';
}

function renderFrame(timestamp) {
  if (!running) return;
  if (landmarker && classifier && video.readyState >= 2 && timestamp >= nextCameraPoll) {
    nextCameraPoll = timestamp + CAMERA_POLL_MS;
    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }
    if (video.currentTime !== lastFrameAt) {
      lastFrameAt = video.currentTime;
      const result = landmarker.detectForVideo(video, timestamp);
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      (result.landmarks || []).forEach((points, index) => drawHand(points, handSide(result, index, points)));
      processHands(result, timestamp);
    }
  }
  requestAnimationFrame(renderFrame);
}

function addSpace() { addTextAtCursor(' '); resetPoseLock(); }
function deleteLastCharacter() {
  const field = $('phrase');
  if (field.selectionStart !== field.selectionEnd) field.setRangeText('', field.selectionStart, field.selectionEnd, 'end');
  else if (field.selectionStart > 0) field.setRangeText('', field.selectionStart - 1, field.selectionStart, 'end');
  field.dispatchEvent(new Event('input', { bubbles: true }));
  $('speakButton').disabled = field.value.trim().length === 0;
  resetPoseLock();
}

function exportDynamicSamples() {
  const dataset = {
    format: 'libras-dynamic-dtw-v1',
    frameCount: DYNAMIC_FRAME_COUNT,
    featureCount: 81,
    samples: dynamicSamples,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(dataset)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'libras-dynamic-samples.json';
  link.click();
  URL.revokeObjectURL(url);
  updateDynamicStatus(`${dynamicSamples.length} exemplos exportados para JSON.`);
}

async function importDynamicSamples(file) {
  try {
    const dataset = JSON.parse(await file.text());
    if (dataset.format !== 'libras-dynamic-dtw-v1' || !Array.isArray(dataset.samples)) {
      throw new Error('O arquivo não é um conjunto de exemplos dinâmicos deste projeto.');
    }
    const imported = dataset.samples.filter(isValidDynamicSample).map((sample) => ({
      ...sample,
      id: sample.id || globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    }));
    if (!imported.length) throw new Error('Nenhum exemplo Ç, H, J, K, X, Y ou Z válido foi encontrado.');
    const existingIds = new Set(dynamicSamples.map((sample) => sample.id));
    const newSamples = imported.filter((sample) => !existingIds.has(sample.id));
    for (const sample of newSamples) {
      dynamicSamples.push(sample);
      await saveDynamicSample(sample);
    }
    updateDynamicStatus(`${newSamples.length} exemplos importados. Os exemplos são guardados neste navegador.`);
  } catch (error) {
    updateDynamicStatus(`Falha ao importar: ${error.message}`);
  } finally {
    $('importDynamicInput').value = '';
  }
}

$('cameraButton').addEventListener('click', startCamera);
$('dynamicModeToggle').addEventListener('click', toggleDynamicMode);
$('recordDynamicButton').addEventListener('click', beginDynamicRecording);
$('exportDynamicButton').addEventListener('click', exportDynamicSamples);
$('importDynamicInput').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) importDynamicSamples(file);
});
$('dynamicSide').addEventListener('change', () => updateDynamicStatus());
$('spaceButton').addEventListener('click', addSpace);
$('deleteButton').addEventListener('click', deleteLastCharacter);
$('clearPhrase').addEventListener('click', () => {
  $('phrase').value = '';
  $('speakButton').disabled = true;
  resetPoseLock();
});
$('phrase').addEventListener('input', () => { $('speakButton').disabled = $('phrase').value.trim().length === 0; });
$('speakButton').addEventListener('click', () => {
  const text = $('phrase').value.trim();
  if (!text || !('speechSynthesis' in window)) { notify('Áudio não disponível neste navegador.'); return; }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'pt-BR';
  window.speechSynthesis.speak(utterance);
});

const guideDialog = $('guideDialog');
document.querySelectorAll('.letter-example').forEach((button) => {
  button.addEventListener('click', () => {
    const letter = button.dataset.letter;
    const image = $('guideDialogImage');
    image.src = button.dataset.image;
    image.alt = `Referência visual para a letra ${letter}`;
    $('guideDialogLetter').textContent = letter;
    $('guideDialogStatus').textContent = button.dataset.supported === 'true' ? 'Esta letra está incluída no reconhecimento automático.' : 'Imagem para estudo. Esta letra ainda não é reconhecida automaticamente.';
    guideDialog.showModal();
  });
});
$('closeGuide').addEventListener('click', () => guideDialog.close());
guideDialog.addEventListener('click', (event) => {
  if (event.target === guideDialog) guideDialog.close();
});

loadClassifier().catch((error) => {
  $('modelStatus').textContent = 'ERRO AO CARREGAR';
  $('modelStatus').classList.add('model-error');
  $('cameraHint').textContent = 'Modelo não encontrado';
  console.error(error);
  notify(`Não foi possível carregar o modelo: ${error.message}`);
});
loadDynamicSamples();
