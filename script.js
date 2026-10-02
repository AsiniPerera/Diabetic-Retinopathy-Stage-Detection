const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('imageFile');
const button = document.getElementById('analyzeButton');
const status = document.getElementById('status');
const result = document.getElementById('result');
const empty = document.getElementById('emptyState');
let revision = 0;
let currentResult = null;

async function readJsonResponse(response) {
  const body = await response.text();
  try {
    return JSON.parse(body);
  } catch {
    if (body.trimStart().startsWith('<')) {
      throw Error(`The app returned an HTML page instead of its JSON API (HTTP ${response.status}). This Space must run the Docker app, not a static website. Check the Space runtime and restart its build.`);
    }
    throw Error(`The app returned an invalid API response (HTTP ${response.status}). Check the Space logs and try again.`);
  }
}

const preprocessingRows = [
  ['01 · Checking your upload', 'Make sure the image opens, is large enough, and has the correct orientation and colors.'],
  ['02 · Finding the retina', 'Locate the retinal area and trim away excess dark border while keeping the visible eye region.'],
  ['03 · Fitting the model frame', 'Fit the image into the model’s square frame without stretching it; add space around it where needed.'],
  ['04 · Softening image noise', 'Compare a gently smoothed preview with the original and check that fine details remain visible.'],
  ['05 · Evening out the lighting', 'Preview a lighting adjustment and check that bright areas and edges still look natural.'],
  ['06 · Making details easier to see', 'Compare local contrast changes; these can also make image noise more noticeable.'],
  ['07 · Bringing out fine details', 'Compare subtle detail-enhancement options and watch for halos or artificial-looking edges.'],
  ['08 · Trying another color view', 'Compare a separate color-based enhancement with the other image views.'],
  ['09 · Checking for bright spots', 'Mark possible glare and compare the original with an optional small-spot repair.'],
  ['10 · Balancing uneven brightness', 'Compare another gentle lighting adjustment with the earlier lighting preview.'],
  ['11 · Highlighting vessel patterns', 'Create a separate vessel-pattern view for inspection; it is not used as the model’s color image.'],
  ['12 · Preparing the model image', 'Keep the selected image in color, trim the outside area, and check its size before model input.'],
  ['13 · Keeping the process consistent', 'Record the settings and software details so the same preparation can be repeated.'],
  ['14 · Checking image clarity', 'Review focus, brightness, glare, and how much of the retinal area is visible. Poor-quality images do not receive a grade.'],
  ['15 · Preparing values for the model', 'Convert image colors to the numeric range expected by the model. This changes values, not the displayed picture.']
];

const augmentationRows = [
  ['Rotation', 'Rotate images by up to ±15° to represent differences in camera orientation. Keep the range small to limit cropping of retinal tissue.'],
  ['Horizontal flipping', 'Flip images with a probability of 0.5 to add orientation variation. The DR severity label remains unchanged because the task does not predict eye laterality.'],
  ['Scaling and zooming', 'Apply a scale between 0.9 and 1.1 to represent differences in retinal size and framing. Check enlarged images for loss of peripheral tissue.'],
  ['Translation', 'Shift images horizontally or vertically by up to 5% to represent minor differences in image position.'],
  ['Brightness adjustment', 'Make images slightly lighter or darker to represent differences in exposure while keeping retinal details visible.'],
  ['Contrast adjustment', 'Apply a contrast factor between 0.9 and 1.1 to represent variation in image appearance without obscuring subtle features.']
];

function populateTechniqueTable(id, rows) {
  const body = document.getElementById(id);
  rows.forEach(([name, purpose]) => {
    const row = document.createElement('tr');
    const term = document.createElement('th');
    const detail = document.createElement('td');
    term.scope = 'row';
    term.textContent = name;
    detail.textContent = purpose;
    row.append(term, detail);
    body.append(row);
  });
}

populateTechniqueTable('preprocessingTechniques', preprocessingRows);
populateTechniqueTable('augmentationTechniques', augmentationRows);

function clearImageGalleries(message = 'Upload and analyze an image to see its image previews here.') {
  const preprocessing = document.getElementById('preprocessingGallery');
  preprocessing.replaceChildren();
  const preprocessingEmpty = document.createElement('li');
  preprocessingEmpty.className = 'gallery-empty';
  preprocessingEmpty.textContent = message;
  preprocessing.append(preprocessingEmpty);
  document.getElementById('preprocessingStop').hidden = true;

  const augmentation = document.getElementById('augmentationGallery');
  const augmentationEmpty = document.createElement('p');
  augmentationEmpty.className = 'gallery-empty';
  augmentationEmpty.textContent = message;
  augmentation.replaceChildren(augmentationEmpty);
}

function createImageCard(title, description, src, alt, badgeText = '') {
  const card = document.createElement('li');
  card.className = 'image-step-card';
  const image = document.createElement('img');
  image.src = src;
  image.alt = alt;
  image.loading = 'lazy';
  const heading = document.createElement('h2');
  heading.textContent = title;
  const detail = document.createElement('p');
  detail.textContent = description;
  card.append(image);
  if (badgeText) {
    const badge = document.createElement('span');
    badge.className = 'evidence-badge';
    badge.textContent = badgeText;
    card.append(badge);
  }
  card.append(heading, detail);
  return card;
}

function renderPreprocessingGallery(steps, classificationPermitted) {
  const gallery = document.getElementById('preprocessingGallery');
  gallery.replaceChildren();
  if (!Array.isArray(steps) || steps.length === 0) {
    clearImageGalleries();
    return;
  }

  const preferredImages = {
    1: 'Validated RGB',
    2: 'Cropped retina',
    3: '224 × 224 RGB',
    4: 'Bilateral',
    5: 'Corrected',
    6: 'CLAHE',
    7: 'Unsharp',
    8: 'LAB lightness',
    9: 'Optional inpaint',
    10: 'Division',
    11: 'Vessel response',
    12: 'Selected reconstructed RGB',
    13: 'Baseline model image',
    14: 'Glare candidates',
    15: 'RGB image before normalization'
  };
  const friendlyTitles = preprocessingRows.map(([title]) => title.replace(/^\d+ · /, ''));
  steps.forEach(step => {
    const images = step.images || [];
    const selected = images.find(image => image.label === preferredImages[step.number])
      || images[images.length - 1];
    const imageUrl = selected?.data_url || step.image;
    if (!imageUrl) return;
    const badgeText = step.number >= 4 && step.number <= 11
      ? 'PREVIEW ONLY · DOES NOT CHANGE RESULT'
      : 'IMAGE PREPARATION STEP';
    const friendlyDescription = preprocessingRows[step.number - 1]?.[1] || step.description || '';
    gallery.append(createImageCard(
      `${String(step.number).padStart(2, '0')} · ${friendlyTitles[step.number - 1] || 'Image step'}`,
      friendlyDescription,
      imageUrl,
      `Image showing step ${step.number}: ${friendlyTitles[step.number - 1] || 'image preparation'}`,
      badgeText
    ));
  });
  const stopped = document.getElementById('preprocessingStop');
  stopped.hidden = classificationPermitted && !currentResult.preprocessing_evidence_error;
  stopped.textContent = currentResult.preprocessing_evidence_error || (classificationPermitted
    ? ''
    : 'Preprocessing stopped after validation and quality checks; the image was not sent to the model.');
}

function renderAugmentationGallery(file, renderRevision) {
  const gallery = document.getElementById('augmentationGallery');
  if (!file) {
    clearImageGalleries();
    return;
  }

  const sourceUrl = URL.createObjectURL(file);
  const source = new Image();
  source.onload = () => {
    URL.revokeObjectURL(sourceUrl);
    if (renderRevision !== revision || file !== fileInput.files[0]) return;

    const canvasWidth = 280;
    const canvasHeight = 190;
    const fit = Math.min((canvasWidth - 18) / source.naturalWidth, (canvasHeight - 18) / source.naturalHeight);
    const imageWidth = source.naturalWidth * fit;
    const imageHeight = source.naturalHeight * fit;
    const transforms = [
      { title: 'Rotation', description: 'Illustration: rotate by +12°.', draw: context => {
        context.translate(canvasWidth / 2, canvasHeight / 2);
        context.rotate(12 * Math.PI / 180);
        context.drawImage(source, -imageWidth / 2, -imageHeight / 2, imageWidth, imageHeight);
      } },
      { title: 'Horizontal flipping', description: 'Illustration: flip horizontally.', draw: context => {
        context.translate(canvasWidth, 0);
        context.scale(-1, 1);
        context.drawImage(source, (canvasWidth - imageWidth) / 2, (canvasHeight - imageHeight) / 2, imageWidth, imageHeight);
      } },
      { title: 'Scaling and zooming', description: 'Illustration: zoom by 10%.', draw: context => {
        context.drawImage(source, (canvasWidth - imageWidth * 1.1) / 2, (canvasHeight - imageHeight * 1.1) / 2, imageWidth * 1.1, imageHeight * 1.1);
      } },
      { title: 'Translation', description: 'Illustration: shift by about 5% of the frame.', draw: context => {
        context.translate(canvasWidth * 0.05, -canvasHeight * 0.05);
        context.drawImage(source, (canvasWidth - imageWidth) / 2, (canvasHeight - imageHeight) / 2, imageWidth, imageHeight);
      } },
      { title: 'Brightness adjustment', description: 'Illustration: slightly increase brightness.', filter: 'brightness(1.1)' },
      { title: 'Contrast adjustment', description: 'Illustration: slightly increase contrast.', filter: 'contrast(1.1)' }
    ];

    const cards = document.createDocumentFragment();
    transforms.forEach(transform => {
      const canvas = document.createElement('canvas');
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Could not create an image preview canvas.');
      context.fillStyle = '#07090a';
      context.fillRect(0, 0, canvasWidth, canvasHeight);
      context.filter = transform.filter || 'none';
      if (transform.draw) {
        transform.draw(context);
      } else {
        context.drawImage(source, (canvasWidth - imageWidth) / 2, (canvasHeight - imageHeight) / 2, imageWidth, imageHeight);
      }
      cards.append(createImageCard(
        transform.title,
        transform.description,
        canvas.toDataURL('image/jpeg', 0.82),
        `${transform.title} training-time illustration of the uploaded image`
      ));
    });
    if (renderRevision === revision && file === fileInput.files[0]) {
      gallery.replaceChildren(cards);
    }
  };
  source.onerror = () => {
    URL.revokeObjectURL(sourceUrl);
    if (renderRevision !== revision || file !== fileInput.files[0]) return;
    const message = document.createElement('p');
    message.className = 'gallery-empty';
    message.textContent = 'Could not create the augmentation previews for this image.';
    gallery.replaceChildren(message);
  };
  source.src = sourceUrl;
}

function activateView(name) {
  const views = {
    analysis: document.getElementById('analysisView'),
    preprocessing: document.getElementById('preprocessingView'),
    augmentation: document.getElementById('augmentationView'),
    model: document.getElementById('modelView')
  };

  Object.entries(views).forEach(([key, view]) => {
    view.hidden = key !== name;
  });
  document.querySelectorAll('.demo-tab').forEach(tab => {
    const active = tab.dataset.view === name;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

const tabs = [...document.querySelectorAll('.demo-tab')];
tabs.forEach(tab => {
  tab.addEventListener('click', () => activateView(tab.dataset.view));
  tab.addEventListener('keydown', event => {
    const available = tabs.filter(item => !item.hidden);
    const currentIndex = available.indexOf(tab);
    let nextIndex;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      nextIndex = (currentIndex + 1) % available.length;
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      nextIndex = (currentIndex - 1 + available.length) % available.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = available.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    available[nextIndex].focus();
    activateView(available[nextIndex].dataset.view);
  });
});

document.querySelectorAll('.model-guide-button').forEach(button => {
  button.addEventListener('click', () => activateView(button.dataset.view));
});

fileInput.addEventListener('change', () => {
  revision++;
  const file = fileInput.files[0];
  document.getElementById('fileName').textContent = file?.name || 'Choose a retinal image';
  const dropzone = fileInput.closest('.dropzone');
  dropzone.classList.remove('has-preview');
  dropzone.querySelector('.drop-preview')?.remove();
  if (file && file.type.startsWith('image/')) {
    const preview = document.createElement('img');
    preview.className = 'drop-preview';
    preview.alt = 'Selected retinal image preview';
    preview.src = URL.createObjectURL(file);
    preview.onload = () => URL.revokeObjectURL(preview.src);
    dropzone.insertBefore(preview, document.getElementById('fileName'));
    dropzone.classList.add('has-preview');
  }
  result.hidden = true;
  document.getElementById('nonRetinalBlock').hidden = true;
  document.querySelector('.workspace').classList.remove('is-non-retinal');
  currentResult = null;
  document.getElementById('resultTools').hidden = true;
  document.getElementById('chatMessages').replaceChildren();
  document.getElementById('chatQuestion').value = '';
  clearImageGalleries('Analyze this image to view its preprocessing and augmentation examples.');
  empty.hidden = false;
  status.textContent = '';
  status.className = '';
});

function displayLogValue(value) {
  if (value === null) return 'not available';
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (Array.isArray(value)) return value.join(' × ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

const resultLevels = [
  {
    label: 'Clear / Healthy Retina',
    explanation: 'No signs of diabetic retinopathy were detected by this prototype. This does not rule out eye disease.',
    nextStep: 'Continue regular eye-care checkups.'
  },
  {
    label: 'Mild Risk',
    explanation: 'The model suggests minor early changes.',
    nextStep: 'Regular eye-care checkups are recommended.'
  },
  {
    label: 'Moderate Risk',
    explanation: 'The model suggests noticeable retinal changes.',
    nextStep: 'Consider discussing this result with an eye-care professional.'
  },
  {
    label: 'High Risk',
    explanation: 'The model suggests significant retinal changes.',
    nextStep: 'Please arrange prompt review with an eye-care professional.'
  },
  {
    label: 'Critical Risk',
    explanation: 'The model suggests advanced retinal changes.',
    nextStep: 'Please seek prompt review from an eye-care professional.'
  }
];

const resultLevelColors = ['#28A745', '#8BC34A', '#FFC107', '#FD7E14', '#DC3545'];

function getResultLevel(grade) {
  const gradeIndex = Number(grade);
  return Number.isInteger(gradeIndex) && gradeIndex >= 0 && gradeIndex < resultLevels.length
    ? { ...resultLevels[gradeIndex], number: gradeIndex + 1 }
    : { label: 'Unspecified level', explanation: 'Review the recorded model output with an eye-care professional.', nextStep: 'This prototype cannot provide medical advice.', number: null };
}

function renderSeverityScale(grade, permitted) {
  const scale = document.getElementById('severityScale');
  const stages = document.getElementById('severityStages');
  const gradeIndex = Number(grade);
  stages.replaceChildren();
  if (!permitted || !Number.isInteger(gradeIndex) || gradeIndex < 0 || gradeIndex >= resultLevels.length) {
    scale.hidden = true;
    return;
  }

  const progress = ((gradeIndex + 0.5) / resultLevels.length) * 100;
  stages.style.setProperty('--severity-progress', `${progress}%`);
  stages.style.setProperty('--severity-progress-color', resultLevelColors[gradeIndex]);
  stages.setAttribute('aria-label', `Prototype estimate level ${gradeIndex + 1} of 5: ${resultLevels[gradeIndex].label}`);
  resultLevels.forEach((level, index) => {
    const stage = document.createElement('li');
    const number = document.createElement('span');
    const name = document.createElement('span');
    stage.className = `severity-stage severity-${index} ${index < gradeIndex ? 'is-complete' : index > gradeIndex ? 'is-upcoming' : 'is-current'}`;
    number.className = 'severity-number';
    number.textContent = String(index + 1);
    name.className = 'severity-label';
    name.textContent = level.label;
    if (index === gradeIndex) {
      stage.setAttribute('aria-current', 'step');
    }
    stage.append(number, name);
    stages.append(stage);
  });
  scale.hidden = false;
}

function getSuggestions(data) {
  if (!data.classification_permitted) {
    return [
      'This upload was stopped before model inference, so this run has no grade to interpret.',
      'For a new demo run, try a supported JPEG or PNG with the retinal area in focus, fully visible, and evenly lit with minimal glare.',
      'You can ask the chatbot about the recorded quality notes or download this run’s report. This prototype cannot assess eye health.'
    ];
  }
  const level = getResultLevel(data.predicted_grade);
  const gradeSuggestion = `Prototype estimate: Level ${level.number} · ${level.label}. ${level.explanation} ${level.nextStep}`;
  if (data.status === 'review_required') {
    return [
      gradeSuggestion,
      'This result was routed to “review required” by the prototype. That is a demo workflow status, not a clinical review.',
      'Ask the chatbot about the recorded probability or uncertainty, or download the report to inspect this run’s output.'
    ];
  }
  return [
    gradeSuggestion,
    'The image passed this prototype’s routing checks; “accepted” does not mean medically confirmed.',
    'You can ask the chatbot about this run or download the report. Do not use this prototype alone to make health decisions.'
  ];
}

function renderResultTools(data) {
  const tools = document.getElementById('resultTools');
  const suggestions = document.getElementById('suggestionsList');
  const findings = document.getElementById('qualityFindings');
  const reasons = document.getElementById('qualityReasons');
  suggestions.replaceChildren();
  getSuggestions(data).forEach(text => {
    const item = document.createElement('li');
    item.textContent = text;
    suggestions.append(item);
  });

  const qualityReasons = data.quality?.reasons || [];
  reasons.replaceChildren();
  qualityReasons.forEach(text => {
    const item = document.createElement('li');
    item.textContent = text;
    reasons.append(item);
  });
  findings.hidden = qualityReasons.length === 0 && data.classification_permitted;
  tools.hidden = false;
}

function addChatMessage(role, text) {
  const messages = document.getElementById('chatMessages');
  const item = document.createElement('li');
  item.className = `chat-message ${role}`;
  item.textContent = text;
  messages.append(item);
  while (messages.children.length > 20) messages.firstElementChild.remove();
  messages.scrollTop = messages.scrollHeight;
}

function answerResultQuestion(question, data) {
  const query = question.toLocaleLowerCase();
  const rejected = !data.classification_permitted;
  const qualityReasons = data.quality?.reasons || data.reasons || [];
  const reasonsText = qualityReasons.length ? qualityReasons.join(' ') : 'No specific quality issue was recorded.';

  if (rejected) {
    if (/suggest|next|should|upload|do now/.test(query)) {
      return getSuggestions(data).join(' ');
    }
    if (/grade|predict|result|confidence|probab|severity/.test(query)) {
      return `No estimate is available: the image was rejected before model inference. ${reasonsText} You can try a clearer supported image or download the quality report.`;
    }
    if (/why|reject|flag|quality|blur|glare|exposure|retina|image/.test(query)) {
      return `The image was rejected before model inference. ${reasonsText} For another demo run, try a supported image with the retinal area in focus, fully visible, and evenly lit. These checks are heuristics and cannot verify retinal validity.`;
    }
    return 'No prediction is available for this image. I can explain its recorded quality notes or how to try another upload. I cannot assess eye health or give medical advice.';
  }

  if (/grade|predict|result|severity|mean/.test(query)) {
    const level = getResultLevel(data.predicted_grade);
    const review = data.status === 'review_required'
      ? ' This result is flagged for human review.'
      : ' It passed the prototype checks, which is not medical confirmation.';
    return `The model estimated Level ${level.number} · ${level.label}, with a model probability of ${(data.confidence * 100).toFixed(1)}%. ${level.explanation} ${level.nextStep} This is a prototype estimate for this image only, not a diagnosis or a measure of disease progression.${review}`;
  }
  if (/confidence|uncertain|probab|review|flag/.test(query)) {
    const uncertainty = Number.isFinite(data.uncertainty)
      ? ` Recorded dropout uncertainty: ${(data.uncertainty * 100).toFixed(1)}%.`
      : '';
    const review = data.status === 'review_required'
      ? ` Human review is required by the prototype routing. ${data.reasons?.join(' ') || ''}`
      : ' The prototype marked this result as accepted; that does not confirm a diagnosis.';
    return `Model probability: ${(data.confidence * 100).toFixed(1)}%.${uncertainty}${review}`;
  }
  if (/quality|blur|glare|exposure|retina|image/.test(query)) {
    const metrics = data.quality?.metrics || {};
    const summary = Object.entries(metrics)
      .slice(0, 4)
      .map(([name, value]) => `${name}: ${displayLogValue(value)}`)
      .join('; ');
    return `Image-quality status: ${data.quality?.quality_status || 'not recorded'}. ${summary || 'No quality measurements were returned.'} These measurements are heuristic and do not prove retinal validity.`;
  }
  if (/suggest|next|should|do now|advice/.test(query)) {
    return getSuggestions(data).join(' ');
  }
  if (/diagnos|treat|medicine|medication|cure|symptom/.test(query)) {
    return 'I cannot assess symptoms, diagnose a condition, or recommend treatment. This is a research-demo estimate only. For medical questions, consult a qualified eye-care professional.';
  }
  return 'I can explain this run’s displayed level, model probability, review status, or recorded image-quality notes. I cannot assess eye health or give medical advice.';
}

document.getElementById('chatForm').addEventListener('submit', event => {
  event.preventDefault();
  const input = document.getElementById('chatQuestion');
  const question = input.value.trim();
  if (!question || !currentResult) return;
  addChatMessage('user', question);
  addChatMessage('assistant', answerResultQuestion(question, currentResult));
  input.value = '';
  input.focus();
});

document.querySelectorAll('.prompt-chip').forEach(button => {
  button.addEventListener('click', () => {
    document.getElementById('chatQuestion').value = button.dataset.question;
    document.getElementById('chatForm').requestSubmit();
  });
});

document.getElementById('downloadReport').addEventListener('click', async event => {
  const file = fileInput.files[0];
  if (!currentResult || !file) return;
  const reportButton = event.currentTarget;
  const requestRevision = revision;
  reportButton.disabled = true;
  reportButton.textContent = 'Creating PDF...';
  try {
    const response = await fetch('/api/report', {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'X-Filename': encodeURIComponent(file.name)
      },
      body: file
    });
    if (!response.ok) {
      const error = await readJsonResponse(response);
      throw Error(error.error || 'The PDF could not be created.');
    }
    const pdf = await response.blob();
    if (requestRevision !== revision) return;
    const url = URL.createObjectURL(pdf);
    const link = document.createElement('a');
    const safeName = file.name.replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-') || 'retinal-screening';
    link.href = url;
    link.download = `${safeName}-report.pdf`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    status.className = 'error';
    status.textContent = error.message || 'The PDF could not be created.';
  } finally {
    reportButton.disabled = false;
    reportButton.textContent = 'Download PDF report';
  }
});

function showResult(data) {
  empty.hidden = true;
  const nonRetinalBlock = document.getElementById('nonRetinalBlock');
  nonRetinalBlock.hidden = data.status !== 'non_retinal';
  document.querySelector('.workspace').classList.toggle('is-non-retinal', data.status === 'non_retinal');
  if (data.status === 'non_retinal') {
    result.hidden = true;
    currentResult = data;
    return;
  }
  result.hidden = false;
  currentResult = data;
  renderResultTools(data);
  addChatMessage(
    'assistant',
    data.classification_permitted
      ? 'I can explain the displayed level, model probability, image-quality notes, and review status.'
      : 'This image did not receive an estimate. I can explain the quality notes and ways to submit a clearer photo.'
  );
  renderPreprocessingGallery(
    data.preprocessing_evidence || data.visual_steps,
    data.classification_permitted
  );
  renderAugmentationGallery(fileInput.files[0], revision);
  const permitted = data.classification_permitted;
  document.getElementById('grade').textContent = permitted
    ? `Level ${getResultLevel(data.predicted_grade).number} · ${getResultLevel(data.predicted_grade).label}`
    : 'Try a clearer image';
  const confidenceMeter = document.getElementById('confidence');
  const confidenceValue = document.getElementById('confidenceValue');
  const confidence = Number(data.confidence);
  const validConfidence = permitted && Number.isFinite(confidence);
  confidenceMeter.hidden = !validConfidence;
  if (validConfidence) {
    const boundedConfidence = Math.max(0, Math.min(1, confidence));
    const percentage = Math.round(boundedConfidence * 100);
    const level = getResultLevel(data.predicted_grade);
    confidenceValue.textContent = `${percentage}%`;
    confidenceMeter.style.setProperty('--confidence-progress', `${percentage}%`);
    confidenceMeter.style.setProperty(
      '--confidence-color',
      resultLevelColors[Number(data.predicted_grade)]
    );
    confidenceMeter.setAttribute(
      'aria-label',
      `${percentage}% model confidence for Level ${level.number}, ${level.label}. This is a model probability, not medical certainty.`
    );
  } else {
    confidenceValue.textContent = '';
    confidenceMeter.removeAttribute('aria-label');
  }
  renderSeverityScale(data.predicted_grade, permitted);
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  const requestRevision = ++revision;
  result.hidden = true;
  document.getElementById('nonRetinalBlock').hidden = true;
  document.querySelector('.workspace').classList.remove('is-non-retinal');
  currentResult = null;
  document.getElementById('resultTools').hidden = true;
  document.getElementById('chatMessages').replaceChildren();
  empty.hidden = true;
  status.className = 'loading';
  status.textContent = 'Analyzing with trained model…';
  button.disabled = true;
  fileInput.disabled = true;
  try {
    if (location.protocol === 'file:') {
      throw Error('Start the local app using start-viva.cmd, then open http://127.0.0.1:8765.');
    }
    if (file.size > 128 * 1024 * 1024) throw Error('Choose an image under 128 MB.');
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'X-Filename': encodeURIComponent(file.name)
      },
      body: file
    });
    const data = await readJsonResponse(response);
    if (!response.ok) throw Error(data.error || 'Analysis failed.');
    if (requestRevision !== revision) return;
    status.textContent = '';
    status.className = '';
    showResult(data);
  } catch (error) {
    if (requestRevision === revision) {
      status.className = 'error';
      status.textContent = error instanceof TypeError
        ? 'Cannot reach the model. Start the local app with start-viva.cmd and open http://127.0.0.1:8765.'
        : (error.message || 'The model could not analyze this image.');
    }
  } finally {
    button.disabled = false;
    fileInput.disabled = false;
  }
});
