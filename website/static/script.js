// ===== RETAIN// Churn Risk Console — Flask-connected build =====
// Mengambil prediksi sungguhan dari endpoint Flask /predict yang
// menjalankan model Logistic Regression + scaler hasil training kamu.

const form = document.getElementById('churnForm');
const tenureInput = document.getElementById('tenure');
const monthlyInput = document.getElementById('MonthlyCharges');
const tenureVal = document.getElementById('tenureVal');
const monthlyVal = document.getElementById('monthlyVal');
const totalChargesInput = document.getElementById('TotalCharges');

const gaugeFill = document.getElementById('gaugeFill');
const gaugeNeedle = document.getElementById('gaugeNeedle');
const scoreNumber = document.getElementById('scoreNumber');
const verdictText = document.getElementById('verdictText');
const resultBadge = document.getElementById('resultBadge');
const signalList = document.getElementById('signalList');
const footTime = document.getElementById('footTime');
const clockEl = document.getElementById('clock');
const statusDot = document.getElementById('statusDot');
const modelWarning = document.getElementById('modelWarning');
const runBtn = document.querySelector('.run-btn');

const GAUGE_ARC_LENGTH = 315; // approx path length of the half-donut

// ---------- live label updates ----------
tenureInput.addEventListener('input', () => { tenureVal.textContent = tenureInput.value; });
monthlyInput.addEventListener('input', () => { monthlyVal.textContent = Number(monthlyInput.value).toFixed(0); });

// ---------- sample data loader ----------
document.getElementById('loadSample').addEventListener('click', () => {
  document.getElementById('gender').value = 'Female';
  document.getElementById('SeniorCitizen').checked = false;
  document.getElementById('Partner').checked = false;
  document.getElementById('Dependents').checked = false;
  tenureInput.value = 2;
  tenureVal.textContent = 2;
  document.getElementById('PhoneService').value = 'Yes';
  document.getElementById('MultipleLines').value = 'No';
  document.getElementById('InternetService').value = 'Fiber optic';
  document.getElementById('OnlineSecurity').value = 'No';
  document.getElementById('OnlineBackup').value = 'No';
  document.getElementById('DeviceProtection').value = 'No';
  document.getElementById('TechSupport').value = 'No';
  document.getElementById('StreamingTV').value = 'Yes';
  document.getElementById('StreamingMovies').value = 'Yes';
  document.getElementById('Contract').value = 'Month-to-month';
  document.getElementById('PaperlessBilling').value = 'Yes';
  document.getElementById('PaymentMethod').value = 'Electronic check';
  monthlyInput.value = 95;
  monthlyVal.textContent = 95;
  totalChargesInput.value = 190;
});

// ---------- clock ----------
function setClock() {
  const now = new Date();
  if (footTime) footTime.textContent = 'LAST SYNC ' + now.toLocaleTimeString('en-GB', { hour12: false });
}
setClock();
setInterval(setClock, 1000);

// ---------- health check on load ----------
async function checkHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.model_loaded && data.scaler_loaded) {
      clockEl.textContent = 'MODEL ARMED';
      statusDot.style.background = 'var(--signal-safe)';
      modelWarning.hidden = true;
      runBtn.disabled = false;
    } else {
      clockEl.textContent = 'MODEL OFFLINE';
      statusDot.style.background = 'var(--signal-risk)';
      modelWarning.hidden = false;
      runBtn.disabled = true;
    }
  } catch (err) {
    clockEl.textContent = 'SERVER UNREACHABLE';
    statusDot.style.background = 'var(--signal-risk)';
    modelWarning.hidden = false;
  }
}
checkHealth();

// ---------- payment method mapping ----------
function mapPaymentMethod(method) {
  const mapping = {
    'Bank transfer (automatic)': 'BT',
    'Credit card (automatic)': 'CC',
    'Electronic check': 'EC',
    'Mailed check': 'MC'
  };
  return mapping[method] || method;
}

// ---------- form -> payload mentah sesuai nama kolom dataset ----------
function readForm() {
  const fd = new FormData(form);
  return {
    gender: fd.get('gender'),
    SeniorCitizen: document.getElementById('SeniorCitizen').checked ? 'Yes' : 'No',
    Partner: document.getElementById('Partner').checked ? 'Yes' : 'No',
    Dependents: document.getElementById('Dependents').checked ? 'Yes' : 'No',
    tenure: Number(tenureInput.value),
    PhoneService: fd.get('PhoneService'),
    PaperlessBilling: fd.get('PaperlessBilling'),
    MultipleLines: fd.get('MultipleLines'),
    InternetService: fd.get('InternetService'),
    OnlineSecurity: fd.get('OnlineSecurity'),
    OnlineBackup: fd.get('OnlineBackup'),
    DeviceProtection: fd.get('DeviceProtection'),
    TechSupport: fd.get('TechSupport'),
    StreamingTV: fd.get('StreamingTV'),
    StreamingMovies: fd.get('StreamingMovies'),
    Contract: fd.get('Contract'),
    PaymentMethod: mapPaymentMethod(fd.get('PaymentMethod')),
    MonthlyCharges: Number(monthlyInput.value),
    TotalCharges: totalChargesInput.value ? Number(totalChargesInput.value) : null
  };
}

// ---------- render hasil dari Flask ----------
function renderResult({ prediction, churn_probability, risk_percent }) {
  const score = risk_percent; // 0-100

  const fillLength = (score / 100) * GAUGE_ARC_LENGTH;
  gaugeFill.style.strokeDasharray = `${fillLength} ${GAUGE_ARC_LENGTH}`;

  const angle = -90 + (score / 100) * 180;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;

  scoreNumber.textContent = score;

  let tier, badgeClass, fillColor, verdict;
  if (prediction === 0) {
    tier = 'TIDAK CHURN'; badgeClass = 'is-low'; fillColor = 'var(--signal-safe)';
    verdict = `Model memprediksi pelanggan TIDAK akan churn (kelas 0), dengan probabilitas churn ${score}%. Risiko relatif rendah berdasarkan kombinasi fitur saat ini.`;
  } else {
    tier = 'CHURN'; badgeClass = 'is-high'; fillColor = 'var(--signal-risk)';
    verdict = `Model memprediksi pelanggan BERPOTENSI churn (kelas 1), dengan probabilitas churn ${score}%. Disarankan tindakan retensi proaktif.`;
  }

  // tier menengah secara visual jika probabilitas dekat 50% meski kelas sudah pasti
  if (score >= 40 && score <= 60) {
    badgeClass = 'is-mid'; fillColor = 'var(--signal-warn)';
  }

  gaugeFill.style.stroke = fillColor;
  resultBadge.textContent = `${tier} (${score}%)`;
  resultBadge.className = `badge ${badgeClass}`;
  verdictText.textContent = verdict;
}

function renderSignals(payload) {
  // Penjelasan kontribusi fitur versi sederhana: menampilkan nilai input
  // mentah yang paling sering relevan terhadap churn, BUKAN nilai koefisien
  // model. Untuk kontribusi koefisien sungguhan per-fitur, lihat catatan
  // di bagian bawah file ini (opsional, butuh endpoint tambahan).
  const items = [];

  if (payload.Contract === 'Month-to-month') {
    items.push({ label: 'Kontrak month-to-month', type: 'risk' });
  } else {
    items.push({ label: `Kontrak ${payload.Contract}`, type: 'safe' });
  }

  items.push({
    label: `Tenure ${payload.tenure} bulan`,
    type: payload.tenure <= 6 ? 'risk' : payload.tenure >= 36 ? 'safe' : 'warn'
  });

  items.push({
    label: `Internet: ${payload.InternetService}`,
    type: payload.InternetService === 'Fiber optic' ? 'risk' : 'safe'
  });

  items.push({
    label: `Pembayaran: ${payload.PaymentMethod}`,
    type: payload.PaymentMethod === 'Electronic check' ? 'risk' : 'safe'
  });

  items.push({
    label: `Tagihan bulanan $${payload.MonthlyCharges}`,
    type: payload.MonthlyCharges >= 90 ? 'risk' : payload.MonthlyCharges <= 40 ? 'safe' : 'warn'
  });

  signalList.innerHTML = '';
  items.forEach(s => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="sig-dot ${s.type}"></span><span>${s.label}</span>`;
    signalList.appendChild(li);
  });
}

// ---------- submit handler ----------
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = readForm();

  runBtn.disabled = true;
  const originalLabel = runBtn.querySelector('span').textContent;
  runBtn.querySelector('span').textContent = 'Memproses…';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      verdictText.textContent = `Error: ${data.error || 'Gagal memproses prediksi.'}`;
      resultBadge.textContent = 'ERROR';
      resultBadge.className = 'badge is-high';
      console.error(data);
      return;
    }

    renderResult(data);
    renderSignals(payload);

  } catch (err) {
    verdictText.textContent = 'Tidak dapat menghubungi server Flask. Pastikan app.py sedang berjalan.';
    resultBadge.textContent = 'OFFLINE';
    resultBadge.className = 'badge is-high';
    console.error(err);
  } finally {
    runBtn.disabled = false;
    runBtn.querySelector('span').textContent = originalLabel;
  }
});