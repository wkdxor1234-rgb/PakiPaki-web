// app.js — WAV + M4A 지원, 서버 /health로 허용 확장자/최대용량 동기화

const statusEl  = document.getElementById('status');
const resultEl  = document.getElementById('result');
const fileInput = document.getElementById('fileInput');
const btnUpload = document.getElementById('btnUpload');

let pickedFile = null;
let allowedExt = new Set(['wav','m4a']); // 기본값 (서버 /health에서 갱신)
let maxMB = 30;                           // 기본값 (서버 /health에서 갱신)

// 1) 페이지 로드 시 서버 설정 동기화
(async function init() {
  try {
    const res  = await fetch('/health');
    const text = await res.text();
    const json = JSON.parse(text);
    if (Array.isArray(json.allowed_ext)) {
      allowedExt = new Set(json.allowed_ext.map(x => String(x).toLowerCase()));
    }
    if (typeof json.max_mb === 'number') {
      maxMB = json.max_mb;
    }
    statusEl.textContent = `대기중 (허용: ${[...allowedExt].join(', ').toUpperCase()}, 최대 ${maxMB}MB)`;
  } catch {
    // /health 실패해도 기본값으로 진행
    statusEl.textContent = `대기중 (허용: WAV,M4A, 최대 ${maxMB}MB)`;
  }
})();

// 2) 파일 선택 핸들러 (확장자/용량 사전 체크)
fileInput.onchange = () => {
  const f = fileInput.files?.[0];
  if (!f) { btnUpload.disabled = true; pickedFile = null; return; }

  const name = f.name.toLowerCase();
  const ext  = name.includes('.') ? name.split('.').pop() : '';

  if (!allowedExt.has(ext)) {
    statusEl.textContent = `허용되지 않는 형식입니다. (${[...allowedExt].join(', ').toUpperCase()} 만 가능)`;
    btnUpload.disabled = true;
    pickedFile = null;
    return;
  }

  const sizeMB = f.size / (1024 * 1024);
  if (sizeMB > maxMB) {
    statusEl.textContent = `파일이 너무 큽니다. (현재 ${sizeMB.toFixed(1)}MB > 최대 ${maxMB}MB)`;
    btnUpload.disabled = true;
    pickedFile = null;
    return;
  }

  pickedFile = f;
  statusEl.textContent = `선택됨: ${f.name} (${Math.round(f.size/1024)} KB)`;
  btnUpload.disabled = false;
};

// 3) 업로드 버튼
btnUpload.onclick = async () => {
  if (!pickedFile) return;

  statusEl.textContent = '업로드 중...';
  resultEl.innerHTML = '';

  const form = new FormData();
  // 서버 계약: 필드명 'file'
  form.append('file', pickedFile, pickedFile.name);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: form });

    // 먼저 텍스트로 받고 → JSON 파싱 시도 (HTML 에러 페이지 대응)
    const text = await res.text();
    let json = null;
    try { json = JSON.parse(text); }
    catch {
      statusEl.textContent = `실패: ${res.status}`;
      resultEl.innerHTML = `<pre>${text}</pre>`;
      return;
    }

    if (!res.ok || json.ok === false) {
      statusEl.textContent = `실패: ${res.status} ${json?.error || ''}`;
      resultEl.innerHTML = json?.detail ? `<pre>${json.detail}</pre>` : '';
      return;
    }

    const diag = json.diagnosis || '-';
    const cls  = diag.includes('정상') ? 'normal' : 'suspect';
    const conf = json.confidence!=null
      ? `${(json.confidence*100).toFixed(1)}%`
      : (json.probability!=null ? `${(json.probability*100).toFixed(1)}%` : '-');

    statusEl.textContent = '성공';
    resultEl.innerHTML = `
      <p>진단: <span class="badge ${cls}">${diag}</span></p>
      <p>신뢰도: ${conf}</p>
      ${json.features ? `<details><summary>추출 특징</summary><pre>${JSON.stringify(json.features, null, 2)}</pre></details>` : ''}
    `;
  } catch (e) {
    statusEl.textContent = `에러: ${e.message}`;
  }
};

};


