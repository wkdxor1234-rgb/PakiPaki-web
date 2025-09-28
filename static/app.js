// 같은 오리진이므로 절대주소 필요 없음. 상대경로 '/predict'로 호출합니다.
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const fileInput = document.getElementById('fileInput');
const btnUpload = document.getElementById('btnUpload');

let wavFile = null;

fileInput.onchange = () => {
  const f = fileInput.files?.[0];
  if (!f) { btnUpload.disabled = true; return; }
  if (!f.name.toLowerCase().endsWith('.wav')) {
    statusEl.textContent = 'WAV 파일만 업로드 가능합니다.';
    btnUpload.disabled = true;
    return;
  }
  wavFile = f;
  statusEl.textContent = `선택됨: ${f.name} (${Math.round(f.size/1024)} KB)`;
  btnUpload.disabled = false;
};

btnUpload.onclick = async () => {
  if (!wavFile) return;
  statusEl.textContent = '업로드 중...';

  const form = new FormData();
  // 서버 계약: form-data 필드명은 반드시 'file'
  form.append('file', wavFile, wavFile.name);

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    const json = await res.json();

    if (!res.ok || json.ok === false) {
      statusEl.textContent = `실패: ${res.status} ${json?.error || ''}`;
      resultEl.innerHTML = json?.detail ? `<pre>${json.detail}</pre>` : '';
      return;
    }

    const diag = json.diagnosis || '-';
    const cls = (diag.includes('정상')) ? 'normal' : 'suspect';
    const conf = (json.confidence != null)
      ? `${(json.confidence*100).toFixed(1)}%`
      : (json.probability != null ? `${(json.probability*100).toFixed(1)}%` : '-');

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

