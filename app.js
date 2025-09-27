// ★ API 주소를 실제 값으로 바꾸세요 (HTTPS 필수). 예: https://api.your-domain.com
const API = "https://CHANGE_ME.example.com";

const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const fileInput = document.getElementById('fileInput');
const btnUpload = document.getElementById('btnUpload');

let wavFile = null;

fileInput.onchange = () => {
  const f = fileInput.files?.[0];
  if (!f) { btnUpload.disabled = true; return; }
  const name = f.name.toLowerCase();
  if (!(name.endsWith('.wav') || f.type === 'audio/wav')) {
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
  // Flask 서버 스펙: form-data 필드명은 반드시 'file'
  form.append('file', wavFile, wavFile.name);

  try {
    const res = await fetch(`${API}/predict`, { method: 'POST', body: form });
    const text = await res.text();
    let json;
    try { json = JSON.parse(text); } catch { json = { raw: text }; }

    if (!res.ok) {
      statusEl.textContent = `실패: ${res.status} ${json?.error || ''}`;
      resultEl.innerHTML = json?.raw ? `<pre>${json.raw}</pre>` : '';
      return;
    }

    // 예상 응답: { diagnosis: "정상입니다" | "파킨슨병 의심", confidence: 0~1, ... }
    const diag = json.diagnosis || '-';
    const conf = (json.confidence != null)
      ? `${(json.confidence*100).toFixed(1)}%`
      : (json.probability != null ? `${(json.probability*100).toFixed(1)}%` : '-');

    statusEl.textContent = '성공';
    const cls = (typeof diag === 'string' && diag.includes('정상')) ? 'normal' : 'suspect';
    resultEl.innerHTML = `
      <p>진단: <span class="badge ${cls}">${diag}</span></p>
      <p>신뢰도: ${conf}</p>
      ${json.error ? `<pre>${json.error}</pre>` : ''}
      ${json.features ? `<details><summary>추출 특징</summary><pre>${JSON.stringify(json.features, null, 2)}</pre></details>` : ''}
    `;
  } catch (e) {
    statusEl.textContent = `에러: ${e.message}`;
  }
};
