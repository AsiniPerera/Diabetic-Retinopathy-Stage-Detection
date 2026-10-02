// Runs the unchanged website inside a Streamlit component: requests to the
// website's /api endpoints are answered by app.py through Streamlit instead of
// viva_server.py, so script.js works without modification.
(() => {
  const API_ENDPOINTS = ['/api/predict', '/api/report'];
  const pending = new Map();
  const originalFetch = window.fetch.bind(window);

  const send = (type, data = {}) =>
    window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, '*');

  // Fill the browser window so the page scrolls and its sticky header behave
  // as they do on the standalone website.
  const fitViewport = () => {
    let height = 900;
    try { height = window.parent.innerHeight; } catch { /* cross-origin parent */ }
    send('streamlit:setFrameHeight', { height });
  };

  // <body> is the scroll container inside the frame (see app.py), so send
  // window scrolling there.
  window.scrollTo = (...args) => document.body.scrollTo(...args);

  const toBase64 = blob => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] || '');
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });

  window.addEventListener('message', event => {
    if (event.data?.type !== 'streamlit:render') return;
    const response = event.data.args?.response;
    const resolve = response && pending.get(response.id);
    if (!resolve) return;
    pending.delete(response.id);
    const bytes = Uint8Array.from(atob(response.body), character => character.charCodeAt(0));
    resolve(new Response(bytes, {
      status: response.status,
      headers: { 'Content-Type': response.content_type }
    }));
  });

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (!API_ENDPOINTS.includes(url.pathname)) return originalFetch(input, init);
    const headers = new Headers(init.headers);
    const data = await toBase64(await new Response(init.body).blob());
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    return new Promise(resolve => {
      pending.set(id, resolve);
      send('streamlit:setComponentValue', {
        dataType: 'json',
        value: {
          id,
          endpoint: url.pathname,
          filename: decodeURIComponent(headers.get('X-Filename') || 'upload.png'),
          data
        }
      });
    });
  };

  // The logo links to "/", which would load Streamlit inside this frame.
  document.addEventListener('click', event => {
    if (event.target.closest?.('a[href="/"]')) {
      event.preventDefault();
      location.reload();
    }
  }, true);

  send('streamlit:componentReady', { apiVersion: 1 });
  fitViewport();
  try { window.parent.addEventListener('resize', fitViewport); } catch { /* cross-origin parent */ }
})();
