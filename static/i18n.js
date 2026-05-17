/**
 * Lightweight i18n module.
 * Loads the active locale JSON synchronously so t() is available
 * before any other script runs. Locale is persisted in localStorage.
 */
(function () {
  const SUPPORTED = ['en', 'ca', 'es'];
  const stored    = localStorage.getItem('locale');
  const locale    = SUPPORTED.includes(stored) ? stored : 'en';

  let messages = {};
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', `/static/locales/${locale}.json`, false /* sync */);
    xhr.send(null);
    if (xhr.status === 200) messages = JSON.parse(xhr.responseText);
  } catch (e) {
    console.warn('[i18n] Failed to load locale:', locale, e);
  }

  // Set <html lang> immediately
  document.documentElement.lang = locale;

  /* ── Core API ──────────────────────────────────────────────────────────── */

  /**
   * Translate a dot-separated key, optionally interpolating {var} placeholders.
   * Falls back to the key string if not found.
   */
  function t(keyPath, vars) {
    const keys = keyPath.split('.');
    let val = messages;
    for (const k of keys) {
      if (val == null || typeof val !== 'object') return keyPath;
      val = val[k];
    }
    if (typeof val !== 'string') return keyPath;
    if (!vars) return val;
    return val.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : ''));
  }

  function setLocale(lang) {
    localStorage.setItem('locale', lang);
    location.reload();
  }

  function getCurrentLocale() {
    return locale;
  }

  /**
   * Walk the DOM and apply translations to elements that carry
   * data-i18n (textContent), data-i18n-placeholder, or data-i18n-title
   * (on <html> to set document.title).
   */
  function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
      el.textContent = t(el.getAttribute('data-i18n'));
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });
    const titleKey = document.documentElement.getAttribute('data-i18n-title');
    if (titleKey) document.title = t(titleKey);
  }

  /**
   * Inject a small language-switcher button fixed to the top-right corner.
   * Shows the *other* locale label (clicking switches to it).
   */
  function injectSwitcher() {
    const locales = [
      { code: 'en',  label: 'EN'  },
      { code: 'ca',  label: 'CAT' },
      { code: 'es',  label: 'ES'  },
    ];

    const wrapper = document.createElement('div');
    wrapper.id = 'lang-switcher';
    Object.assign(wrapper.style, {
      position:  'fixed',
      top:       '1rem',
      right:     '1rem',
      zIndex:    '500',
      display:   'flex',
      gap:       '0.3rem',
    });

    locales.forEach(({ code, label }) => {
      const btn = document.createElement('button');
      btn.textContent = label;
      const active = code === locale;
      Object.assign(btn.style, {
        padding:       '0.3rem 0.6rem',
        fontSize:      '0.78rem',
        fontWeight:    active ? '700' : '400',
        background:    active ? '#222' : '#fff',
        color:         active ? '#fff' : '#333',
        border:        '1.5px solid #ccc',
        borderRadius:  '6px',
        cursor:        active ? 'default' : 'pointer',
        letterSpacing: '0.06em',
        boxShadow:     '0 1px 4px rgba(0,0,0,0.12)',
        fontFamily:    'inherit',
      });
      if (!active) btn.onclick = () => setLocale(code);
      wrapper.appendChild(btn);
    });

    document.body.appendChild(wrapper);
  }

  /* ── Auto-apply on DOMContentLoaded ────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', () => {
    applyTranslations();
    injectSwitcher();
  });

  /* ── Exports (globals, no module system needed) ─────────────────────────── */
  window.t                 = t;
  window.i18n              = { t, setLocale, getCurrentLocale, applyTranslations, injectSwitcher };
})();
