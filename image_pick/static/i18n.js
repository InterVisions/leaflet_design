/**
 * Lightweight i18n module.
 * Loads the active locale JSON synchronously so t() is available
 * before any other script runs. Locale is persisted in localStorage.
 */
(function () {
  const SUPPORTED = ['en', 'es'];
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
    const other = locale === 'en' ? 'es' : 'en';
    const label = locale === 'en' ? 'ES' : 'EN';

    const btn = document.createElement('button');
    btn.id        = 'lang-switcher';
    btn.textContent = label;
    btn.title     = locale === 'en' ? 'Cambiar a español' : 'Switch to English';
    btn.onclick   = () => setLocale(other);

    Object.assign(btn.style, {
      position:     'fixed',
      top:          '1rem',
      right:        '1rem',
      zIndex:       '500',
      padding:      '0.3rem 0.75rem',
      fontSize:     '0.78rem',
      fontWeight:   '700',
      background:   '#fff',
      color:        '#333',
      border:       '1.5px solid #ccc',
      borderRadius: '6px',
      cursor:       'pointer',
      letterSpacing:'0.06em',
      boxShadow:    '0 1px 4px rgba(0,0,0,0.12)',
      fontFamily:   'inherit',
    });

    document.body.appendChild(btn);
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
