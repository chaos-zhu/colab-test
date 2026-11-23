import { ref, watch } from 'vue'
import zh from './locales/zh'

// English source strings are the keys; a locale without an entry falls back to the English text.
const DICTIONARIES = { en: {}, zh }
export const LOCALES = [{ id: 'en', label: 'EN' }, { id: 'zh', label: '中文' }]
const STORAGE_KEY = 'h3.locale'

function initialLocale() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved in DICTIONARIES) return saved
  } catch { /* storage unavailable */ }
  return navigator.language?.toLowerCase().startsWith('zh') ? 'zh' : 'en'
}

export const locale = ref(initialLocale())
watch(locale, (value) => {
  try { localStorage.setItem(STORAGE_KEY, value) } catch { /* storage unavailable */ }
  document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en'
}, { immediate: true })

// Reading locale.value makes templates and computeds that call t() re-render on a switch.
// 'Text|context' tells apart words that translate differently; the context never shows in English.
export function t(text, params) {
  const message = DICTIONARIES[locale.value]?.[text] ?? text.split('|')[0]
  return params ? message.replace(/\{(\w+)\}/g, (m, key) => (key in params ? params[key] : m)) : message
}

export function dateLocale() {
  return locale.value === 'zh' ? 'zh-CN' : 'en-US'
}
