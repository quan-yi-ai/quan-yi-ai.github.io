import DefaultTheme from 'vitepress/theme'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp() {
    if (typeof window === 'undefined') return

    const installClickEffect = () => {
      if (document.documentElement.dataset.asbClicks === 'ready') return
      document.documentElement.dataset.asbClicks = 'ready'
      document.addEventListener('pointerdown', (event) => {
        if (event.button !== 0 || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
        if (event.pointerType === 'touch') return
        const burst = document.createElement('span')
        burst.className = 'asb-click-burst'
        burst.style.left = `${event.clientX}px`
        burst.style.top = `${event.clientY}px`
        burst.setAttribute('aria-hidden', 'true')
        document.body.appendChild(burst)
        burst.addEventListener('animationend', () => burst.remove(), { once: true })
      }, { passive: true })
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', installClickEffect, { once: true })
    } else {
      installClickEffect()
    }
  }
}
