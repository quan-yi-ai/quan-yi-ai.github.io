import DefaultTheme from 'vitepress/theme'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp() {
    if (typeof window === 'undefined') return

    const installInteractions = () => {
      if (document.documentElement.dataset.asbInteractions === 'ready') return
      document.documentElement.dataset.asbInteractions = 'ready'

      // Champagne-gold four-point starburst. Deliberately restrained:
      // decorative pointer feedback only, no action interception or layout changes.
      const starPath = 'M0 -10 Q2.3 -2.3 10 0 Q2.3 2.3 0 10 Q-2.3 2.3 -10 0 Q-2.3 -2.3 0 -10Z'
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
      document.addEventListener('pointerdown', (event) => {
        if (event.button !== 0 || event.pointerType === 'touch' || reducedMotion.matches) return
        if (!(event.target instanceof Element)) return
        // Clicking a control should activate it, not cover it in decorative particles.
        if (event.target.closest('a, button, input, textarea, select, summary, [role="button"], .asb-image-lightbox')) return

        const burst = document.createElement('span')
        burst.className = 'asb-starburst'
        burst.setAttribute('aria-hidden', 'true')
        burst.style.left = `${event.clientX}px`
        burst.style.top = `${event.clientY}px`

        // A slightly irregular radial composition looks organic, not mechanical.
        const count = 8
        const start = Math.random() * 18
        for (let i = 0; i < count; i++) {
          const angle = (start + i * 45 + (Math.random() - 0.5) * 13) * Math.PI / 180
          const distance = 30 + Math.random() * 18
          const star = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
          star.setAttribute('viewBox', '-12 -12 24 24')
          star.setAttribute('class', 'asb-starburst-star')
          star.setAttribute('aria-hidden', 'true')
          star.style.setProperty('--dx', `${Math.cos(angle) * distance}px`)
          star.style.setProperty('--dy', `${Math.sin(angle) * distance}px`)
          star.style.setProperty('--turn', `${(Math.random() - 0.5) * 105}deg`)
          star.style.setProperty('--star-size', `${8 + Math.random() * 6}px`)
          star.style.setProperty('--delay', `${Math.floor(Math.random() * 65)}ms`)
          const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path')
          shape.setAttribute('d', starPath)
          star.appendChild(shape)
          burst.appendChild(star)
        }

        // A single, very small center sparkle marks the precise click location.
        const center = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
        center.setAttribute('viewBox', '-12 -12 24 24')
        center.setAttribute('class', 'asb-starburst-center')
        center.setAttribute('aria-hidden', 'true')
        const centerShape = document.createElementNS('http://www.w3.org/2000/svg', 'path')
        centerShape.setAttribute('d', starPath)
        center.appendChild(centerShape)
        burst.appendChild(center)
        document.body.appendChild(burst)

        burst.addEventListener('animationend', (e) => {
          if (e.target === center) burst.remove()
        })
      }, { passive: true })


      // Accessible lightbox for all article figures, including vector SVG.
      const dialog = document.createElement('dialog')
      dialog.className = 'asb-image-lightbox'
      dialog.setAttribute('aria-label', '图片放大预览')
      const close = document.createElement('button')
      close.type = 'button'
      close.className = 'asb-lightbox-close'
      close.setAttribute('aria-label', '关闭图片预览')
      close.textContent = '×'
      const image = document.createElement('img')
      image.className = 'asb-lightbox-image'
      image.alt = ''
      dialog.append(close, image)
      document.body.appendChild(dialog)

      close.addEventListener('click', () => dialog.close())
      dialog.addEventListener('click', (event) => {
        if (event.target === dialog) dialog.close()
      })
      dialog.addEventListener('close', () => image.removeAttribute('src'))
      document.addEventListener('click', (event) => {
        const target = event.target
        if (!(target instanceof Element)) return
        const trigger = target.closest('.asb-zoom-trigger')
        if (!trigger || dialog.open) return
        const original = trigger.querySelector('img')
        if (!original) return
        image.src = original.currentSrc || original.src
        image.alt = original.alt
        dialog.showModal()
      })
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', installInteractions, { once: true })
    } else {
      installInteractions()
    }
  }
}
