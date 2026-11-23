import DefaultTheme from 'vitepress/theme'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp() {
    if (typeof window === 'undefined') return

    const installInteractions = () => {
      if (document.documentElement.dataset.asbInteractions === 'ready') return
      document.documentElement.dataset.asbInteractions = 'ready'

      // A playful, short confetti pop: vivid paper bits burst radially from the click.
      // Keep it decorative and out of the way of real interactive controls.
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
      const confettiColors = ['#ff6b8e', '#ffd65c', '#55d9d0', '#9d91ff', '#ff9670', '#75d46d', '#7ebcff']
      document.addEventListener('pointerdown', (event) => {
        if (event.button !== 0 || event.pointerType === 'touch' || reducedMotion.matches) return
        if (!(event.target instanceof Element)) return
        if (event.target.closest('a, button, input, textarea, select, summary, [role="button"], .asb-image-lightbox')) return

        const burst = document.createElement('span')
        burst.className = 'asb-confetti-burst'
        burst.setAttribute('aria-hidden', 'true')
        burst.style.left = `${event.clientX}px`
        burst.style.top = `${event.clientY}px`

        const count = 16
        const baseAngle = Math.random() * Math.PI * 2
        for (let i = 0; i < count; i++) {
          const angle = baseAngle + (i / count) * Math.PI * 2 + (Math.random() - .5) * .22
          const distance = 34 + Math.random() * 37
          const piece = document.createElement('span')
          piece.className = 'asb-confetti-piece'
          piece.style.setProperty('--dx', `${Math.cos(angle) * distance}px`)
          piece.style.setProperty('--dy', `${Math.sin(angle) * distance + 11}px`)
          piece.style.setProperty('--turn', `${-230 + Math.random() * 460}deg`)
          piece.style.setProperty('--piece-color', confettiColors[i % confettiColors.length])
          piece.style.setProperty('--delay', `${Math.floor(Math.random() * 52)}ms`)
          piece.style.setProperty('--width', `${3 + Math.random() * 3}px`)
          piece.style.setProperty('--height', `${6 + Math.random() * 4}px`)
          if (i % 5 === 0) piece.classList.add('asb-confetti-dot')
          if (i % 4 === 0) piece.classList.add('asb-confetti-diamond')
          burst.appendChild(piece)
        }

        // Center flash is intentionally tiny; it doubles as the cleanup clock.
        const center = document.createElement('span')
        center.className = 'asb-confetti-center'
        burst.appendChild(center)
        document.body.appendChild(burst)
        center.addEventListener('animationend', () => burst.remove(), { once: true })
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
