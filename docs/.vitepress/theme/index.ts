import DefaultTheme from 'vitepress/theme'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp() {
    if (typeof window === 'undefined') return

    const installInteractions = () => {
      if (document.documentElement.dataset.asbInteractions === 'ready') return
      document.documentElement.dataset.asbInteractions = 'ready'

      // The global pointer animation is paused pending a style choice.
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
