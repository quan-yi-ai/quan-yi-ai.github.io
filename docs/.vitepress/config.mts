import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI Systems & Beyond',
  description: 'Exploring Artificial Intelligence, Systems, and Beyond.',
  cleanUrls: true,
  markdown: { math: true },
  themeConfig: {
    siteTitle: 'AI Systems & Beyond',
    nav: [
      { text: '首页', link: '/' },
      { text: '文章', link: '/posts/' },
      { text: '关于', link: '/about' }
    ],
    sidebar: {
      '/posts/': [
        { text: '技术文章', items: [
          { text: '文章索引', link: '/posts/' },
          { text: 'TransferQueue', link: '/posts/transferqueue' },
          { text: '并行方式', link: '/posts/transformer-parallel-attention' },
          { text: '从TransferQueue重构理解python、cpp、Rust', link: '/posts/transferqueue-python-cpp-rust' },
          { text: 'AI & Machine Learning', link: '/posts/ai' },
          { text: 'Systems & Computing', link: '/posts/systems' },
          { text: 'Engineering', link: '/posts/engineering' },
          { text: 'Research & Notes', link: '/posts/research' }
        ] }
      ]
    },
    search: { provider: 'local' },
    footer: { message: 'Built with VitePress', copyright: 'AI Systems & Beyond' }
  }
})
