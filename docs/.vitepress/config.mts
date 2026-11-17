import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI Systems & Beyond',
  description: 'Exploring Artificial Intelligence, Systems, and Beyond.',
  cleanUrls: true,
  themeConfig: {
    siteTitle: 'AI Systems & Beyond',
    nav: [
      { text: '首页', link: '/' },
      { text: '文章', link: '/posts/' },
      { text: '关于', link: '/about' },
      { text: 'GitHub', link: 'https://github.com/quan-yi-ai' }
    ],
    sidebar: {
      '/posts/': [
        { text: '技术文章', items: [
          { text: '文章索引', link: '/posts/' },
          { text: 'AI & Machine Learning', link: '/posts/ai' },
          { text: 'Systems & Computing', link: '/posts/systems' },
          { text: 'Engineering', link: '/posts/engineering' },
          { text: 'Research & Notes', link: '/posts/research' }
        ] }
      ]
    },
    socialLinks: [{ icon: 'github', link: 'https://github.com/quan-yi-ai' }],
    search: { provider: 'local' },
    footer: { message: 'Built with VitePress', copyright: 'AI Systems & Beyond' }
  }
})
