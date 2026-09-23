# mgupta8143.github.io

Personal research site: reproductions of machine learning papers.

## Adding a reproduction

Create one Markdown file in `_reproductions/`, for example `_reproductions/alexnet.md`:

```markdown
---
title: AlexNet on ImageNet
paper: Krizhevsky et al. (2012)
paper_url: https://papers.nips.cc/paper/4824
code: https://github.com/mgupta8143/alexnet
status: in progress     # in progress | complete
date: 2026-10-01
summary: One sentence for the homepage.
---

Write the report here in Markdown. Images go in `assets/`:

![Learning curve](/assets/alexnet-curve.png)
```

It appears on the homepage automatically, newest first. Commit and push; GitHub Pages rebuilds
the site in about a minute.

## Previewing locally (optional)

```sh
bundle install
bundle exec jekyll serve
```

GitHub builds the site on push, so this is only for checking changes before publishing.
