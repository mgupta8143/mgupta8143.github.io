# mgupta8143.github.io

Personal research site: reproductions of machine learning papers.

## Adding a reproduction

Copy `_reproductions/_template.md` to `_reproductions/your-paper.md`, delete the
`published: false` line, and fill it in. It appears on the homepage and on /reproductions/
automatically, newest first.

The template has the section order I use: summary, the claim, setup with a paper-vs-mine table,
what I built, results, what matched, what did not, what cost me time, open questions, and how to
run it.

Figures go in `assets/` and are referenced as `/assets/name.png`. Diagrams can be drawn and saved
as images, or written inline in a ```mermaid code block, which renders in the browser.

## Front matter

| Field | Used for |
|---|---|
| `title` | heading and list entry |
| `paper`, `paper_url` | the citation and its link |
| `code` | link to the repository |
| `venue` | the small tag box at the left of a list entry |
| `authors`, `published_in` | the citation lines under the title |
| `status` | `in progress` or `done` |
| `date` | sort order |
| `summary` | one line on the list |

## Previewing locally (optional)

```sh
bundle install
bundle exec jekyll serve
```

GitHub builds the site on push, so this is only for checking changes before publishing.
