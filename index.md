---
layout: default
---

# Reproductions

I reproduce machine learning papers from scratch and write up what happened: the code, the
figures, and where my results diverged from the paper's. Each entry links to a repository you
can run yourself.

<ul class="entries">
{% assign entries = site.reproductions | sort: "date" | reverse %}
{% for entry in entries %}
  <li>
    <a class="entry" href="{{ entry.url | relative_url }}">
      <span class="entry-head">
        <span class="entry-title">{{ entry.title }}</span>
        {% if entry.status %}<span class="status {{ entry.status | slugify }}">{{ entry.status }}</span>{% endif %}
      </span>
      {% if entry.paper %}<span class="entry-paper">{{ entry.paper }}</span>{% endif %}
      {% if entry.summary %}<span class="entry-summary">{{ entry.summary }}</span>{% endif %}
    </a>
  </li>
{% endfor %}
</ul>
