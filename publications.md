---
layout: default
title: Publications
permalink: /publications/
---

<h1 style="font-size:2.4rem">Publications</h1>

{% assign papers = site.publications | sort: "date" | reverse %}
{% if papers.size > 0 %}
  {% for paper in papers %}
<div class="pub">
  <div class="tag">{{ paper.venue }}</div>
  <div class="body">
    <div class="title">{% if paper.pdf %}<a href="{{ paper.pdf }}">{{ paper.title }}</a>{% else %}{{ paper.title }}{% endif %}</div>
    <div class="who">{{ paper.authors }}</div>
    <div class="where">{{ paper.published_in }}</div>
    <div class="links">
      {% if paper.pdf %}<a href="{{ paper.pdf }}">PDF</a>{% endif %}
      {% if paper.code %}<a href="{{ paper.code }}">Code</a>{% endif %}
      {% if paper.arxiv %}<a href="{{ paper.arxiv }}">arXiv</a>{% endif %}
    </div>
  </div>
</div>
  {% endfor %}
{% else %}
<p class="subtitle">In progress. Current work is a research project with a PhD student at Columbia
University; write-ups of papers I have reimplemented are under
<a href="{{ '/reproductions/' | relative_url }}">Reproductions</a>.</p>
{% endif %}
