---
layout: default
title: Reproductions
permalink: /reproductions/
---

# Reproductions

<ol class="entries">
{% assign entries = site.reproductions | sort: "date" | reverse %}
{% for entry in entries %}
  <li>
    <span class="entry-venue">{{ entry.paper }}{% if entry.status %} &middot; {{ entry.status }}{% endif %}</span>
    <span class="entry-title"><a href="{{ entry.url | relative_url }}">{{ entry.title }}</a></span>
    {% if entry.summary %}<p class="entry-note">{{ entry.summary }}</p>{% endif %}
    <span class="entry-links">
      <a href="{{ entry.url | relative_url }}">Report</a>
      {% if entry.code %}<a href="{{ entry.code }}">Code</a>{% endif %}
      {% if entry.paper_url %}<a href="{{ entry.paper_url }}">Paper</a>{% endif %}
    </span>
  </li>
{% endfor %}
</ol>
