---
layout: default
title: Reproductions
permalink: /reproductions/
---

# Reproductions

<p>Papers I have implemented from scratch, with the code and an account of what matched the
original and what did not.</p>

<ul class="bare">
{% assign entries = site.reproductions | sort: "date" | reverse %}
{% for entry in entries %}
  <li>
    <span class="item-title"><a href="{{ entry.url | relative_url }}">{{ entry.title }}</a></span><br>
    <span class="item-where">{{ entry.paper }}{% if entry.status %} &middot; {{ entry.status }}{% endif %}</span>
    {% if entry.summary %}<br>{{ entry.summary }}{% endif %}
    <div class="item-links">
      <a href="{{ entry.url | relative_url }}">Report</a>
      {% if entry.code %}<a href="{{ entry.code }}">Code</a>{% endif %}
      {% if entry.paper_url %}<a href="{{ entry.paper_url }}">Paper</a>{% endif %}
    </div>
  </li>
{% endfor %}
</ul>
