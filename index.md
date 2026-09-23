---
layout: default
---

# Manu Gupta

<p class="role">Software engineer, New York</p>

<!-- Replace this paragraph with your own. -->
I work at Dub on AI agents and evaluation infrastructure. Previously Palantir, and a quant
development internship at Citadel. B.S. in computer science from Georgia Tech. I am interested in
the fundamentals of large language models and I am looking for research opportunities.

<p><a href="mailto:mgupta8143@gmail.com">mgupta8143@gmail.com</a></p>

## Reproductions

<ul class="bare">
{% assign entries = site.reproductions | sort: "date" | reverse %}
{% for entry in entries %}
  <li><a href="{{ entry.url | relative_url }}">{{ entry.title }}</a>{% if entry.status %} <span class="date">({{ entry.status }})</span>{% endif %}</li>
{% endfor %}
</ul>

## Experience

<ul class="bare">
  <li><span class="date">2025&ndash;</span> Software Engineer, Dub</li>
  <li><span class="date">2024&ndash;25</span> Software Engineer, Palantir</li>
  <li><span class="date">2024</span> Quantitative Developer Intern, Citadel</li>
  <li><span class="date">2023</span> Software Engineer Intern, Palantir</li>
  <li><span class="date">2021&ndash;24</span> B.S. Computer Science, Georgia Tech</li>
</ul>
