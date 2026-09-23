---
layout: default
---

# Manu Gupta

<p class="subtitle">Software engineer in New York. Interested in the fundamentals of large language models.</p>

<!-- Rewrite this in your own words. A few sentences: what you work on, what you want to work on. -->

I reproduce machine learning papers from scratch and write up what happened. The reports are below,
each with a repository you can run.

I work at Dub on AI agents and evaluation infrastructure. Before that, Palantir and a quant
development internship at Citadel. I studied computer science at Georgia Tech.

I am looking for research opportunities.

<p><a href="mailto:mgupta8143@gmail.com">Email</a> &middot;
<a href="https://github.com/mgupta8143">GitHub</a></p>

<hr>

## Reproductions

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

<hr>

## Experience

<ul class="plain">
  <li><span class="when">2025 &ndash;</span> Software Engineer, <strong>Dub</strong>, New York</li>
  <li><span class="when">2024 &ndash; 2025</span> Software Engineer, <strong>Palantir Technologies</strong>, New York</li>
  <li><span class="when">2024</span> Quantitative Developer Intern, <strong>Citadel</strong>, New York</li>
  <li><span class="when">2023</span> Software Engineer Intern, <strong>Palantir Technologies</strong>, Washington DC</li>
  <li><span class="when">2021 &ndash; 2024</span> B.S. Computer Science, <strong>Georgia Tech</strong></li>
</ul>
