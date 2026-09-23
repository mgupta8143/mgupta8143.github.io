---
layout: default
---

<div class="intro">
<div class="text">

<h1>Manu Gupta</h1>
<p class="role">Applied AI Engineer at Dub, New York</p>

<!-- Everything in this file is yours to rewrite. -->

<p><span class="label">Research interests.</span>
I am interested in the fundamentals of large language models: how a model stores information and
retrieves it later, which computations an architecture can and cannot express, and how training
and evaluation choices show up in behaviour. Memory is the thread through most of it, from
augmenting networks with addressable external memory to what attention does and does not retain
over long contexts. I work on this by reproducing papers from scratch: implementing them without
reading existing code, running the experiments, and writing down where the results agree with the
paper and where they do not.</p>

<p><span class="label">Current and previous work.</span>
I am an applied AI engineer at Dub, where I founded and led Arlo, an AI agent product, and built the
offline and online evaluation infrastructure behind it. Before that I was at Palantir
Technologies for a year working on geospatial services, and I spent a summer at Citadel as a
quantitative developer on the Credit Core team. I studied computer science at Georgia Tech.</p>

<p>I am looking for research opportunities. Email is the best way to reach me.</p>

</div>

<figure>
  <img src="{{ '/assets/profile.jpg' | relative_url }}" alt="Manu Gupta">
  <div class="contact">
    <a href="mailto:mgupta8143@gmail.com">Email</a><br>
    <a href="https://github.com/mgupta8143">GitHub</a><br>
    <a href="https://www.linkedin.com/in/mgupta8143">LinkedIn</a>
  </div>
</figure>
</div>

<h2>Selected reproductions</h2>

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
