---
layout: default
---

# Manu Gupta

<p class="subtitle">Software engineer. Reproducing machine learning papers, working toward research.</p>

I am interested in the fundamentals of large language models: how these systems store and retrieve
information, what their architectures can and cannot represent, and how training choices show up
in behaviour. The way I learn that is by reproducing papers from scratch, implementing them
without reading existing code, running the experiments, and writing down where my results agree
with the paper and where they don't. Those write-ups are below, each with a repository you can run.

I am currently a software engineer at **Dub** in New York, where I founded and led Arlo, a 0-to-1
AI agent initiative, and built the offline and online evaluation infrastructure that keeps it
compliant with FINRA and SEC rules. Before that I spent a year at **Palantir Technologies** on
mission-critical geospatial services, and a summer at **Citadel** as a quantitative developer
building data-persistence and risk tooling for researchers and traders. I studied computer
science at **Georgia Tech**, graduating in 2024.

I am looking for research opportunities in machine learning.

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
  <li><span class="when">2025 &ndash;</span> Software Engineer, <strong>Dub</strong>, New York. AI agents, evaluation infrastructure, GitOps provisioning.</li>
  <li><span class="when">2024 &ndash; 2025</span> Software Engineer, <strong>Palantir Technologies</strong>, New York. Geospatial services for edge deployment.</li>
  <li><span class="when">2024</span> Quantitative Developer Intern, <strong>Citadel</strong>, New York. Data persistence and risk tooling for the Credit Core team.</li>
  <li><span class="when">2023</span> Software Engineer Intern, <strong>Palantir Technologies</strong>, Washington DC. Real-time geotemporal streaming.</li>
  <li><span class="when">2021 &ndash; 2024</span> B.S. Computer Science, <strong>Georgia Institute of Technology</strong>.</li>
</ul>
