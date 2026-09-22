---
layout: page
title: Publications
permalink: /publications/
nav: true
nav_order: 2
---

{% capture paper_count %}{% bibliography_count %}{% endcapture %}
{% assign paper_count = paper_count | plus: 0 %}
{% if paper_count > 0 %}
{% include bib_search.liquid %}

  <div class="publications">
    {% bibliography %}
  </div>
{% else %}
Publications will be listed here.
{% endif %}
