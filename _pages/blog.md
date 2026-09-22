---
layout: page
title: Blog
permalink: /blog/
nav: true
nav_order: 1
---

{% if site.posts.size > 0 %}

<ul class="post-list">
  {% for post in site.posts %}
    <li>
      <h2><a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
      <p class="post-meta">{{ post.date | date: "%B %d, %Y" }}</p>
      {% if post.description %}<p>{{ post.description }}</p>{% endif %}
    </li>
  {% endfor %}
</ul>
{% else %}
No posts yet.
{% endif %}
