---
layout: home
title: Posts
permalink: /posts/
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> – {{ post.date | date: "%b %d, %Y" }}
    </li>
  {% endfor %}
</ul>
