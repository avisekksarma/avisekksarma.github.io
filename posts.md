---
layout: home
title: Posts
permalink: /posts/
---

## 💻 Tech Articles
<ul>
  {% for post in site.categories.Tech %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> – {{ post.date | date: "%b %d, %Y" }}
    </li>
  {% endfor %}
</ul>

<br>

## 🧠 Thinking
<ul>
  {% for post in site.categories.Thinking %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> – {{ post.date | date: "%b %d, %Y" }}
    </li>
  {% endfor %}
</ul>
