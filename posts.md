---
layout: home
title: Posts
permalink: /posts/
---

## 💻 Tech
<ul>
  {% assign tech_posts = site.categories.Tech | sort: "path" %}
  {% for post in tech_posts %}
    {% assign display_title = post.title | split: ":" | first %}
    <li>
      <a href="{{ post.url }}">{{ display_title }}</a> – {{ post.date | date: "%b %d, %Y" }}
    </li>
  {% endfor %}
</ul>

<br>

<!-- ## 🧠 Thinking
<ul>
  {% for post in site.categories.Thinking %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> – {{ post.date | date: "%b %d, %Y" }}
    </li>
  {% endfor %}
</ul> -->

<br>

## 🧬 Neuroscience
<ul>
  {% assign neuroscience_posts = site.categories.Neuroscience | sort: "path" %}
  {% for post in neuroscience_posts %}
    {% assign display_title = post.title | split: ":" | first %}
    <li>
      <a href="{{ post.url }}">{{ display_title }}</a> – {{ post.date | date: "%b %d, %Y" }}
    </li>
  {% endfor %}
</ul>
