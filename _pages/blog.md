---
layout: default
permalink: /blog/
title: Writing
nav: true
nav_order: 7
description: Essays and technical notes on machine learning, databases, neuroscience, and AR/VR.
pagination:
  enabled: false
---

<div class="post writing-index">

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% if blog_name_size > 0 or blog_description_size > 0 %}
<header class="writing-header">
{% if blog_name_size > 0 %}
<h1>{{ site.blog_name }}</h1>
{% endif %}
{% if blog_description_size > 0 %}
<p>{{ site.blog_description }}</p>
{% endif %}
<p class="writing-note">
Many, though not all, parts of these writings are AI-assisted. I mainly use AI tools for the deeper essays—helping with diagrams, explanations, and organizing the notes and chats I’ve made while learning each topic into a single place. I only write about things I’ve actually read.

      </p>
    </header>

{% endif %}

{% assign blog_sections = site.data.blog_sections %}

{% if blog_sections and blog_sections.size > 0 %}
<nav class="writing-nav" aria-label="Writing sections">
{% for section in blog_sections %}
<a href="#{{ section.id }}">{{ section.title }}</a>
{% endfor %}
</nav>
{% endif %}

{% for section in blog_sections %}
{% if section.tag %}
{% assign section_posts = site.posts | where_exp: "post", "post.tags contains section.tag" | sort: "date" | reverse %}
{% else %}
{% assign section_posts = site.posts | where_exp: "post", "post.categories contains section.category" | sort: "date" | reverse %}
{% endif %}

    {% if section_posts.size > 0 %}
      <section id="{{ section.id }}" class="writing-section">
        <header class="writing-section-header">
          <h2>{{ section.title }}</h2>
          {% if section.description and section.description != "" %}
            <p>{{ section.description }}</p>
          {% endif %}
        </header>

        <ul class="writing-list">
          {% for post in section_posts %}
            {% if post.external_source == blank %}
              {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
            {% else %}
              {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
            {% endif %}
            {% if post.reading_time %}
              {% assign read_time = post.reading_time %}
            {% endif %}

            <li class="writing-item">
              <h3>
                {% if post.redirect == blank %}
                  <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
                {% elsif post.redirect contains '://' %}
                  <a href="{{ post.redirect }}" target="_blank" rel="noopener">{{ post.title }}</a>
                {% else %}
                  <a href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
                {% endif %}
              </h3>

              {% if post.description %}
                <p class="writing-excerpt">{{ post.description }}</p>
              {% endif %}

              <p class="writing-meta">
                <time datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: '%b %Y' }}</time>
                <span aria-hidden="true">·</span>
                <span>{{ read_time }} min</span>
                {% if post.series %}
                  <span aria-hidden="true">·</span>
                  <span>{{ post.series }}</span>
                {% endif %}
              </p>

              {% if post.tags and post.tags.size > 0 %}
                <ul class="writing-tags">
                  {% for tag in post.tags limit: 5 %}
                    <li>
                      <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag }}</a>
                    </li>
                  {% endfor %}
                </ul>
              {% endif %}
            </li>
          {% endfor %}
        </ul>
      </section>
    {% endif %}

{% endfor %}

</div>
