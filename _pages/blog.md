---
layout: default
permalink: /blog/
title: Blog
nav: true
nav_order: 7
pagination:
  enabled: false
---

<div class="post">

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% if blog_name_size > 0 or blog_description_size > 0 %}
  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>
{% endif %}

{% comment %}
  Sectioned blog index. Each post should match exactly one section via a distinguishing category.
{% endcomment %}
{% assign blog_sections = site.data.blog_sections %}

{% if blog_sections and blog_sections.size > 0 %}
  <nav class="tag-category-list" aria-label="Blog sections">
    <ul class="p-0 m-0">
      {% for section in blog_sections %}
        <li>
          <a href="#{{ section.id }}">{{ section.title }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </nav>
{% endif %}

{% for section in blog_sections %}
  {% assign section_category = section.category %}
  {% assign section_posts = site.posts | where_exp: "post", "post.categories contains section_category" %}

  {% if section_posts.size > 0 %}
    <section id="{{ section.id }}" class="blog-section" style="margin-top: 2.5rem;">
      <h2 style="margin-bottom: 0.35rem;">{{ section.title }}</h2>
      {% if section.description and section.description != "" %}
        <p class="post-meta" style="margin-bottom: 1.25rem;">{{ section.description }}</p>
      {% endif %}

      <ul class="post-list">
        {% for post in section_posts %}
          {% if post.external_source == blank %}
            {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
          {% else %}
            {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
          {% endif %}
          {% assign year = post.date | date: "%Y" %}
          {% assign tags = post.tags | join: "" %}
          {% assign categories = post.categories | join: "" %}

          <li>
            {% if post.thumbnail %}
              <div class="row">
                <div class="col-sm-9">
            {% endif %}

            <h3>
              {% if post.redirect == blank %}
                <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
              {% elsif post.redirect contains '://' %}
                <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
                <svg width="2rem" height="2rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                  <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
                </svg>
              {% else %}
                <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
              {% endif %}
            </h3>
            <p>{{ post.description }}</p>
            <p class="post-meta">
              {{ read_time }} min read &nbsp; &middot; &nbsp;
              {{ post.date | date: '%B %d, %Y' }}
              {% if post.external_source %}
                &nbsp; &middot; &nbsp; {{ post.external_source }}
              {% endif %}
              {% if post.series %}
                &nbsp; &middot; &nbsp; {{ post.series }}
              {% endif %}
            </p>
            <p class="post-tags">
              <a href="{{ year | prepend: '/blog/' | relative_url }}">
                <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
              </a>

              {% if tags != "" %}
                &nbsp; &middot; &nbsp;
                {% for tag in post.tags %}
                  <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">
                    <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}</a>
                  {% unless forloop.last %}&nbsp;{% endunless %}
                {% endfor %}
              {% endif %}

              {% if categories != "" %}
                &nbsp; &middot; &nbsp;
                {% for category in post.categories %}
                  <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">
                    <i class="fa-solid fa-tag fa-sm"></i> {{ category }}</a>
                  {% unless forloop.last %}&nbsp;{% endunless %}
                {% endfor %}
              {% endif %}
            </p>

            {% if post.thumbnail %}
                </div>
                <div class="col-sm-3">
                  <img class="card-img" src="{{ post.thumbnail | relative_url }}" style="object-fit: cover; height: 90%" alt="image">
                </div>
              </div>
            {% endif %}
          </li>
        {% endfor %}
      </ul>
    </section>

    {% unless forloop.last %}
      <hr>
    {% endunless %}
  {% endif %}
{% endfor %}

</div>
